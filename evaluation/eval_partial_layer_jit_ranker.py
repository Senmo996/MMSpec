"""Offline falsification test for staged Partial-Layer JIT drafting.

The diagnostic uses hidden states already produced by a strict target-model
reference trace.  At every generation state it applies the target model's final
normalization and LM-head rows to post-root hidden states after selected decoder
depths.  This answers whether a genuinely contextual, shallow target prefix can
rank prompt-native visual candidates before we implement a partial-forward
runtime.

Hidden states from the full reference forward are an oracle trace, not a free
runtime mechanism.  The summary therefore reports an optimistic layer-FLOP
lower bound for a width-3 recursive tree and never reports measured drafting
latency.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.eval_hidden_state_transport_ranker import _paired_control_metrics
from evaluation.eval_sam_grounded_mmspec import _select_topic_indices
from evaluation.eval_transition_kernel_ranker import (
    STATIC_CONFIG_KEY,
    _coverage_metrics,
    _distribution,
    _question_lexical_token_ids,
    _run_reference_forward,
    _seed_transition_rows,
    _synchronize_if_cuda,
    _target_ranks_from_grid,
)
from evaluation.utils import build_prompt, load_mmspec_data
from method.sam_grounded.controller import VisualGroundingCalibrator
from method.sam_grounded.counterfactual_probes import build_visual_probe_layout
from method.sam_grounded.spec_model import SpecModel
from method.sam_grounded.visual_lexical_inventory import (
    build_visual_lexical_inventories,
    interleave_rankings,
)
from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids


PARENT_PROJECTION_KEY = "parent_projection"


def _parse_nonnegative_ints(value: str) -> List[int]:
    result: List[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid integer layer depth: {raw}"
            ) from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError("layer depths must be non-negative")
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("at least one layer depth is required")
    return result


def _layer_key(depth: int) -> str:
    return f"partial_layer_{int(depth):02d}"


def _candidate_scores(
    hidden_state: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    output_projection,
) -> torch.Tensor:
    """Project one hidden state onto only the requested vocabulary rows."""

    if hidden_state.ndim != 1:
        raise ValueError("hidden_state must be one-dimensional")
    candidate_token_ids = candidate_token_ids.to(
        device=output_projection.weight.device, dtype=torch.long
    )
    weight = output_projection.weight.index_select(0, candidate_token_ids)
    bias = getattr(output_projection, "bias", None)
    if bias is not None:
        bias = bias.index_select(0, candidate_token_ids)
    return F.linear(hidden_state.to(device=weight.device, dtype=weight.dtype), weight, bias)


def _normalized_layer_states(
    hidden_states: Sequence[torch.Tensor],
    layer_depths: Sequence[int],
    *,
    num_layers: int,
    final_norm,
) -> Dict[str, torch.Tensor]:
    """Return logit-lens states with correct Qwen hidden-state semantics.

    ``hidden_states[d]`` is the residual stream after ``d`` decoder layers.
    Entries below the full depth have not received the model's final RMSNorm;
    the last entry has already received it and must not be normalized twice.
    """

    if len(hidden_states) != int(num_layers) + 1:
        raise ValueError(
            f"expected {int(num_layers) + 1} hidden-state entries, "
            f"found {len(hidden_states)}"
        )
    result = {}
    for depth in layer_depths:
        depth = int(depth)
        if not 0 <= depth <= int(num_layers):
            raise ValueError(
                f"layer depth {depth} is outside [0, {int(num_layers)}]"
            )
        state = hidden_states[depth][0, 0]
        if depth < int(num_layers):
            state = final_norm(state)
        result[_layer_key(depth)] = state
    return result


def _consecutive_path_metrics(
    records: Sequence[dict],
    pool_name: str,
    config_key: str,
    width: int,
) -> dict:
    """Measure a true same-ranker two-token path on the greedy trace.

    The first state must be a high-visual state where token recycling has no
    exact row.  The second rank comes from the next state's post-token shallow
    hidden state, which is the oracle branch corresponding to the actual first
    target.  It is therefore an accuracy upper bound for recursive Partial-Layer
    JIT, not an end-to-end acceptance measurement.
    """

    by_state = {
        (int(record["sample_index"]), int(record["step_index"])): record
        for record in records
    }
    eligible = []
    for record in records:
        if not record.get("high_visual_state", False):
            continue
        if record.get("baseline_row_available", False):
            continue
        if pool_name not in record.get("pools", {}):
            continue
        next_record = by_state.get(
            (int(record["sample_index"]), int(record["step_index"]) + 1)
        )
        if next_record is None or pool_name not in next_record.get("pools", {}):
            continue
        eligible.append((record, next_record))

    first_hits = 0
    second_hits = 0
    path_hits = 0
    for record, next_record in eligible:
        first_rank = record["pools"][pool_name]["target_ranks"].get(config_key)
        second_rank = next_record["pools"][pool_name]["target_ranks"].get(
            config_key
        )
        first_hit = first_rank is not None and int(first_rank) <= int(width)
        second_hit = second_rank is not None and int(second_rank) <= int(width)
        first_hits += int(first_hit)
        second_hits += int(second_hit)
        path_hits += int(first_hit and second_hit)

    count = len(eligible)
    return {
        "num_states_with_next": count,
        "width": int(width),
        "first_hits": first_hits,
        "first_hit_rate": first_hits / count if count else None,
        "second_hits": second_hits,
        "second_hit_rate": second_hits / count if count else None,
        "path_hits": path_hits,
        "path_hit_rate": path_hits / count if count else None,
        "second_given_first_rate": path_hits / first_hits if first_hits else None,
    }


def _optimistic_cost_model(
    path: dict,
    *,
    depth: int,
    num_layers: int,
    branch_width: int,
) -> dict:
    """Best-case layer-FLOP break-even screen for a two-stage draft tree."""

    one_hop_cost = float(depth) / float(num_layers)
    recursive_cost = (
        float(depth) * float(1 + int(branch_width)) / float(num_layers)
    )
    first_rate = path.get("first_hit_rate")
    path_rate = path.get("path_hit_rate")
    one_hop_margin = (
        float(first_rate) - one_hop_cost if first_rate is not None else None
    )
    recursive_saved = (
        float(first_rate) + float(path_rate)
        if first_rate is not None and path_rate is not None
        else None
    )
    recursive_margin = (
        recursive_saved - recursive_cost if recursive_saved is not None else None
    )
    return {
        "depth": int(depth),
        "num_model_layers": int(num_layers),
        "branch_width": int(branch_width),
        "one_hop_partial_full_forward_equivalents": one_hop_cost,
        "one_hop_optimistic_saved_full_forward_equivalents": first_rate,
        "one_hop_optimistic_margin": one_hop_margin,
        "recursive_partial_tokens": 1 + int(branch_width),
        "recursive_partial_full_forward_equivalents": recursive_cost,
        "recursive_optimistic_saved_full_forward_equivalents": recursive_saved,
        "recursive_optimistic_margin": recursive_margin,
        "recursive_break_even": (
            recursive_margin is not None and recursive_margin > 0.0
        ),
        "assumptions": (
            "Linear cost in executed layer-token pairs; perfect batching; no "
            "LM-head, launch, cache, tree-build, or verifier-reuse overhead."
        ),
    }


def _gate_components(
    metric: dict,
    path: dict,
    specificity: dict,
    cost: dict,
    *,
    min_gate_states: int,
    min_top3_hit_rate: float,
    min_path_hit_rate: float,
) -> dict:
    delta = specificity.get("observed_minus_control_pp")
    result = {
        "enough_states": int(metric.get("num_states") or 0) >= int(min_gate_states),
        "top3_pass": (
            metric.get("top3_hit_rate") is not None
            and metric["top3_hit_rate"] >= float(min_top3_hit_rate)
        ),
        "path_pass": (
            path.get("path_hit_rate") is not None
            and path["path_hit_rate"] >= float(min_path_hit_rate)
        ),
        "paired_specificity_pass": delta is not None and float(delta) > 0.0,
        "optimistic_break_even_pass": bool(cost.get("recursive_break_even", False)),
    }
    result["pass"] = all(result.values())
    return result


def _select_depth(
    metrics: Dict[str, dict],
    paths: Dict[str, dict],
    paired_specificity: Dict[str, dict],
    layer_depths: Sequence[int],
    *,
    num_layers: int,
    max_selection_depth: int,
    branch_width: int,
    min_gate_states: int,
    min_top3_hit_rate: float,
    min_path_hit_rate: float,
) -> tuple[str, bool, List[dict], Dict[str, dict], Dict[str, dict]]:
    selectable = [
        int(depth)
        for depth in layer_depths
        if 0 < int(depth) <= int(max_selection_depth)
    ]
    if not selectable:
        raise ValueError("no positive layer depth is within max_selection_depth")

    costs: Dict[str, dict] = {}
    gates: Dict[str, dict] = {}
    eligible: List[dict] = []
    for depth in selectable:
        key = _layer_key(depth)
        cost = _optimistic_cost_model(
            paths[key],
            depth=depth,
            num_layers=num_layers,
            branch_width=branch_width,
        )
        gate = _gate_components(
            metrics[key],
            paths[key],
            paired_specificity[key],
            cost,
            min_gate_states=min_gate_states,
            min_top3_hit_rate=min_top3_hit_rate,
            min_path_hit_rate=min_path_hit_rate,
        )
        costs[key] = cost
        gates[key] = gate
        if gate["pass"]:
            eligible.append(
                {
                    "config": key,
                    "depth": depth,
                    "top3_hit_rate": metrics[key]["top3_hit_rate"],
                    "path_hit_rate": paths[key]["path_hit_rate"],
                    "specificity_delta_pp": paired_specificity[key][
                        "observed_minus_control_pp"
                    ],
                    "optimistic_margin": cost["recursive_optimistic_margin"],
                }
            )

    if eligible:
        winner = max(
            eligible,
            key=lambda row: (
                row["optimistic_margin"],
                row["path_hit_rate"],
                row["top3_hit_rate"],
                row["specificity_delta_pp"],
                -row["depth"],
            ),
        )
        return winner["config"], True, eligible, costs, gates

    def diagnostic_value(depth: int):
        key = _layer_key(depth)
        gate = gates[key]
        metric = metrics[key]
        path = paths[key]
        specificity = paired_specificity[key]
        cost = costs[key]
        return (
            sum(bool(value) for name, value in gate.items() if name != "pass"),
            metric.get("top3_hit_rate") or 0.0,
            path.get("path_hit_rate") or 0.0,
            specificity.get("observed_minus_control_pp") or float("-inf"),
            cost.get("recursive_optimistic_margin") or float("-inf"),
            -depth,
        )

    selected_depth = max(selectable, key=diagnostic_value)
    return _layer_key(selected_depth), False, [], costs, gates


def summarize_records(
    records: Sequence[dict],
    inventory_records: Sequence[dict],
    layer_depths: Sequence[int],
    num_layers: int,
    args,
) -> dict:
    layer_keys = [_layer_key(depth) for depth in layer_depths]
    config_keys = [STATIC_CONFIG_KEY, PARENT_PROJECTION_KEY, *layer_keys]
    subsets = {
        "all_row_absent": [
            row for row in records if not row.get("baseline_row_available", False)
        ],
        "high_visual_row_absent": [
            row
            for row in records
            if row.get("high_visual_state", False)
            and not row.get("baseline_row_available", False)
        ],
        "low_visual_row_absent": [
            row
            for row in records
            if not row.get("high_visual_state", False)
            and not row.get("baseline_row_available", False)
        ],
    }
    pool_names = ("observed_visual", "mismatched_visual", "text_control")
    metrics = {
        subset_name: {
            pool_name: {
                key: _coverage_metrics(rows, pool_name, key)
                for key in config_keys
            }
            for pool_name in pool_names
        }
        for subset_name, rows in subsets.items()
    }
    consecutive_paths = {
        pool_name: {
            f"width{width}": {
                key: _consecutive_path_metrics(records, pool_name, key, width)
                for key in config_keys
            }
            for width in (1, 2, 3, 4)
        }
        for pool_name in pool_names
    }
    paired_controls = {
        "mismatched_visual": {
            key: _paired_control_metrics(records, key) for key in config_keys
        },
        "text_control": {
            key: _paired_control_metrics(
                records, key, control_pool="text_control"
            )
            for key in config_keys
        },
    }
    primary = metrics["high_visual_row_absent"]["observed_visual"]
    primary_paths = consecutive_paths["observed_visual"]["width3"]
    selected, selected_was_eligible, eligible, costs, gates = _select_depth(
        primary,
        primary_paths,
        paired_controls["mismatched_visual"],
        layer_depths,
        num_layers=num_layers,
        max_selection_depth=args.max_selection_depth,
        branch_width=args.branch_width,
        min_gate_states=args.min_gate_states,
        min_top3_hit_rate=args.min_top3_hit_rate,
        min_path_hit_rate=args.min_path_hit_rate,
    )

    all_depth_costs = {}
    for depth in layer_depths:
        key = _layer_key(depth)
        all_depth_costs[key] = _optimistic_cost_model(
            primary_paths[key],
            depth=depth,
            num_layers=num_layers,
            branch_width=args.branch_width,
        )

    full_key = _layer_key(num_layers)
    full_metric = primary[full_key]
    full_projection_valid = (
        full_metric.get("top1_given_pool_rate") is None
        or full_metric["top1_given_pool_rate"] == 1.0
    )
    if not full_projection_valid:
        raise RuntimeError(
            "full-depth candidate projection is inconsistent with reference logits"
        )

    return {
        "method": "Staged Partial-Layer JIT offline kill test",
        "diagnostic_only": True,
        "runtime_partial_forward_implemented": False,
        "num_samples": len(inventory_records),
        "num_states": len(records),
        "configuration": {
            "topic_offset": args.topic_offset,
            "max_new_token": args.max_new_token,
            "candidate_pool_size": args.candidate_pool_size,
            "visual_threshold": args.visual_threshold,
            "candidate_row_budget": args.candidate_row_budget,
            "layer_depths": list(layer_depths),
            "num_model_layers": int(num_layers),
            "max_selection_depth": args.max_selection_depth,
            "branch_width": args.branch_width,
            "attention_implementation": args.attn_implementation,
        },
        "layer_configs": [
            {
                "key": _layer_key(depth),
                "depth": int(depth),
                "fraction_of_model_layers": float(depth) / float(num_layers),
                "selectable": 0 < int(depth) <= int(args.max_selection_depth),
            }
            for depth in layer_depths
        ],
        "subsets": {name: len(rows) for name, rows in subsets.items()},
        "metrics": metrics,
        "consecutive_path_metrics": consecutive_paths,
        "paired_controls": paired_controls,
        "optimistic_cost_model": all_depth_costs,
        "latency_ms": {
            "reference_full_forward_with_hidden_states": _distribution(
                row["reference_forward_ms"] for row in records
            ),
            "all_depth_candidate_projection_sweep": _distribution(
                row["projection_sweep_ms"] for row in records
            ),
            "inventory_construction": _distribution(
                row["inventory_construction_ms"] for row in inventory_records
            ),
            "note": (
                "Projection sweep latency does not include partial transformer "
                "layers. Full-forward hidden states are trace-only."
            ),
        },
        "sanity_checks": {
            "full_depth_key": full_key,
            "full_depth_top1_given_pool_rate": full_metric.get(
                "top1_given_pool_rate"
            ),
            "full_depth_projection_matches_reference": full_projection_valid,
        },
        "within_split_selection": {
            "selected_config": selected,
            "selected_depth": int(selected.rsplit("_", 1)[1]),
            "selected_was_eligible": selected_was_eligible,
            "eligible_config_count": len(eligible),
            "eligible_configs": eligible,
            "selection_protocol": (
                "Among predeclared positive depths <= max_selection_depth, "
                "require enough states, top3 and same-depth path thresholds, "
                "positive paired image specificity, and positive optimistic "
                "recursive break-even margin; then maximize margin."
            ),
            "selected_metrics": primary[selected],
            "selected_consecutive_path_width3": primary_paths[selected],
            "selected_paired_specificity": paired_controls[
                "mismatched_visual"
            ][selected],
            "selected_optimistic_cost": costs[selected],
            "selected_gate_components": gates[selected],
            "static_metrics": primary[STATIC_CONFIG_KEY],
            "parent_projection_metrics": primary[PARENT_PROJECTION_KEY],
            "full_depth_metrics": full_metric,
            "full_depth_consecutive_path_width3": primary_paths[full_key],
            "warning": (
                "Freeze offset-0 depth before interpreting offsets 1/2. The "
                "cost screen is optimistic and passing it authorizes only a "
                "small runtime prototype."
            ),
        },
        "local_gate": {
            "decision": "go" if selected_was_eligible else "no-go",
            "pass": selected_was_eligible,
            "thresholds": {
                "min_gate_states": args.min_gate_states,
                "min_top3_hit_rate": args.min_top3_hit_rate,
                "min_path_hit_rate": args.min_path_hit_rate,
                "positive_paired_specificity_required": True,
                "positive_optimistic_recursive_margin_required": True,
                "max_selection_depth": args.max_selection_depth,
            },
        },
    }


@torch.inference_mode()
def evaluate(args) -> None:
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path="",
        total_token=args.candidate_row_budget,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    base_model = model.base_model
    embedding_layer = base_model.get_input_embeddings()
    output_projection = base_model.get_output_embeddings()
    # Local checkpoints created with older Transformers releases expose the
    # text decoder directly as ``base_model.model``.  Newer releases wrap it in
    # ``base_model.model.language_model``.  Support both without changing the
    # loaded model or its cache implementation.
    language_model = getattr(base_model.model, "language_model", base_model.model)
    final_norm = language_model.norm
    num_layers = len(language_model.layers)

    requested_depths = sorted(set(int(depth) for depth in args.layer_depths))
    if any(depth > num_layers for depth in requested_depths):
        raise ValueError(
            f"requested depths {requested_depths} exceed model depth {num_layers}"
        )
    layer_depths = sorted(set([*requested_depths, num_layers]))
    if not any(0 < depth <= args.max_selection_depth for depth in layer_depths):
        raise ValueError("no requested positive depth is selectable")

    stop_token_ids = _collect_stop_token_ids(model.tokenizer, base_model)
    past_key_values, _past_data, current_length_data = initialize_past_key_values(
        base_model
    )
    data = load_mmspec_data(args.data_folder)
    if args.samples_per_topic is not None:
        data = data.select(
            _select_topic_indices(data, args.samples_per_topic, args.topic_offset)
        )
    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "ranker_records.jsonl"
    inventory_path = output_root / "inventory_records.jsonl"
    records: List[dict] = []
    inventory_records: List[dict] = []
    excluded_token_ids = set(int(token) for token in model.tokenizer.all_special_ids)
    for attribute in ("image_token_id", "video_token_id"):
        token_id = getattr(base_model.config, attribute, None)
        if token_id is not None:
            excluded_token_ids.add(int(token_id))

    print(f"Loaded {len(data)} samples")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"topic_offset={args.topic_offset}")
    print(f"num_model_layers={num_layers}")
    print(f"layer_depths={layer_depths}")
    print(f"records_path={records_path}")
    print(f"inventory_path={inventory_path}")

    previous_visual_pool: Optional[List[int]] = None
    previous_question_id: Optional[str] = None
    with records_path.open("w", encoding="utf-8") as records_file, inventory_path.open(
        "w", encoding="utf-8"
    ) as inventory_file:
        for sample_index, sample in enumerate(
            tqdm(data, desc=f"Partial-layer JIT offset {args.topic_offset}")
        ):
            current_length_data.zero_()
            model_inputs = build_prompt(sample, args)
            input_ids = model_inputs["input_ids"]
            prompt_length = int(input_ids.shape[1])
            visual_mask = model._build_visual_token_mask(input_ids).detach()
            layout = build_visual_probe_layout(
                visual_mask,
                model_inputs.get("image_grid_thw"),
                int(base_model.config.vision_config.spatial_merge_size),
                args.inventory_regions,
            )
            init_output = base_model(
                **dict(model_inputs),
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
            )
            observed_lengths = current_length_data.unique().tolist()
            if observed_lengths != [prompt_length]:
                raise RuntimeError(
                    f"prefill KV lengths {observed_lengths} do not match prompt {prompt_length}"
                )
            current_length_data.fill_(prompt_length)
            prompt_hidden = init_output.hidden_states[-1][0]
            prompt_logits = init_output.logits[0]
            calibrator = VisualGroundingCalibrator.from_prompt(
                prompt_hidden, visual_mask
            )
            question = sample.get("turns", [sample.get("prompt", "")])[0]
            question_token_ids = _question_lexical_token_ids(
                model.tokenizer, question
            )

            device = prompt_logits.device
            _synchronize_if_cuda(device)
            inventory_started = time.perf_counter()
            inventories = build_visual_lexical_inventories(
                prompt_logits,
                visual_mask,
                layout,
                question_token_ids,
                args.candidate_pool_size,
                excluded_token_ids=sorted(excluded_token_ids),
                valid_vocab_size=min(
                    len(model.tokenizer), int(prompt_logits.shape[-1])
                ),
            )
            _synchronize_if_cuda(device)
            inventory_latency = 1000.0 * (time.perf_counter() - inventory_started)
            observed_pool = [
                int(token)
                for token in inventories["visual_max"][: args.candidate_pool_size]
            ]
            text_pool = [
                int(token)
                for token in inventories["text_mean_control"][
                    : args.candidate_pool_size
                ]
            ]
            hybrid_pool = interleave_rankings(
                [observed_pool, text_pool], args.candidate_pool_size
            )
            pool_candidates: Dict[str, List[int]] = {
                "observed_visual": observed_pool,
                "text_control": text_pool,
                "hybrid_language": hybrid_pool,
            }
            if previous_visual_pool is not None:
                pool_candidates["mismatched_visual"] = previous_visual_pool

            candidate_tensors = {
                pool_name: torch.tensor(
                    candidates, dtype=torch.long, device=output_projection.weight.device
                )
                for pool_name, candidates in pool_candidates.items()
            }
            inventory_record = {
                "question_id": sample["id"],
                "topic": sample.get("topic", "unknown"),
                "category": sample.get("category", "unknown"),
                "sample_index": sample_index,
                "topic_offset": args.topic_offset,
                "prompt_length": prompt_length,
                "num_visual_tokens": int(visual_mask.sum().item()),
                "inventory_construction_ms": inventory_latency,
                "observed_visual_pool": observed_pool,
                "text_control_pool": text_pool,
                "hybrid_language_pool": hybrid_pool,
                "mismatched_visual_pool": previous_visual_pool,
                "mismatched_question_id": previous_question_id,
            }
            inventory_records.append(inventory_record)
            inventory_file.write(json.dumps(inventory_record) + "\n")
            inventory_file.flush()

            transition_rows = _seed_transition_rows(
                input_ids[0], prompt_logits, args.candidate_row_budget
            )
            root_token = int(prompt_logits[-1].argmax().item())
            parent_hidden = prompt_hidden[-1]
            if root_token in stop_token_ids:
                previous_visual_pool = observed_pool
                previous_question_id = str(sample["id"])
                continue

            for step_index in range(max(args.max_new_token - 1, 0)):
                prefix_length = int(current_length_data[0].item())
                baseline_previous = transition_rows.get(root_token)

                _synchronize_if_cuda(device)
                reference_started = time.perf_counter()
                reference_output = _run_reference_forward(
                    base_model,
                    root_token,
                    prefix_length,
                    past_key_values,
                    current_length_data,
                )
                _synchronize_if_cuda(device)
                reference_latency = 1000.0 * (
                    time.perf_counter() - reference_started
                )
                reference_logits = reference_output.logits[0, 0]
                post_root_hidden = reference_output.hidden_states[-1][0, 0]
                target_token = int(reference_logits.argmax().item())
                grounding_score = calibrator.score(post_root_hidden)

                _synchronize_if_cuda(device)
                projection_started = time.perf_counter()
                normalized_states = _normalized_layer_states(
                    reference_output.hidden_states,
                    layer_depths,
                    num_layers=num_layers,
                    final_norm=final_norm,
                )
                pools_payload = {}
                for pool_name, candidates in pool_candidates.items():
                    candidate_ids = candidate_tensors[pool_name]
                    score_grid = {
                        PARENT_PROJECTION_KEY: _candidate_scores(
                            parent_hidden, candidate_ids, output_projection
                        ),
                        **{
                            key: _candidate_scores(
                                state, candidate_ids, output_projection
                            )
                            for key, state in normalized_states.items()
                        },
                    }
                    ranks = _target_ranks_from_grid(
                        score_grid, candidate_ids, target_token
                    )
                    try:
                        static_rank = candidates.index(target_token) + 1
                    except ValueError:
                        static_rank = None
                    pools_payload[pool_name] = {
                        "candidate_count": len(candidates),
                        "target_in_pool": static_rank is not None,
                        "static_rank": static_rank,
                        "target_ranks": {
                            STATIC_CONFIG_KEY: static_rank,
                            **ranks,
                        },
                    }
                _synchronize_if_cuda(device)
                projection_latency = 1000.0 * (
                    time.perf_counter() - projection_started
                )

                record = {
                    "question_id": sample["id"],
                    "topic": sample.get("topic", "unknown"),
                    "category": sample.get("category", "unknown"),
                    "sample_index": sample_index,
                    "topic_offset": args.topic_offset,
                    "step_index": step_index,
                    "prompt_length": prompt_length,
                    "prefix_kv_length": prefix_length,
                    "root_token": root_token,
                    "target_token": target_token,
                    "grounding_score": float(grounding_score),
                    "high_visual_state": grounding_score >= args.visual_threshold,
                    "baseline_row_available": baseline_previous is not None,
                    "baseline_candidates": (
                        list(baseline_previous) if baseline_previous is not None else []
                    ),
                    "reference_forward_ms": reference_latency,
                    "projection_sweep_ms": projection_latency,
                    "pools": pools_payload,
                }
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
                records_file.flush()

                transition_rows[root_token] = [
                    int(token)
                    for token in reference_logits.topk(
                        args.candidate_row_budget
                    ).indices.tolist()
                ]
                parent_hidden = post_root_hidden
                root_token = target_token
                if target_token in stop_token_ids:
                    break

            previous_visual_pool = observed_pool
            previous_question_id = str(sample["id"])

    summary = summarize_records(
        records, inventory_records, layer_depths, num_layers, args
    )
    summary["records_path"] = str(records_path.resolve())
    summary["inventory_records_path"] = str(inventory_path.resolve())
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    print(
        json.dumps(
            {
                "method": summary["method"],
                "num_samples": summary["num_samples"],
                "num_states": summary["num_states"],
                "subsets": summary["subsets"],
                "sanity_checks": summary["sanity_checks"],
                "within_split_selection": summary["within_split_selection"],
                "local_gate": summary["local_gate"],
                "latency_ms": summary["latency_ms"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary_path={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged Partial-Layer JIT offline kill test"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--samples-per-topic", type=int, default=1)
    parser.add_argument("--topic-offset", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=64)
    parser.add_argument("--inventory-regions", type=int, default=4)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--candidate-row-budget", type=int, default=4)
    parser.add_argument("--visual-threshold", type=float, default=0.55)
    parser.add_argument(
        "--layer-depths",
        type=_parse_nonnegative_ints,
        default=_parse_nonnegative_ints("0,1,2,4,6,8,12,16,20,24,28"),
    )
    parser.add_argument("--max-selection-depth", type=int, default=8)
    parser.add_argument("--branch-width", type=int, default=3)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-gate-states", type=int, default=20)
    parser.add_argument("--min-top3-hit-rate", type=float, default=0.25)
    parser.add_argument("--min-path-hit-rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.samples_per_topic is not None and args.samples_per_topic <= 0:
        parser.error("--samples-per-topic must be positive")
    if args.topic_offset < 0:
        parser.error("--topic-offset must be non-negative")
    if args.max_new_token <= 1:
        parser.error("--max-new-token must be greater than one")
    if args.candidate_pool_size <= 0 or args.candidate_row_budget <= 0:
        parser.error("candidate pool and row budgets must be positive")
    if args.max_selection_depth <= 0:
        parser.error("--max-selection-depth must be positive")
    if args.branch_width <= 0:
        parser.error("--branch-width must be positive")
    if args.min_gate_states <= 0:
        parser.error("--min-gate-states must be positive")
    if not 0.0 <= args.min_top3_hit_rate <= 1.0:
        parser.error("--min-top3-hit-rate must be in [0, 1]")
    if not 0.0 <= args.min_path_hit_rate <= 1.0:
        parser.error("--min-path-hit-rate must be in [0, 1]")
    args.model = args.base_model_path
    evaluate(args)


if __name__ == "__main__":
    main()
