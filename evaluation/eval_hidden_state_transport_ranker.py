"""Falsification-first evaluation of prompt-local hidden-state transport."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.eval_sam_grounded_mmspec import _select_topic_indices
from evaluation.eval_transition_kernel_ranker import (
    STATIC_CONFIG_KEY,
    _coverage_metrics,
    _distribution,
    _parse_floats,
    _parse_positive_ints,
    _path_metrics,
    _question_lexical_token_ids,
    _run_reference_forward,
    _seed_transition_rows,
    _synchronize_if_cuda,
    _target_ranks_from_grid,
)
from evaluation.utils import build_prompt, load_mmspec_data
from method.sam_grounded.controller import VisualGroundingCalibrator
from method.sam_grounded.counterfactual_probes import build_visual_probe_layout
from method.sam_grounded.hidden_state_transport import (
    HiddenStateTransportBank,
    HiddenStateTransportConfig,
    SOURCE_SCOPES,
    TRANSPORT_MODES,
    build_transport_config_grid,
)
from method.sam_grounded.spec_model import SpecModel
from method.sam_grounded.visual_lexical_inventory import (
    build_visual_lexical_inventories,
    interleave_rankings,
)
from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids


PARENT_PROJECTION_KEY = "parent_projection"
ORACLE_POSTROOT_KEY = "oracle_postroot_projection"
RECURSIVE_PATH_POLICIES = (
    "bridge_only",
    "visual_hst",
    "text_hst",
    "hybrid_hst",
    "bridge2_text1",
    "text2_bridge1",
    "bridge2_hybrid1",
    "hybrid2_bridge1",
)


def _parse_strings(value: str) -> List[str]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in result:
            result.append(item)
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _paired_control_metrics(
    records: Sequence[dict],
    config_key: str,
    *,
    observed_pool: str = "observed_visual",
    control_pool: str = "mismatched_visual",
    width: int = 3,
) -> dict:
    rows = [
        row
        for row in records
        if row.get("high_visual_state", False)
        and not row.get("baseline_row_available", False)
        and observed_pool in row.get("pools", {})
        and control_pool in row.get("pools", {})
    ]
    observed_hits = 0
    control_hits = 0
    observed_only = 0
    control_only = 0
    both = 0
    neither = 0
    observed_pool_hits = 0
    control_pool_hits = 0
    for row in rows:
        observed_rank = row["pools"][observed_pool]["target_ranks"].get(
            config_key
        )
        control_rank = row["pools"][control_pool]["target_ranks"].get(
            config_key
        )
        observed_pool_hits += int(observed_rank is not None)
        control_pool_hits += int(control_rank is not None)
        observed_hit = observed_rank is not None and observed_rank <= width
        control_hit = control_rank is not None and control_rank <= width
        observed_hits += int(observed_hit)
        control_hits += int(control_hit)
        observed_only += int(observed_hit and not control_hit)
        control_only += int(control_hit and not observed_hit)
        both += int(observed_hit and control_hit)
        neither += int(not observed_hit and not control_hit)
    count = len(rows)
    return {
        "num_paired_states": count,
        "width": int(width),
        "observed_hits": observed_hits,
        "control_hits": control_hits,
        "observed_hit_rate": observed_hits / count if count else None,
        "control_hit_rate": control_hits / count if count else None,
        "observed_minus_control_pp": (
            100.0 * (observed_hits - control_hits) / count if count else None
        ),
        "observed_pool_recall": observed_pool_hits / count if count else None,
        "control_pool_recall": control_pool_hits / count if count else None,
        "observed_minus_control_pool_recall_pp": (
            100.0 * (observed_pool_hits - control_pool_hits) / count
            if count
            else None
        ),
        "observed_only_hits": observed_only,
        "control_only_hits": control_only,
        "both_hits": both,
        "neither_hits": neither,
    }


def _select_config(
    metrics: Dict[str, dict],
    paths: Dict[str, dict],
    paired_specificity: Dict[str, dict],
    config_keys: Sequence[str],
    *,
    min_top3_hit_rate: float,
    min_path_hit_rate: float,
) -> tuple[str, bool, List[str]]:
    eligible = []
    for key in config_keys:
        row = metrics[key]
        path = paths[key]
        specificity = paired_specificity[key]
        if (
            row.get("top3_hit_rate") is not None
            and row["top3_hit_rate"] >= min_top3_hit_rate
            and path.get("path_hit_rate") is not None
            and path["path_hit_rate"] >= min_path_hit_rate
            and specificity.get("observed_minus_control_pp") is not None
            and specificity["observed_minus_control_pp"] > 0.0
        ):
            eligible.append(key)

    candidates = eligible if eligible else list(config_keys)

    def value(key: str):
        row = metrics[key]
        path = paths[key]
        specificity = paired_specificity[key]
        return (
            row.get("top3_hit_rate") or 0.0,
            path.get("path_hit_rate") or 0.0,
            specificity.get("observed_minus_control_pp") or float("-inf"),
            row.get("top1_hit_rate") or 0.0,
            row.get("mrr") or 0.0,
            key,
        )

    return max(candidates, key=value), bool(eligible), eligible


def _ranked_candidate_ids(
    bank: HiddenStateTransportBank,
    hidden_state: torch.Tensor,
    *,
    width: int,
    prior_weight: float,
) -> List[int]:
    projected = bank.project_hidden(hidden_state)
    score = (
        (1.0 - float(prior_weight)) * projected
        + float(prior_weight) * bank.candidate_prior
    )
    indices = score.topk(min(int(width), score.numel())).indices
    return [
        int(token)
        for token in bank.candidate_token_ids.index_select(0, indices).tolist()
    ]


def _fuse_rankings(
    primary: Sequence[int],
    secondary: Sequence[int],
    *,
    primary_slots: int,
    width: int,
) -> List[int]:
    selected = []
    seen = set()
    for token in list(primary)[: int(primary_slots)]:
        token = int(token)
        if token not in seen:
            selected.append(token)
            seen.add(token)
    for token in secondary:
        token = int(token)
        if token not in seen:
            selected.append(token)
            seen.add(token)
        if len(selected) >= int(width):
            return selected
    for token in primary[int(primary_slots) :]:
        token = int(token)
        if token not in seen:
            selected.append(token)
            seen.add(token)
        if len(selected) >= int(width):
            break
    return selected


def _build_recursive_paths(
    *,
    config: HiddenStateTransportConfig,
    observed_bank: HiddenStateTransportBank,
    text_bank: HiddenStateTransportBank,
    hybrid_bank: HiddenStateTransportBank,
    embedding_layer,
    query_token_vector: torch.Tensor,
    query_context_hidden: torch.Tensor,
    transition_rows: Dict[int, List[int]],
    width: int,
) -> dict:
    predicted_root_hidden = observed_bank.transport_hidden(
        query_token_vector=query_token_vector,
        query_context_hidden=query_context_hidden,
        config=config,
    )
    first_scores = observed_bank.score(
        query_token_vector=query_token_vector,
        query_context_hidden=query_context_hidden,
        config=config,
    )
    first_indices = first_scores.topk(min(int(width), first_scores.numel())).indices
    first_candidates = [
        int(token)
        for token in observed_bank.candidate_token_ids.index_select(
            0, first_indices
        ).tolist()
    ]
    branches = {}
    for first_token in first_candidates:
        token_tensor = torch.tensor(
            [first_token],
            dtype=torch.long,
            device=query_token_vector.device,
        )
        first_token_vector = embedding_layer(token_tensor)[0]
        predicted_first_hidden = observed_bank.transport_hidden(
            query_token_vector=first_token_vector,
            query_context_hidden=predicted_root_hidden,
            config=config,
        )
        visual = _ranked_candidate_ids(
            observed_bank,
            predicted_first_hidden,
            width=width,
            prior_weight=config.visual_weight,
        )
        text = _ranked_candidate_ids(
            text_bank,
            predicted_first_hidden,
            width=width,
            prior_weight=config.visual_weight,
        )
        hybrid = _ranked_candidate_ids(
            hybrid_bank,
            predicted_first_hidden,
            width=width,
            prior_weight=config.visual_weight,
        )
        bridge = [
            int(token)
            for token in (transition_rows.get(first_token) or [])[: int(width)]
        ]
        branches[str(first_token)] = {
            "bridge_only": bridge,
            "visual_hst": visual,
            "text_hst": text,
            "hybrid_hst": hybrid,
            "bridge2_text1": _fuse_rankings(
                bridge, text, primary_slots=2, width=width
            ),
            "text2_bridge1": _fuse_rankings(
                text, bridge, primary_slots=2, width=width
            ),
            "bridge2_hybrid1": _fuse_rankings(
                bridge, hybrid, primary_slots=2, width=width
            ),
            "hybrid2_bridge1": _fuse_rankings(
                hybrid, bridge, primary_slots=2, width=width
            ),
        }
    return {
        "width": int(width),
        "first_candidates": first_candidates,
        "branches": branches,
    }


def _recursive_path_metrics(
    records: Sequence[dict], config_key: str, policy: str
) -> dict:
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
        recursive = record.get("recursive_paths", {}).get(config_key)
        if recursive is None:
            continue
        next_record = by_state.get(
            (int(record["sample_index"]), int(record["step_index"]) + 1)
        )
        if next_record is not None:
            eligible.append((record, next_record, recursive))
    first_hits = 0
    path_hits = 0
    for record, next_record, recursive in eligible:
        first_token = int(record["target_token"])
        first_hit = first_token in {
            int(token) for token in recursive["first_candidates"]
        }
        branch = recursive["branches"].get(str(first_token), {})
        second_candidates = branch.get(policy, [])
        second_hit = int(next_record["target_token"]) in {
            int(token) for token in second_candidates
        }
        first_hits += int(first_hit)
        path_hits += int(first_hit and second_hit)
    count = len(eligible)
    return {
        "num_states_with_next": count,
        "policy": policy,
        "first_hits": first_hits,
        "first_hit_rate": first_hits / count if count else None,
        "path_hits": path_hits,
        "path_hit_rate": path_hits / count if count else None,
        "second_given_first_rate": path_hits / first_hits if first_hits else None,
    }


def _select_recursive_policy(
    *,
    primary_metrics: Dict[str, dict],
    paired_specificity: Dict[str, dict],
    recursive_metrics: Dict[str, Dict[str, dict]],
    config_keys: Sequence[str],
    min_top3_hit_rate: float,
    min_path_hit_rate: float,
) -> tuple[Optional[str], Optional[str], List[dict]]:
    eligible = []
    for key in config_keys:
        top3 = primary_metrics[key].get("top3_hit_rate")
        specificity = paired_specificity[key].get("observed_minus_control_pp")
        for policy, path in recursive_metrics[key].items():
            path_rate = path.get("path_hit_rate")
            if (
                top3 is not None
                and top3 >= min_top3_hit_rate
                and specificity is not None
                and specificity > 0.0
                and path_rate is not None
                and path_rate >= min_path_hit_rate
            ):
                eligible.append(
                    {
                        "config": key,
                        "policy": policy,
                        "top3_hit_rate": top3,
                        "path_hit_rate": path_rate,
                        "specificity_delta_pp": specificity,
                    }
                )
    if not eligible:
        return None, None, []
    selected = max(
        eligible,
        key=lambda row: (
            row["path_hit_rate"],
            row["top3_hit_rate"],
            row["specificity_delta_pp"],
            row["config"],
            row["policy"],
        ),
    )
    return selected["config"], selected["policy"], eligible


def summarize_records(
    records: Sequence[dict],
    inventory_records: Sequence[dict],
    configs: Sequence[HiddenStateTransportConfig],
    args,
) -> dict:
    kernel_keys = [config.key for config in configs]
    config_keys = [
        STATIC_CONFIG_KEY,
        PARENT_PROJECTION_KEY,
        ORACLE_POSTROOT_KEY,
        *kernel_keys,
    ]
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
                config_key: _coverage_metrics(subset, pool_name, config_key)
                for config_key in config_keys
            }
            for pool_name in pool_names
        }
        for subset_name, subset in subsets.items()
    }
    path_metrics = {
        pool_name: {
            f"width{width}": {
                config_key: _path_metrics(
                    records, pool_name, config_key, width
                )
                for config_key in config_keys
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
    recursive_path_metrics = {
        key: {
            policy: _recursive_path_metrics(records, key, policy)
            for policy in RECURSIVE_PATH_POLICIES
        }
        for key in args.recursive_configs
    }
    primary_metrics = metrics["high_visual_row_absent"]["observed_visual"]
    primary_paths = path_metrics["observed_visual"]["width3"]
    selected, has_eligible, eligible = _select_config(
        primary_metrics,
        primary_paths,
        paired_controls["mismatched_visual"],
        kernel_keys,
        min_top3_hit_rate=args.min_top3_hit_rate,
        min_path_hit_rate=args.min_path_hit_rate,
    )
    selected_metrics = primary_metrics[selected]
    static_metrics = primary_metrics[STATIC_CONFIG_KEY]
    selected_path = primary_paths[selected]
    selected_specificity = paired_controls["mismatched_visual"][selected]
    recursive_config, recursive_policy, recursive_eligible = (
        _select_recursive_policy(
            primary_metrics=primary_metrics,
            paired_specificity=paired_controls["mismatched_visual"],
            recursive_metrics=recursive_path_metrics,
            config_keys=args.recursive_configs,
            min_top3_hit_rate=args.min_top3_hit_rate,
            min_path_hit_rate=args.min_path_hit_rate,
        )
    )
    top3_gain_pp = (
        100.0
        * (
            (selected_metrics.get("top3_hit_rate") or 0.0)
            - (static_metrics.get("top3_hit_rate") or 0.0)
        )
        if selected_metrics.get("num_states")
        else None
    )

    return {
        "method": "Prompt-local Hidden-State Transport kill test",
        "diagnostic_only": True,
        "num_samples": len(inventory_records),
        "num_states": len(records),
        "configuration": {
            "topic_offset": args.topic_offset,
            "max_new_token": args.max_new_token,
            "candidate_pool_size": args.candidate_pool_size,
            "visual_threshold": args.visual_threshold,
            "candidate_row_budget": args.candidate_row_budget,
            "token_weights": args.token_weights,
            "neighbors": args.neighbors,
            "temperatures": args.temperatures,
            "transport_modes": args.transport_modes,
            "source_scopes": args.source_scopes,
            "visual_weights": args.visual_weights,
            "num_transport_configs": len(configs),
            "default_latency_config": args.default_latency_config,
            "recursive_configs": args.recursive_configs,
            "recursive_width": args.recursive_width,
            "attention_implementation": args.attn_implementation,
        },
        "transport_configs": [
            {
                "key": config.key,
                "token_weight": config.token_weight,
                "neighbors": config.neighbors,
                "temperature": config.temperature,
                "transport_mode": config.transport_mode,
                "source_scope": config.source_scope,
                "visual_weight": config.visual_weight,
            }
            for config in configs
        ],
        "subsets": {name: len(rows) for name, rows in subsets.items()},
        "metrics": metrics,
        "path_metrics": path_metrics,
        "paired_controls": paired_controls,
        "recursive_path_metrics": recursive_path_metrics,
        "latency_ms": {
            "default_single_config_observed_pool": _distribution(
                row["ranker_latency_ms"]
                for row in records
                if row.get("ranker_latency_ms") is not None
            ),
            "inventory_construction": _distribution(
                row["inventory_construction_ms"] for row in inventory_records
            ),
            "recursive_path_construction_by_config": {
                key: _distribution(
                    row.get("recursive_latency_ms", {}).get(key)
                    for row in records
                    if row.get("recursive_latency_ms", {}).get(key) is not None
                )
                for key in args.recursive_configs
            },
        },
        "within_split_selection": {
            "selected_config": selected,
            "selection_protocol": (
                "Require top3>=threshold, path@3>=threshold, and positive "
                "paired observed-vs-mismatched specificity; maximize top3, "
                "then path, specificity, top1, and MRR."
            ),
            "eligible_config_count": len(eligible),
            "eligible_configs": eligible,
            "selected_was_eligible": has_eligible,
            "selected_metrics": selected_metrics,
            "static_metrics": static_metrics,
            "parent_projection_metrics": primary_metrics[PARENT_PROJECTION_KEY],
            "oracle_postroot_metrics": primary_metrics[ORACLE_POSTROOT_KEY],
            "selected_path_width3": selected_path,
            "selected_paired_specificity": selected_specificity,
            "top3_gain_pp": top3_gain_pp,
            "warning": (
                "Freeze offset-0 selection before interpreting offsets 1/2. "
                "Oracle post-root projection is an upper bound only."
            ),
        },
        "local_gate": {
            "decision": "go" if has_eligible else "no-go",
            "pass": has_eligible,
            "thresholds": {
                "min_top3_hit_rate": args.min_top3_hit_rate,
                "min_path_hit_rate": args.min_path_hit_rate,
                "positive_paired_specificity_required": True,
            },
        },
        "recursive_selection": {
            "selected_config": recursive_config,
            "selected_policy": recursive_policy,
            "selected_was_eligible": recursive_config is not None,
            "eligible_action_count": len(recursive_eligible),
            "eligible_actions": recursive_eligible,
            "selection_protocol": (
                "For the predeclared recursive configs, require top3 and "
                "paired-specificity gates, require recursive path@3>=threshold, "
                "then maximize recursive path, top3, and specificity."
            ),
            "selected_metrics": (
                primary_metrics[recursive_config]
                if recursive_config is not None
                else None
            ),
            "selected_paired_specificity": (
                paired_controls["mismatched_visual"][recursive_config]
                if recursive_config is not None
                else None
            ),
            "selected_recursive_path": (
                recursive_path_metrics[recursive_config][recursive_policy]
                if recursive_config is not None and recursive_policy is not None
                else None
            ),
        },
        "recursive_local_gate": {
            "decision": "go" if recursive_config is not None else "no-go",
            "pass": recursive_config is not None,
        },
    }


def _make_bank(
    *,
    candidates: Sequence[int],
    output_projection,
    prompt_token_ids: torch.Tensor,
    prompt_token_embeddings: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    visual_mask: torch.Tensor,
    excluded_token_ids: Sequence[int],
) -> HiddenStateTransportBank:
    candidate_ids = torch.tensor(
        [int(token) for token in candidates],
        dtype=torch.long,
        device=prompt_hidden_states.device,
    )
    return HiddenStateTransportBank.from_prompt(
        candidate_token_ids=candidate_ids,
        output_projection_weight=output_projection.weight,
        output_projection_bias=getattr(output_projection, "bias", None),
        prompt_token_ids=prompt_token_ids,
        prompt_token_embeddings=prompt_token_embeddings,
        prompt_hidden_states=prompt_hidden_states,
        visual_mask=visual_mask,
        excluded_token_ids=excluded_token_ids,
    )


@torch.inference_mode()
def evaluate(args) -> None:
    configs = build_transport_config_grid(
        args.token_weights,
        args.neighbors,
        args.temperatures,
        args.transport_modes,
        args.source_scopes,
        args.visual_weights,
    )
    config_by_key = {config.key: config for config in configs}
    if args.default_latency_config not in config_by_key:
        raise ValueError(
            "--default-latency-config must be present in the configured grid"
        )
    missing_recursive = [
        key for key in args.recursive_configs if key not in config_by_key
    ]
    if missing_recursive:
        raise ValueError(
            f"--recursive-configs are absent from the configured grid: {missing_recursive}"
        )
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
    print(f"num_transport_configs={len(configs)}")
    print(f"records_path={records_path}")
    print(f"inventory_path={inventory_path}")

    previous_visual_pool: Optional[List[int]] = None
    previous_question_id: Optional[str] = None
    with records_path.open("w", encoding="utf-8") as records_file, inventory_path.open(
        "w", encoding="utf-8"
    ) as inventory_file:
        for sample_index, sample in enumerate(
            tqdm(data, desc=f"Hidden-state transport offset {args.topic_offset}")
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

            prompt_token_embeddings = embedding_layer(input_ids[0])
            banks = {
                pool_name: _make_bank(
                    candidates=candidates,
                    output_projection=output_projection,
                    prompt_token_ids=input_ids[0],
                    prompt_token_embeddings=prompt_token_embeddings,
                    prompt_hidden_states=prompt_hidden,
                    visual_mask=visual_mask,
                    excluded_token_ids=sorted(excluded_token_ids),
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
                "prompt_transition_count": banks["observed_visual"].source_count,
                "post_visual_transition_count": banks[
                    "observed_visual"
                ].post_visual_source_count,
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
                query_token_id = torch.tensor(
                    [root_token], dtype=torch.long, device=input_ids.device
                )
                query_token_vector = embedding_layer(query_token_id)[0]
                score_grids: Dict[str, Dict[str, torch.Tensor]] = {}
                recursive_paths = {}
                recursive_latency_ms = {}
                ranker_latency_ms = None
                if baseline_previous is None:
                    observed_bank = banks["observed_visual"]
                    _synchronize_if_cuda(device)
                    latency_started = time.perf_counter()
                    observed_bank.score(
                        query_token_vector=query_token_vector,
                        query_context_hidden=parent_hidden,
                        config=config_by_key[args.default_latency_config],
                    )
                    _synchronize_if_cuda(device)
                    ranker_latency_ms = 1000.0 * (
                        time.perf_counter() - latency_started
                    )
                    for recursive_key in args.recursive_configs:
                        _synchronize_if_cuda(device)
                        recursive_started = time.perf_counter()
                        recursive_paths[recursive_key] = _build_recursive_paths(
                            config=config_by_key[recursive_key],
                            observed_bank=banks["observed_visual"],
                            text_bank=banks["text_control"],
                            hybrid_bank=banks["hybrid_language"],
                            embedding_layer=embedding_layer,
                            query_token_vector=query_token_vector,
                            query_context_hidden=parent_hidden,
                            transition_rows=transition_rows,
                            width=args.recursive_width,
                        )
                        _synchronize_if_cuda(device)
                        recursive_latency_ms[recursive_key] = 1000.0 * (
                            time.perf_counter() - recursive_started
                        )
                    for pool_name in (
                        "observed_visual",
                        "mismatched_visual",
                        "text_control",
                    ):
                        if pool_name not in banks:
                            continue
                        bank = banks[pool_name]
                        score_grids[pool_name] = {
                            PARENT_PROJECTION_KEY: bank.project_hidden(parent_hidden),
                            **bank.score_grid(
                                query_token_vector=query_token_vector,
                                query_context_hidden=parent_hidden,
                                configs=configs,
                            ),
                        }

                reference_output = _run_reference_forward(
                    base_model,
                    root_token,
                    prefix_length,
                    past_key_values,
                    current_length_data,
                )
                reference_logits = reference_output.logits[0, 0]
                post_root_hidden = reference_output.hidden_states[-1][0, 0]
                target_token = int(reference_logits.argmax().item())
                grounding_score = calibrator.score(post_root_hidden)
                bridge_previous = transition_rows.get(target_token)

                pools_payload = {}
                for pool_name, grid in score_grids.items():
                    bank = banks[pool_name]
                    ranks = _target_ranks_from_grid(
                        grid, bank.candidate_token_ids, target_token
                    )
                    oracle_rank = _target_ranks_from_grid(
                        {ORACLE_POSTROOT_KEY: bank.project_hidden(post_root_hidden)},
                        bank.candidate_token_ids,
                        target_token,
                    )[ORACLE_POSTROOT_KEY]
                    candidates = pool_candidates[pool_name]
                    try:
                        static_rank = candidates.index(target_token) + 1
                    except ValueError:
                        static_rank = None
                    ranks = {
                        STATIC_CONFIG_KEY: static_rank,
                        ORACLE_POSTROOT_KEY: oracle_rank,
                        **ranks,
                    }
                    pools_payload[pool_name] = {
                        "candidate_count": len(candidates),
                        "target_in_pool": static_rank is not None,
                        "static_rank": static_rank,
                        "target_ranks": ranks,
                    }

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
                    "bridge_candidates_before": (
                        list(bridge_previous) if bridge_previous is not None else []
                    ),
                    "ranker_latency_ms": ranker_latency_ms,
                    "recursive_latency_ms": recursive_latency_ms,
                    "pools": pools_payload,
                    "recursive_paths": recursive_paths,
                }
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
                records_file.flush()

                for bank in banks.values():
                    bank.append(
                        source_token_vector=query_token_vector,
                        source_context_hidden=parent_hidden,
                        source_post_hidden=post_root_hidden,
                    )
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

    summary = summarize_records(records, inventory_records, configs, args)
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
                "latency_ms": summary["latency_ms"],
                "within_split_selection": summary["within_split_selection"],
                "local_gate": summary["local_gate"],
                "recursive_selection": summary["recursive_selection"],
                "recursive_local_gate": summary["recursive_local_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary_path={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt-local Hidden-State Transport kill test"
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
        "--token-weights", type=_parse_floats, default=_parse_floats("0,0.5,1")
    )
    parser.add_argument(
        "--neighbors", type=_parse_positive_ints, default=_parse_positive_ints("4,16")
    )
    parser.add_argument(
        "--temperatures", type=_parse_floats, default=_parse_floats("0.05,0.15")
    )
    parser.add_argument(
        "--transport-modes",
        type=_parse_strings,
        default=_parse_strings("delta_half,delta_full,post_state"),
    )
    parser.add_argument(
        "--source-scopes",
        type=_parse_strings,
        default=_parse_strings("all_text,post_visual_text"),
    )
    parser.add_argument(
        "--visual-weights", type=_parse_floats, default=_parse_floats("0,0.25,0.5")
    )
    parser.add_argument(
        "--default-latency-config",
        default="tw0.50-k16-t0.15-mdf-spostvis-vw0.25",
    )
    parser.add_argument(
        "--recursive-configs",
        type=_parse_strings,
        default=_parse_strings(
            "tw0.50-k16-t0.05-mdf-sall-vw0.00,"
            "tw0.50-k16-t0.05-mdf-spostvis-vw0.00"
        ),
    )
    parser.add_argument("--recursive-width", type=int, default=3)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-top3-hit-rate", type=float, default=0.15)
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
    if args.recursive_width <= 0:
        parser.error("--recursive-width must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.token_weights):
        parser.error("--token-weights values must be in [0, 1]")
    if any(value <= 0.0 for value in args.temperatures):
        parser.error("--temperatures values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.visual_weights):
        parser.error("--visual-weights values must be in [0, 1]")
    if any(mode not in TRANSPORT_MODES for mode in args.transport_modes):
        parser.error(f"--transport-modes must be drawn from {TRANSPORT_MODES}")
    if any(scope not in SOURCE_SCOPES for scope in args.source_scopes):
        parser.error(f"--source-scopes must be drawn from {SOURCE_SCOPES}")
    args.model = args.base_model_path
    evaluate(args)


if __name__ == "__main__":
    main()
