"""Falsification-first test for prompt-native visual lexical inventories.

This diagnostic asks whether logits already produced at multimodal prefill can
provide useful, history-independent draft candidates.  It measures two things:

1. a fixed-width policy that replaces recycled-row tails with inventory tokens;
2. an oracle upper bound that preserves every baseline hit and asks whether the
   target occurs anywhere in a bounded inventory pool.

The oracle is intentionally not an end-to-end decoder.  Its role is to reject a
weak candidate source before spending time on tree construction and throughput
evaluation.  A cyclic mismatched-image inventory is evaluated as a specificity
control on runs containing at least two samples.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.eval_sam_grounded_mmspec import _select_topic_indices
from evaluation.utils import build_prompt, load_mmspec_data
from method.sam_grounded.controller import VisualGroundingCalibrator
from method.sam_grounded.counterfactual_probes import (
    build_visual_probe_layout,
    make_repeated_position_ids,
)
from method.sam_grounded.spec_model import SpecModel
from method.sam_grounded.visual_lexical_inventory import (
    build_visual_lexical_inventories,
    fuse_equal_budget_candidates,
)
from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids


def _parse_positive_ints(value: str) -> List[int]:
    values = []
    for item in value.split(","):
        parsed = int(item.strip())
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _percentile(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> Dict[str, Optional[float]]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _question_lexical_token_ids(tokenizer, question: str) -> List[int]:
    token_ids = tokenizer(question, add_special_tokens=False).input_ids
    result = []
    for token_id in token_ids:
        decoded = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if any(character.isalnum() for character in decoded):
            result.append(int(token_id))
    return result


def _seed_transition_rows(
    prompt_ids: torch.Tensor, prompt_logits: torch.Tensor, max_budget: int
) -> Dict[int, List[int]]:
    top_tokens = prompt_logits.topk(max_budget, dim=-1).indices.detach().to("cpu")
    rows: Dict[int, List[int]] = {}
    for position, token in enumerate(prompt_ids.detach().to("cpu").tolist()):
        rows[int(token)] = [
            int(value) for value in top_tokens[position, :max_budget].tolist()
        ]
    return rows


def _run_reference_forward(
    base_model,
    root_token: int,
    prefix_length: int,
    past_key_values,
    current_length_data: torch.Tensor,
):
    device = base_model.device
    query = torch.tensor([[root_token]], dtype=torch.long, device=device)
    position_ids = make_repeated_position_ids(
        prefix_length, 1, base_model.rope_deltas, device
    )
    cache_position = torch.tensor([prefix_length], dtype=torch.long, device=device)
    output = base_model(
        input_ids=query,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=True,
        use_cache=True,
        output_hidden_states=True,
    )
    current_length_data.fill_(prefix_length + 1)
    return output


def _subset_rows(states: Sequence[dict]) -> Dict[str, List[dict]]:
    return {
        "all_states": list(states),
        "high_visual_states": [row for row in states if row["high_visual_state"]],
        "low_visual_states": [row for row in states if not row["high_visual_state"]],
        "baseline_available_states": [
            row for row in states if row["baseline_row_available"]
        ],
        "baseline_unavailable_states": [
            row for row in states if not row["baseline_row_available"]
        ],
        "high_visual_baseline_available_states": [
            row
            for row in states
            if row["high_visual_state"] and row["baseline_row_available"]
        ],
        "high_visual_baseline_unavailable_states": [
            row
            for row in states
            if row["high_visual_state"] and not row["baseline_row_available"]
        ],
    }


def _coverage_summary(
    states: Sequence[dict],
    inventory_by_sample: Dict[int, dict],
    source: str,
    pool_size: int,
    candidate_budget: int,
    inventory_slots: int,
    *,
    sample_remap: Optional[Dict[int, int]] = None,
) -> dict:
    count = len(states)
    if count == 0:
        return {
            "num_states": 0,
            "baseline_coverage": None,
            "fixed_budget_coverage": None,
            "fixed_budget_gain_pp": None,
            "oracle_coverage": None,
            "oracle_gain_pp": None,
            "added_hits": 0,
            "lost_hits": 0,
            "oracle_new_hits": 0,
        }

    baseline_hits = 0
    fixed_hits = 0
    oracle_hits = 0
    inventory_hits = 0
    added_hits = 0
    lost_hits = 0
    changed_rows = 0
    for state in states:
        sample_index = int(state["sample_index"])
        inventory_sample = (
            sample_remap.get(sample_index, sample_index)
            if sample_remap is not None
            else sample_index
        )
        inventory = inventory_by_sample[inventory_sample]["inventories"][source][
            :pool_size
        ]
        baseline = [
            int(token)
            for token in state["baseline_candidates"][:candidate_budget]
        ]
        fixed = fuse_equal_budget_candidates(
            baseline, inventory, candidate_budget, inventory_slots
        )
        target = int(state["target_token"])
        baseline_hit = target in baseline
        fixed_hit = target in fixed
        inventory_hit = target in inventory
        oracle_hit = baseline_hit or inventory_hit
        baseline_hits += int(baseline_hit)
        fixed_hits += int(fixed_hit)
        inventory_hits += int(inventory_hit)
        oracle_hits += int(oracle_hit)
        added_hits += int(not baseline_hit and fixed_hit)
        lost_hits += int(baseline_hit and not fixed_hit)
        changed_rows += int(fixed != baseline)

    return {
        "num_states": count,
        "candidate_budget": candidate_budget,
        "inventory_slots": inventory_slots,
        "inventory_pool_size": pool_size,
        "baseline_hits": baseline_hits,
        "fixed_budget_hits": fixed_hits,
        "oracle_hits": oracle_hits,
        "inventory_target_hits": inventory_hits,
        "baseline_coverage": baseline_hits / count,
        "fixed_budget_coverage": fixed_hits / count,
        "fixed_budget_gain_pp": 100.0 * (fixed_hits - baseline_hits) / count,
        "inventory_target_recall": inventory_hits / count,
        "oracle_coverage": oracle_hits / count,
        "oracle_gain_pp": 100.0 * (oracle_hits - baseline_hits) / count,
        "added_hits": added_hits,
        "lost_hits": lost_hits,
        "oracle_new_hits": oracle_hits - baseline_hits,
        "candidate_rows_changed": changed_rows,
        "candidate_rows_changed_ratio": changed_rows / count,
    }


def summarize_records(
    states: Sequence[dict], inventory_records: Sequence[dict], args
) -> dict:
    inventory_by_sample = {
        int(record["sample_index"]): record for record in inventory_records
    }
    if not inventory_by_sample:
        raise ValueError("at least one inventory record is required")
    sources = list(inventory_records[0]["inventories"])
    subsets = _subset_rows(states)
    matrix = {}
    for source in sources:
        source_matrix = {}
        for pool_size in args.inventory_pool_sizes:
            for candidate_budget in args.candidate_budgets:
                for inventory_slots in args.inventory_slots:
                    if inventory_slots > candidate_budget:
                        continue
                    key = (
                        f"pool{pool_size}_budget{candidate_budget}_"
                        f"slots{inventory_slots}"
                    )
                    source_matrix[key] = {
                        subset_name: _coverage_summary(
                            subset_states,
                            inventory_by_sample,
                            source,
                            pool_size,
                            candidate_budget,
                            inventory_slots,
                        )
                        for subset_name, subset_states in subsets.items()
                    }
        matrix[source] = source_matrix

    primary_key = (
        f"pool{args.primary_inventory_pool_size}_"
        f"budget{args.primary_candidate_budget}_"
        f"slots{args.primary_inventory_slots}"
    )
    primary_observed = matrix[args.primary_source][primary_key][
        "high_visual_states"
    ]

    ordered_samples = sorted(inventory_by_sample)
    mismatch_remap = None
    mismatch = None
    if len(ordered_samples) >= 2:
        mismatch_remap = {
            sample_index: ordered_samples[(index + 1) % len(ordered_samples)]
            for index, sample_index in enumerate(ordered_samples)
        }
        mismatch = {
            subset_name: _coverage_summary(
                subset_states,
                inventory_by_sample,
                args.primary_source,
                args.primary_inventory_pool_size,
                args.primary_candidate_budget,
                args.primary_inventory_slots,
                sample_remap=mismatch_remap,
            )
            for subset_name, subset_states in subsets.items()
        }

    enough_states = primary_observed["num_states"] >= args.min_gate_states
    oracle_gain = primary_observed["oracle_gain_pp"]
    oracle_pass = bool(
        enough_states
        and oracle_gain is not None
        and oracle_gain >= 100.0 * args.min_oracle_gain
    )
    mismatch_high = mismatch["high_visual_states"] if mismatch else None
    specificity_delta_pp = (
        oracle_gain - mismatch_high["oracle_gain_pp"]
        if oracle_gain is not None
        and mismatch_high is not None
        and mismatch_high["oracle_gain_pp"] is not None
        else None
    )
    specificity_pass = bool(
        specificity_delta_pp is not None
        and specificity_delta_pp >= 100.0 * args.min_specificity_gain
    )
    if not enough_states or mismatch is None:
        decision = "inconclusive"
    elif oracle_pass and specificity_pass:
        decision = "go"
    else:
        decision = "no-go"

    return {
        "method": "Prompt-native Visual Lexical Inventory kill test",
        "diagnostic_only": True,
        "num_states": len(states),
        "num_samples": len(inventory_records),
        "source_order": sources,
        "configuration": {
            "inventory_regions": args.inventory_regions,
            "inventory_pool_sizes": args.inventory_pool_sizes,
            "candidate_budgets": args.candidate_budgets,
            "inventory_slots": args.inventory_slots,
            "primary_source": args.primary_source,
            "primary_inventory_pool_size": args.primary_inventory_pool_size,
            "primary_candidate_budget": args.primary_candidate_budget,
            "primary_inventory_slots": args.primary_inventory_slots,
            "visual_threshold": args.visual_threshold,
            "max_new_token": args.max_new_token,
            "attention_implementation": args.attn_implementation,
        },
        "inventory_construction_latency_ms": _distribution(
            record["inventory_construction_ms"] for record in inventory_records
        ),
        "candidate_coverage": matrix,
        "mismatched_image_control": {
            "available": mismatch is not None,
            "sample_remap": mismatch_remap,
            "primary_configuration": mismatch,
        },
        "gate": {
            "decision": decision,
            "pass": decision == "go",
            "evaluated_subset": "high_visual_states",
            "primary_configuration": (
                f"{args.primary_source}/{primary_key}"
            ),
            "enough_states": enough_states,
            "oracle_pass": oracle_pass,
            "specificity_pass": specificity_pass,
            "specificity_delta_pp": specificity_delta_pp,
            "thresholds": {
                "min_gate_states": args.min_gate_states,
                "min_oracle_gain": args.min_oracle_gain,
                "min_specificity_gain": args.min_specificity_gain,
            },
            "observed": primary_observed,
            "mismatched_image_observed": mismatch_high,
            "note": (
                "The gate tests candidate-source potential, not deployment readiness. "
                "A go result still requires a learned-free contextual ranker and a "
                "path-coherence test before tree integration."
            ),
        },
    }


@torch.inference_mode()
def evaluate(args):
    model_kwargs = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
    }
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path="",
        total_token=max(args.candidate_budgets),
        **model_kwargs,
    )
    model.eval()
    base_model = model.base_model
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
    inventory_path = output_root / "inventory_records.jsonl"
    states_path = output_root / "state_records.jsonl"
    inventory_records: List[dict] = []
    states: List[dict] = []
    max_pool_size = max(args.inventory_pool_sizes)
    max_candidate_budget = max(args.candidate_budgets)
    excluded_token_ids = set(int(token) for token in model.tokenizer.all_special_ids)
    for attribute in ("image_token_id", "video_token_id"):
        token_id = getattr(base_model.config, attribute, None)
        if token_id is not None:
            excluded_token_ids.add(int(token_id))

    print(f"Loaded {len(data)} samples")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"inventory_regions={args.inventory_regions}")
    print(f"inventory_pool_sizes={args.inventory_pool_sizes}")
    print(f"candidate_budgets={args.candidate_budgets}")
    print(f"inventory_slots={args.inventory_slots}")
    print(f"inventory_path={inventory_path}")
    print(f"states_path={states_path}")

    with inventory_path.open("w", encoding="utf-8") as inventory_file, states_path.open(
        "w", encoding="utf-8"
    ) as states_file:
        for sample_index, sample in enumerate(tqdm(data, desc="VLI kill test")):
            current_length_data.zero_()
            model_inputs = build_prompt(sample, args)
            input_ids = model_inputs["input_ids"]
            prompt_length = int(input_ids.shape[1])
            visual_mask = model._build_visual_token_mask(input_ids).detach()
            image_grid_thw = model_inputs.get("image_grid_thw")
            layout = build_visual_probe_layout(
                visual_mask,
                image_grid_thw,
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
            calibrator = VisualGroundingCalibrator.from_prompt(
                init_output.hidden_states[-1][0], visual_mask
            )
            question = sample.get("turns", [sample.get("prompt", "")])[0]
            question_token_ids = _question_lexical_token_ids(
                model.tokenizer, question
            )

            torch.cuda.synchronize()
            inventory_started = time.perf_counter()
            inventories = build_visual_lexical_inventories(
                init_output.logits[0],
                visual_mask,
                layout,
                question_token_ids,
                max_pool_size,
                excluded_token_ids=sorted(excluded_token_ids),
                valid_vocab_size=min(
                    len(model.tokenizer), int(init_output.logits.shape[-1])
                ),
            )
            torch.cuda.synchronize()
            inventory_construction_ms = 1000.0 * (
                time.perf_counter() - inventory_started
            )
            inventory_record = {
                "question_id": sample["id"],
                "topic": sample.get("topic", "unknown"),
                "category": sample.get("category", "unknown"),
                "sample_index": sample_index,
                "prompt_length": prompt_length,
                "num_visual_tokens": int(visual_mask.sum().item()),
                "used_grid_metadata": layout.used_grid_metadata,
                "inventory_regions": args.inventory_regions,
                "inventory_construction_ms": inventory_construction_ms,
                "inventories": inventories,
            }
            inventory_records.append(inventory_record)
            inventory_file.write(json.dumps(inventory_record) + "\n")
            inventory_file.flush()

            transition_rows = _seed_transition_rows(
                input_ids[0], init_output.logits[0], max_candidate_budget
            )
            root_token = int(init_output.logits[0, -1].argmax().item())
            generated_tokens = [root_token]
            if root_token in stop_token_ids:
                continue

            for step_index in range(max(args.max_new_token - 1, 0)):
                prefix_length = int(current_length_data[0].item())
                baseline_previous = transition_rows.get(root_token)
                reference_output = _run_reference_forward(
                    base_model,
                    root_token,
                    prefix_length,
                    past_key_values,
                    current_length_data,
                )
                reference_logits = reference_output.logits[0, 0]
                target_token = int(reference_logits.argmax().item())
                grounding_score = calibrator.score(
                    reference_output.hidden_states[-1][0, 0]
                )
                state = {
                    "question_id": sample["id"],
                    "topic": sample.get("topic", "unknown"),
                    "category": sample.get("category", "unknown"),
                    "sample_index": sample_index,
                    "step_index": step_index,
                    "generated_prefix_length": len(generated_tokens),
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
                }
                states.append(state)
                states_file.write(json.dumps(state) + "\n")
                states_file.flush()

                transition_rows[root_token] = [
                    int(token)
                    for token in reference_logits.topk(max_candidate_budget).indices.tolist()
                ]
                generated_tokens.append(target_token)
                root_token = target_token
                if target_token in stop_token_ids:
                    break

    summary = summarize_records(states, inventory_records, args)
    summary["inventory_records_path"] = str(inventory_path.resolve())
    summary["state_records_path"] = str(states_path.resolve())
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
                "inventory_construction_latency_ms": summary[
                    "inventory_construction_latency_ms"
                ],
                "gate": summary["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary_path={summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-native Visual Lexical Inventory kill test"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--samples-per-topic", type=int)
    parser.add_argument("--topic-offset", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=64)
    parser.add_argument("--inventory-regions", type=int, default=4)
    parser.add_argument(
        "--inventory-pool-sizes",
        type=_parse_positive_ints,
        default=_parse_positive_ints("16,32,64,128"),
    )
    parser.add_argument(
        "--candidate-budgets",
        type=_parse_positive_ints,
        default=_parse_positive_ints("4,6,8"),
    )
    parser.add_argument(
        "--inventory-slots",
        type=_parse_positive_ints,
        default=_parse_positive_ints("1,2"),
    )
    parser.add_argument("--primary-source", default="visual_max")
    parser.add_argument("--primary-inventory-pool-size", type=int, default=64)
    parser.add_argument("--primary-candidate-budget", type=int, default=4)
    parser.add_argument("--primary-inventory-slots", type=int, default=2)
    parser.add_argument("--visual-threshold", type=float, default=0.55)
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "eager"], default="sdpa"
    )
    parser.add_argument("--min-gate-states", type=int, default=20)
    parser.add_argument("--min-oracle-gain", type=float, default=0.08)
    parser.add_argument("--min-specificity-gain", type=float, default=0.02)
    args = parser.parse_args()
    if args.inventory_regions <= 0:
        parser.error("--inventory-regions must be positive")
    if args.primary_inventory_pool_size not in args.inventory_pool_sizes:
        parser.error(
            "--primary-inventory-pool-size must be in --inventory-pool-sizes"
        )
    if args.primary_candidate_budget not in args.candidate_budgets:
        parser.error("--primary-candidate-budget must be in --candidate-budgets")
    if args.primary_inventory_slots not in args.inventory_slots:
        parser.error("--primary-inventory-slots must be in --inventory-slots")
    if args.primary_inventory_slots > args.primary_candidate_budget:
        parser.error("primary inventory slots cannot exceed the candidate budget")
    if not 0.0 <= args.visual_threshold <= 1.0:
        parser.error("--visual-threshold must lie in [0, 1]")
    if args.min_oracle_gain < 0 or args.min_specificity_gain < 0:
        parser.error("gate gains must be non-negative")
    args.model = args.base_model_path
    evaluate(args)


if __name__ == "__main__":
    main()
