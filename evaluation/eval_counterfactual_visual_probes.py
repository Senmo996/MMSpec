"""Falsification-first diagnostic for counterfactual visual evidence recycling.

This script does not claim an end-to-end speculative decoder.  It tests the
mechanism that such a decoder would rely on:

1. Repeated same-position queries are evaluated in one target-model forward.
2. Each counterfactual query drops a different spatial visual region.
3. A q_len=1 reference pass supplies the exact greedy trajectory and commits
   the only KV state used by later tokens.
4. On repeated token states, candidates recycled from an earlier full view are
   compared with an equal-size full+counterfactual candidate row.

The second reference pass is diagnostic scaffolding only.  If full-view probe
fidelity is high and coverage improves, the probes can later be placed inside
the ordinary verifier pass and the reference pass removed.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional

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
    VisualProbeLayout,
    build_counterfactual_attention_mask,
    build_visual_probe_layout,
    compose_candidate_view_logits,
    cover_candidates,
    full_view_candidates,
    make_repeated_position_ids,
    multiview_jsd,
    pearson_correlation,
    topk_union_size,
)
from method.sam_grounded.spec_model import SpecModel
from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids


def _parse_positive_ints(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        parsed = int(item.strip())
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _percentile(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
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


def _coverage_summary(rows: List[dict], budget: int) -> dict:
    if not rows:
        return {
            "candidate_budget": budget,
            "num_states": 0,
            "baseline_coverage": None,
            "cover_coverage": None,
            "coverage_gain": None,
            "coverage_gain_pp": None,
            "added_hits": 0,
            "lost_hits": 0,
        }
    baseline_hits = sum(int(row["baseline_hit"]) for row in rows)
    cover_hits = sum(int(row["cover_hit"]) for row in rows)
    count = len(rows)
    gain = (cover_hits - baseline_hits) / count
    return {
        "candidate_budget": budget,
        "num_states": count,
        "baseline_hits": baseline_hits,
        "cover_hits": cover_hits,
        "baseline_coverage": baseline_hits / count,
        "cover_coverage": cover_hits / count,
        "coverage_gain": gain,
        "coverage_gain_pp": 100.0 * gain,
        "added_hits": sum(
            int(not row["baseline_hit"] and row["cover_hit"]) for row in rows
        ),
        "lost_hits": sum(
            int(row["baseline_hit"] and not row["cover_hit"]) for row in rows
        ),
        "candidate_rows_changed": sum(int(row["candidate_row_changed"]) for row in rows),
        "candidate_rows_changed_ratio": sum(
            int(row["candidate_row_changed"]) for row in rows
        )
        / count,
    }


def summarize_records(records: List[dict], args) -> dict:
    budgets = {}
    for budget in args.candidate_budgets:
        keyed = str(budget)
        eligible = [
            row["coverage"][keyed]
            for row in records
            if row["coverage"].get(keyed) is not None
        ]
        counterfactual = [
            row["coverage"][keyed]
            for row in records
            if row["has_counterfactual_history"]
            and row["coverage"].get(keyed) is not None
        ]
        high_visual = [
            row["coverage"][keyed]
            for row in records
            if row["has_counterfactual_history"]
            and row["grounding_score"] >= args.visual_threshold
            and row["coverage"].get(keyed) is not None
        ]
        low_visual = [
            row["coverage"][keyed]
            for row in records
            if row["has_counterfactual_history"]
            and row["grounding_score"] < args.visual_threshold
            and row["coverage"].get(keyed) is not None
        ]
        budgets[keyed] = {
            "all_recycled_states": _coverage_summary(eligible, budget),
            "counterfactual_history_states": _coverage_summary(counterfactual, budget),
            "high_visual_counterfactual_states": _coverage_summary(high_visual, budget),
            "low_visual_counterfactual_states": _coverage_summary(low_visual, budget),
        }

    primary_key = str(args.primary_candidate_budget)
    jsd_rows = [
        row
        for row in records
        if row["has_counterfactual_history"]
        and row["coverage"].get(primary_key) is not None
    ]
    jsd_values = [row["view_jsd"] for row in jsd_rows]
    marginal_benefits = [
        int(row["coverage"][primary_key]["cover_hit"])
        - int(row["coverage"][primary_key]["baseline_hit"])
        for row in jsd_rows
    ]
    jsd_benefit_correlation = pearson_correlation(jsd_values, marginal_benefits)
    ordered_pairs = sorted(zip(jsd_values, marginal_benefits))
    quartile_size = max(len(ordered_pairs) // 4, 1) if ordered_pairs else 0
    bottom_benefit = (
        statistics.fmean(value for _, value in ordered_pairs[:quartile_size])
        if quartile_size
        else None
    )
    top_benefit = (
        statistics.fmean(value for _, value in ordered_pairs[-quartile_size:])
        if quartile_size
        else None
    )

    latency = {}
    reference_latencies = [
        row["reference_latency_ms"]
        for row in records
        if row["include_in_latency_summary"]
    ]
    reference_mean = (
        statistics.fmean(reference_latencies) if reference_latencies else None
    )
    for probe_count in args.latency_probe_counts:
        keyed = str(probe_count)
        paired = [
            row
            for row in records
            if row["include_in_latency_summary"]
            and keyed in row["probe_latency_ms"]
        ]
        probe_times = [row["probe_latency_ms"][keyed] for row in paired]
        ratios = [
            row["probe_latency_ms"][keyed] / row["reference_latency_ms"]
            for row in paired
            if row["reference_latency_ms"] > 0
        ]
        latency[keyed] = {
            "num_counterfactual_probes": probe_count,
            "total_same_position_queries": probe_count + 1,
            "probe_forward_ms": _distribution(probe_times),
            "paired_ratio_to_q_len_1": _distribution(ratios),
        }

    top1_match_ratio = (
        sum(int(row["probe_reference_top1_match"]) for row in records) / len(records)
        if records
        else None
    )
    primary_high = budgets[primary_key]["high_visual_counterfactual_states"]
    primary_all = budgets[primary_key]["counterfactual_history_states"]
    primary_latency = latency[str(args.counterfactual_probes)][
        "paired_ratio_to_q_len_1"
    ]["mean"]
    enough_coverage_states = primary_high["num_states"] >= args.min_gate_states
    evaluated_gain = (
        primary_high["coverage_gain"]
        if enough_coverage_states
        else primary_all["coverage_gain"]
    )
    coverage_pass = (
        enough_coverage_states
        and evaluated_gain is not None
        and evaluated_gain >= args.min_coverage_gain
    )
    fidelity_pass = (
        top1_match_ratio is not None
        and top1_match_ratio >= args.min_full_view_match
    )
    latency_pass = (
        primary_latency is not None
        and primary_latency <= args.max_probe_latency_ratio
    )
    jsd_signal_pass = (
        top_benefit is not None
        and bottom_benefit is not None
        and top_benefit > bottom_benefit
    )
    if coverage_pass and fidelity_pass and latency_pass and jsd_signal_pass:
        gate_decision = "go"
    elif enough_coverage_states and evaluated_gain is not None and evaluated_gain <= 0:
        gate_decision = "no-go"
    else:
        gate_decision = "inconclusive"

    return {
        "method": "COVER counterfactual visual evidence recycling diagnostic",
        "diagnostic_only": True,
        "num_records": len(records),
        "num_samples": len({row["question_id"] for row in records}),
        "configuration": {
            "counterfactual_probes": args.counterfactual_probes,
            "total_same_position_queries": args.counterfactual_probes + 1,
            "candidate_budgets": args.candidate_budgets,
            "primary_candidate_budget": args.primary_candidate_budget,
            "anchor_fraction": args.anchor_fraction,
            "candidate_view_mode": args.candidate_view_mode,
            "evidence_scale": args.evidence_scale,
            "visual_threshold": args.visual_threshold,
            "max_new_token": args.max_new_token,
            "attention_implementation": args.attn_implementation,
        },
        "candidate_coverage": budgets,
        "probe_fidelity": {
            "full_view_reference_top1_match_ratio": top1_match_ratio,
            "max_abs_logit_difference": _distribution(
                row["probe_reference_max_abs_logit_diff"] for row in records
            ),
        },
        "view_diversity": {
            "jsd": _distribution(row["view_jsd"] for row in records),
            "topk_union_size": _distribution(row["topk_union_size"] for row in records),
            "counterfactual_top1_disagreement_ratio": (
                statistics.fmean(
                    row["counterfactual_top1_disagreement_ratio"] for row in records
                )
                if records
                else None
            ),
        },
        "jsd_predictiveness": {
            "num_counterfactual_history_states": len(jsd_rows),
            "pearson_jsd_vs_marginal_coverage_benefit": jsd_benefit_correlation,
            "bottom_jsd_quartile_mean_benefit": bottom_benefit,
            "top_jsd_quartile_mean_benefit": top_benefit,
        },
        "latency": {
            "reference_q_len_1_ms": _distribution(reference_latencies),
            "by_counterfactual_probe_count": latency,
            "unpaired_reference_mean_ms": reference_mean,
        },
        "gate": {
            "decision": gate_decision,
            "coverage_pass": coverage_pass,
            "fidelity_pass": fidelity_pass,
            "latency_pass": latency_pass,
            "jsd_signal_pass": jsd_signal_pass,
            "high_visual_states_sufficient": enough_coverage_states,
            "thresholds": {
                "min_gate_states": args.min_gate_states,
                "min_coverage_gain": args.min_coverage_gain,
                "min_full_view_match": args.min_full_view_match,
                "max_probe_latency_ratio": args.max_probe_latency_ratio,
            },
            "note": (
                "A no-go requires enough eligible high-visual states and non-positive "
                "coverage gain; small smoke runs should normally remain inconclusive."
            ),
        },
    }


def _seed_transition_rows(
    prompt_ids: torch.Tensor,
    prompt_logits: torch.Tensor,
    candidate_budgets: List[int],
):
    max_budget = max(candidate_budgets)
    top_tokens = prompt_logits.topk(max_budget, dim=-1).indices.detach().to("cpu")
    baseline = {budget: {} for budget in candidate_budgets}
    cover = {budget: {} for budget in candidate_budgets}
    for position, token in enumerate(prompt_ids.detach().to("cpu").tolist()):
        for budget in candidate_budgets:
            row = [int(value) for value in top_tokens[position, :budget].tolist()]
            baseline[budget][int(token)] = row
            cover[budget][int(token)] = list(row)
    return baseline, cover


def _run_probe_forward(
    base_model,
    root_token: int,
    prefix_length: int,
    layout: VisualProbeLayout,
    past_key_values,
    current_length_data: torch.Tensor,
):
    device = base_model.device
    query_length = layout.num_regions + 1
    query = torch.full(
        (1, query_length), root_token, dtype=torch.long, device=device
    )
    position_ids = make_repeated_position_ids(
        prefix_length,
        query_length,
        base_model.rope_deltas,
        device,
    )
    cache_position = torch.arange(
        prefix_length, prefix_length + query_length, device=device
    )
    attention_mask = build_counterfactual_attention_mask(
        prefix_length,
        layout,
        base_model.dtype,
        device,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = base_model(
        input_ids=query,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=True,
        use_cache=True,
        output_hidden_states=False,
    )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    current_length_data.fill_(prefix_length)
    return output.logits, elapsed_ms


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
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = base_model(
        input_ids=query,
        position_ids=position_ids,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=True,
        use_cache=True,
        output_hidden_states=True,
    )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    current_length_data.fill_(prefix_length + 1)
    return output, elapsed_ms


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
    records_path = output_root / "probe_records.jsonl"
    records: List[dict] = []
    print(f"Loaded {len(data)} samples")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"counterfactual_probes={args.counterfactual_probes}")
    print(f"latency_probe_counts={args.latency_probe_counts}")
    print(f"candidate_budgets={args.candidate_budgets}")
    print(f"records_path={records_path}")

    with records_path.open("w", encoding="utf-8") as record_file:
        for sample_index, sample in enumerate(tqdm(data, desc="COVER probe diagnostic")):
            current_length_data.zero_()
            model_inputs = build_prompt(sample, args)
            input_ids = model_inputs["input_ids"]
            prompt_length = int(input_ids.shape[1])
            visual_mask = model._build_visual_token_mask(input_ids).detach()
            image_grid_thw = model_inputs.get("image_grid_thw")
            spatial_merge_size = int(base_model.config.vision_config.spatial_merge_size)
            layouts = {
                count: build_visual_probe_layout(
                    visual_mask,
                    image_grid_thw,
                    spatial_merge_size,
                    count,
                )
                for count in set(args.latency_probe_counts + [args.counterfactual_probes])
            }

            init_kwargs = dict(model_inputs)
            init_output = base_model(
                **init_kwargs,
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
            baseline_rows, cover_rows = _seed_transition_rows(
                input_ids[0], init_output.logits[0], args.candidate_budgets
            )
            counterfactually_observed_tokens = set()
            root_token = int(init_output.logits[0, -1].argmax().item())
            generated_tokens = [root_token]

            if root_token in stop_token_ids:
                continue
            for step_index in range(max(args.max_new_token - 1, 0)):
                prefix_length = int(current_length_data[0].item())
                probe_latency_ms = {}
                primary_logits = None
                run_counts = [args.counterfactual_probes]
                if step_index < args.latency_steps:
                    run_counts.extend(args.latency_probe_counts)
                run_counts = list(dict.fromkeys(run_counts))
                for count in run_counts:
                    logits, elapsed_ms = _run_probe_forward(
                        base_model,
                        root_token,
                        prefix_length,
                        layouts[count],
                        past_key_values,
                        current_length_data,
                    )
                    probe_latency_ms[str(count)] = elapsed_ms
                    if count == args.counterfactual_probes:
                        primary_logits = logits
                if primary_logits is None:
                    raise RuntimeError("primary probe forward was not executed")

                reference_output, reference_latency_ms = _run_reference_forward(
                    base_model,
                    root_token,
                    prefix_length,
                    past_key_values,
                    current_length_data,
                )
                reference_logits = reference_output.logits[0, 0]
                target_token = int(reference_logits.argmax().item())
                primary_logits = primary_logits[0]
                # The q_len=1 full view isolates candidate-content effects from
                # any BF16 kernel drift caused solely by changing q_len.
                raw_view_logits = torch.cat(
                    [reference_logits.unsqueeze(0), primary_logits[1:]], dim=0
                )
                fusion_logits = compose_candidate_view_logits(
                    reference_logits,
                    primary_logits[1:],
                    mode=args.candidate_view_mode,
                    evidence_scale=args.evidence_scale,
                )
                full_probe_logits = primary_logits[0]
                grounding_score = (
                    calibrator.score(reference_output.hidden_states[-1][0, 0])
                    if calibrator is not None
                    else 0.0
                )
                has_counterfactual_history = root_token in counterfactually_observed_tokens
                coverage = {}
                for budget in args.candidate_budgets:
                    baseline_previous = baseline_rows[budget].get(root_token)
                    cover_previous = cover_rows[budget].get(root_token)
                    if baseline_previous is None or cover_previous is None:
                        coverage[str(budget)] = None
                    else:
                        coverage[str(budget)] = {
                            "baseline_hit": target_token in baseline_previous,
                            "cover_hit": target_token in cover_previous,
                            "candidate_row_changed": baseline_previous != cover_previous,
                            "baseline_candidates": baseline_previous,
                            "cover_candidates": cover_previous,
                        }

                max_budget = max(args.candidate_budgets)
                baseline_new_max = full_view_candidates(reference_logits, max_budget)
                for budget in args.candidate_budgets:
                    baseline_rows[budget][root_token] = baseline_new_max[:budget]
                    cover_rows[budget][root_token] = cover_candidates(
                        fusion_logits,
                        budget,
                        anchor_fraction=args.anchor_fraction,
                    )
                counterfactually_observed_tokens.add(root_token)

                cf_top1 = primary_logits[1:].argmax(dim=-1)
                disagreement = (
                    cf_top1.ne(reference_logits.argmax()).float().mean().item()
                    if cf_top1.numel()
                    else 0.0
                )
                record = {
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
                    "num_visual_tokens": layouts[args.counterfactual_probes].num_visual_tokens,
                    "used_grid_metadata": layouts[
                        args.counterfactual_probes
                    ].used_grid_metadata,
                    "counterfactual_probes": args.counterfactual_probes,
                    "grounding_score": float(grounding_score),
                    "high_visual_state": grounding_score >= args.visual_threshold,
                    "has_counterfactual_history": has_counterfactual_history,
                    "coverage": coverage,
                    "view_jsd": multiview_jsd(raw_view_logits),
                    "topk_union_size": topk_union_size(
                        fusion_logits, args.primary_candidate_budget
                    ),
                    "counterfactual_top1_disagreement_ratio": float(disagreement),
                    "probe_reference_top1_match": int(full_probe_logits.argmax().item())
                    == target_token,
                    "probe_reference_max_abs_logit_diff": float(
                        (full_probe_logits.float() - reference_logits.float())
                        .abs()
                        .max()
                        .item()
                    ),
                    "probe_latency_ms": probe_latency_ms,
                    "reference_latency_ms": reference_latency_ms,
                    "include_in_latency_summary": step_index
                    >= args.latency_warmup_steps,
                }
                records.append(record)
                record_file.write(json.dumps(record) + "\n")
                record_file.flush()

                generated_tokens.append(target_token)
                root_token = target_token
                if target_token in stop_token_ids:
                    break

    summary = summarize_records(records, args)
    summary["records_path"] = str(records_path.resolve())
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary_path={summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual visual evidence recycling diagnostic"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--samples-per-topic", type=int)
    parser.add_argument("--topic-offset", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=32)
    parser.add_argument("--counterfactual-probes", type=int, default=4)
    parser.add_argument(
        "--latency-probe-counts", type=_parse_positive_ints, default=_parse_positive_ints("2,4,8")
    )
    parser.add_argument("--latency-steps", type=int, default=4)
    parser.add_argument("--latency-warmup-steps", type=int, default=2)
    parser.add_argument(
        "--candidate-budgets", type=_parse_positive_ints, default=_parse_positive_ints("4,6,8")
    )
    parser.add_argument("--primary-candidate-budget", type=int, default=6)
    parser.add_argument("--anchor-fraction", type=float, default=0.5)
    parser.add_argument(
        "--candidate-view-mode",
        choices=["masked", "evidence", "both"],
        default="masked",
    )
    parser.add_argument("--evidence-scale", type=float, default=1.0)
    parser.add_argument("--visual-threshold", type=float, default=0.55)
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "eager"], default="sdpa"
    )
    parser.add_argument("--min-gate-states", type=int, default=20)
    parser.add_argument("--min-coverage-gain", type=float, default=0.10)
    parser.add_argument("--min-full-view-match", type=float, default=0.99)
    parser.add_argument("--max-probe-latency-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if args.counterfactual_probes <= 0:
        parser.error("--counterfactual-probes must be positive")
    if args.primary_candidate_budget not in args.candidate_budgets:
        parser.error("--primary-candidate-budget must be in --candidate-budgets")
    if args.counterfactual_probes not in args.latency_probe_counts:
        args.latency_probe_counts.append(args.counterfactual_probes)
    if not 0.0 <= args.anchor_fraction <= 1.0:
        parser.error("--anchor-fraction must lie in [0, 1]")
    if args.evidence_scale < 0:
        parser.error("--evidence-scale must be non-negative")
    args.model = args.base_model_path
    evaluate(args)


if __name__ == "__main__":
    main()
