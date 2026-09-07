"""Auditable discovery/held-out analysis of multimodal selective reuse.

Coverage is reported on all decoder states. Source utility is conditional on
both U and G/C existing and compares shadow trees truncated to an equal node
budget. A rule may be selected only on discovery; held-out is confirmatory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = (
    "u_coverage",
    "gc_coverage",
    "u_conditional_top8_recall",
    "gc_conditional_top8_recall",
    "matched_gc_minus_u_accept",
)
STAT_NAMES = (
    "state_count",
    "u_available",
    "gc_available",
    "u_hit",
    "gc_hit",
    "eligible_count",
    "matched_delta_sum",
    "u_matched_accept_sum",
    "gc_matched_accept_sum",
)
PREDECLARED_MATCHED_BUDGETS = (0, 8, 16, 31, 47, 63)
TRUE_PROBE_PROTOCOL = "teacher_forced_mean_patch_occlusion_v1"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def _benchmark_name(result: dict, path: Path) -> str:
    question_id = str(result.get("question_id", ""))
    if question_id.lower().startswith("mmspec"):
        return "MMSpec"
    topic = str(result.get("topic") or path.parent.parent.name)
    aliases = {
        "MME_Benchmark": "MME",
        "mme_benchmark": "MME",
        "mmt_bench": "MMT-Bench",
        "seedbench": "SEEDBench",
        "scienceqa": "ScienceQA",
        "ocrbench": "OCRBench",
        "chartqa": "ChartQA",
        "mathvista": "MathVista",
        "textvqa": "TextVQA",
    }
    return aliases.get(topic, topic)


def discover_result_paths(results_roots: Sequence[Path], policy: str) -> List[Path]:
    paths = set()
    for root in results_roots:
        candidates = [root] if root.is_file() else root.rglob("results.jsonl")
        for path in candidates:
            if path.name == "results.jsonl" and path.parent.name == policy:
                paths.add(path.resolve())
    return sorted(paths)


def load_selective_records(paths: Sequence[Path]) -> List[dict]:
    records = []
    seen = set()
    for path in paths:
        for result in _read_jsonl(path):
            benchmark = _benchmark_name(result, path)
            question_id = str(result.get("question_id"))
            image_cluster = str(result.get("image_cluster_id") or question_id)
            analysis_split = str(result.get("analysis_split", "unsplit"))
            for choice in result.get("choices", []):
                choice_index = int(choice.get("index", 0))
                for turn_index, trace in enumerate(choice.get("policy_trace", [])):
                    for trace_record in trace:
                        if not trace_record.get("selective_reuse_diagnostics"):
                            continue
                        if trace_record.get("visual_probe_jsd") is None:
                            continue
                        iteration = int(trace_record.get("iteration", 0))
                        key = (benchmark, question_id, choice_index, turn_index, iteration)
                        if key in seen:
                            raise ValueError(
                                "duplicate selective state encountered: " + repr(key)
                            )
                        seen.add(key)
                        u_accept = int(
                            trace_record.get("selective_u_shadow_accept_len", 0)
                        )
                        gc_accept = int(
                            trace_record.get("selective_gc_shadow_accept_len", 0)
                        )
                        u_matched = int(
                            trace_record.get("selective_u_matched_accept_len", u_accept)
                        )
                        gc_matched = int(
                            trace_record.get("selective_gc_matched_accept_len", gc_accept)
                        )
                        u_nodes = int(trace_record.get("selective_u_tree_nodes", 0))
                        gc_nodes = int(trace_record.get("selective_gc_tree_nodes", 0))
                        records.append(
                            {
                                "benchmark": benchmark,
                                "cluster_id": f"{benchmark}:{image_cluster}",
                                "question_id": question_id,
                                "category": result.get("category", "default"),
                                "analysis_split": analysis_split,
                                "analysis_split_seed": result.get("analysis_split_seed"),
                                "choice_index": choice_index,
                                "turn_index": turn_index,
                                "iteration": iteration,
                                "output_position": int(
                                    trace_record.get("selective_output_position", -1)
                                ),
                                # JSD is theoretically nonnegative. Clamp only
                                # floating-point residue from log-softmax math.
                                "visual_probe_jsd": max(
                                    0.0, float(trace_record["visual_probe_jsd"])
                                ),
                                "visual_probe_top1_disagreement_rate": float(
                                    trace_record.get(
                                        "visual_probe_top1_disagreement_rate", 0.0
                                    )
                                ),
                                "visual_probe_topk_union_size": int(
                                    trace_record.get("visual_probe_topk_union_size", 0)
                                ),
                                "visual_probe_max_target_logprob_drop": float(
                                    trace_record.get(
                                        "visual_probe_max_target_logprob_drop", 0.0
                                    )
                                ),
                                "visual_probe_full_target_logprob": float(
                                    trace_record.get(
                                        "visual_probe_full_target_logprob", 0.0
                                    )
                                ),
                                "visual_probe_span2_num_tokens": int(
                                    trace_record.get(
                                        "visual_probe_span2_num_tokens", 0
                                    )
                                ),
                                "visual_probe_span2_mean_jsd": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_mean_jsd"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_mean_jsd"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_mean_mean_jsd": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_mean_mean_jsd"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_mean_mean_jsd"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_wrong_mean_jsd": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_wrong_mean_jsd"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_wrong_mean_jsd"
                                        ]
                                    )
                                ),
                                "visual_probe_mean_jsd": (
                                    None
                                    if trace_record.get("visual_probe_mean_jsd")
                                    is None
                                    else float(trace_record["visual_probe_mean_jsd"])
                                ),
                                "visual_probe_wrong_jsd": (
                                    None
                                    if trace_record.get("visual_probe_wrong_jsd")
                                    is None
                                    else float(trace_record["visual_probe_wrong_jsd"])
                                ),
                                "visual_probe_mean_target_logprob_drop": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_mean_target_logprob_drop"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_mean_target_logprob_drop"
                                        ]
                                    )
                                ),
                                "visual_probe_wrong_target_logprob_drop": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_wrong_target_logprob_drop"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_wrong_target_logprob_drop"
                                        ]
                                    )
                                ),
                                "visual_probe_candidate_alignment_available": bool(
                                    trace_record.get(
                                        "visual_probe_candidate_alignment_available",
                                        False,
                                    )
                                ),
                                "visual_probe_candidate_alignment_budget": int(
                                    trace_record.get(
                                        "visual_probe_candidate_alignment_budget", 0
                                    )
                                ),
                                "visual_probe_candidate_alignment_invalid_reason": (
                                    trace_record.get(
                                        "visual_probe_candidate_alignment_invalid_reason"
                                    )
                                ),
                                "visual_probe_candidate_alignment_uses_target_outcome": bool(
                                    trace_record.get(
                                        "visual_probe_candidate_alignment_uses_target_outcome",
                                        False,
                                    )
                                ),
                                "visual_probe_u_candidate_consensus_support": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_u_candidate_consensus_support"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_u_candidate_consensus_support"
                                        ]
                                    )
                                ),
                                "visual_probe_gc_candidate_consensus_support": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_gc_candidate_consensus_support"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_gc_candidate_consensus_support"
                                        ]
                                    )
                                ),
                                "visual_probe_gc_minus_u_candidate_visual_support": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_gc_minus_u_candidate_visual_support"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_gc_minus_u_candidate_visual_support"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_mean_target_logprob_drop": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_mean_target_logprob_drop"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_mean_target_logprob_drop"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_mean_target_drop_fraction": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_mean_target_drop_fraction"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_mean_target_drop_fraction"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_mean_mean_target_margin_drop": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_mean_mean_target_margin_drop"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_mean_mean_target_margin_drop"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_wrong_mean_target_margin_drop": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_wrong_mean_target_margin_drop"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_wrong_mean_target_margin_drop"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_mean_top1_change_rate": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_mean_top1_change_rate"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_mean_top1_change_rate"
                                        ]
                                    )
                                ),
                                "visual_probe_span2_wrong_top1_change_rate": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_span2_wrong_top1_change_rate"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_span2_wrong_top1_change_rate"
                                        ]
                                    )
                                ),
                                "visual_probe_context3_num_tokens": int(
                                    trace_record.get(
                                        "visual_probe_context3_num_tokens", 0
                                    )
                                ),
                                "visual_probe_context3_target_token_ids": list(
                                    trace_record.get(
                                        "visual_probe_context3_target_token_ids", []
                                    )
                                ),
                                "visual_probe_context3_full_top1_token_ids": list(
                                    trace_record.get(
                                        "visual_probe_context3_full_top1_token_ids",
                                        [],
                                    )
                                ),
                                "visual_probe_context3_full_top1_match_rate": (
                                    None
                                    if trace_record.get(
                                        "visual_probe_context3_full_top1_match_rate"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "visual_probe_context3_full_top1_match_rate"
                                        ]
                                    )
                                ),
                                "visual_probe_context3_mean_target_margin_drops": [
                                    float(value)
                                    for value in trace_record.get(
                                        "visual_probe_context3_mean_target_margin_drops",
                                        [],
                                    )
                                ],
                                "visual_probe_context3_wrong_target_margin_drops": [
                                    float(value)
                                    for value in trace_record.get(
                                        "visual_probe_context3_wrong_target_margin_drops",
                                        [],
                                    )
                                ],
                                "visual_probe_context3_mean_top1_changed": [
                                    bool(value)
                                    for value in trace_record.get(
                                        "visual_probe_context3_mean_top1_changed", []
                                    )
                                ],
                                "visual_probe_context3_wrong_top1_changed": [
                                    bool(value)
                                    for value in trace_record.get(
                                        "visual_probe_context3_wrong_top1_changed", []
                                    )
                                ],
                                "visual_probe_context3_same_text_trajectory": bool(
                                    trace_record.get(
                                        "visual_probe_context3_same_text_trajectory",
                                        False,
                                    )
                                ),
                                "visual_probe_wrong_image_used_category_fallback": bool(
                                    trace_record.get(
                                        "visual_probe_wrong_image_used_category_fallback",
                                        False,
                                    )
                                ),
                                "visual_probe_span2_same_text_trajectory": bool(
                                    trace_record.get(
                                        "visual_probe_span2_same_text_trajectory",
                                        False,
                                    )
                                ),
                                "visual_probe_span2_uses_future_tokens": bool(
                                    trace_record.get(
                                        "visual_probe_span2_uses_future_tokens",
                                        False,
                                    )
                                ),
                                "visual_probe_full_top1_matches_target": bool(
                                    trace_record.get(
                                        "visual_probe_full_top1_matches_target",
                                        False,
                                    )
                                ),
                                "visual_probe_full_top1_token_id": trace_record.get(
                                    "visual_probe_full_top1_token_id"
                                ),
                                "visual_probe_protocol": trace_record.get(
                                    "visual_probe_protocol", "legacy"
                                ),
                                "visual_probe_used_grid_metadata": bool(
                                    trace_record.get(
                                        "visual_probe_used_grid_metadata", False
                                    )
                                ),
                                "visual_probe_same_text_trajectory": bool(
                                    trace_record.get(
                                        "visual_probe_same_text_trajectory", False
                                    )
                                ),
                                "visual_probe_recomputed_vision_encoder": bool(
                                    trace_record.get(
                                        "visual_probe_recomputed_vision_encoder", False
                                    )
                                ),
                                "u_available": bool(
                                    trace_record.get(
                                        "selective_u_source_available", False
                                    )
                                ),
                                "u_available_before_request": bool(
                                    trace_record.get(
                                        "selective_u_available_before_request",
                                        False,
                                    )
                                ),
                                "u_row_top_probability_before_request": (
                                    None
                                    if trace_record.get(
                                        "selective_u_row_top_probability_before_request"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "selective_u_row_top_probability_before_request"
                                        ]
                                    )
                                ),
                                "u_row_top_probability": (
                                    None
                                    if trace_record.get(
                                        "selective_u_row_top_probability"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "selective_u_row_top_probability"
                                        ]
                                    )
                                ),
                                "u_persistent_row_top_probability": (
                                    None
                                    if trace_record.get(
                                        "selective_u_persistent_row_top_probability"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "selective_u_persistent_row_top_probability"
                                        ]
                                    )
                                ),
                                "gc_available": bool(
                                    trace_record.get(
                                        "selective_gc_source_available", False
                                    )
                                ),
                                "g_available": bool(
                                    trace_record.get(
                                        "selective_g_source_available", False
                                    )
                                ),
                                "c_available": bool(
                                    trace_record.get(
                                        "selective_c_source_available", False
                                    )
                                ),
                                "u_top8_hit": bool(
                                    trace_record.get("selective_u_top8_hit", False)
                                ),
                                "gc_top8_hit": bool(
                                    trace_record.get("selective_gc_top8_hit", False)
                                ),
                                "v_source": trace_record.get(
                                    "selective_v_source"
                                ),
                                "v_available": bool(
                                    trace_record.get(
                                        "selective_v_source_available", False
                                    )
                                ),
                                "v_top8_hit": bool(
                                    trace_record.get("selective_v_top8_hit", False)
                                ),
                                "uv_top8_hit": bool(
                                    trace_record.get("selective_uv_top8_hit", False)
                                ),
                                "uv_minus_u_top8_hit": int(
                                    trace_record.get(
                                        "selective_uv_minus_u_top8_hit", 0
                                    )
                                ),
                                "uv_candidate_budget": int(
                                    trace_record.get(
                                        "selective_uv_candidate_budget", 0
                                    )
                                ),
                                "uv_visual_slots": int(
                                    trace_record.get(
                                        "selective_uv_visual_slots", 0
                                    )
                                ),
                                "uv_added_candidate_token_id": (
                                    trace_record.get(
                                        "selective_uv_added_candidate_token_id"
                                    )
                                ),
                                "uv_displaced_candidate_token_id": (
                                    trace_record.get(
                                        "selective_uv_displaced_candidate_token_id"
                                    )
                                ),
                                "u_root_candidate_token_ids": list(
                                    trace_record.get(
                                        "selective_u_root_candidate_token_ids", []
                                    )
                                ),
                                "gc_root_candidate_token_ids": list(
                                    trace_record.get(
                                        "selective_gc_root_candidate_token_ids", []
                                    )
                                ),
                                "v_root_candidate_token_ids": list(
                                    trace_record.get(
                                        "selective_v_root_candidate_token_ids", []
                                    )
                                ),
                                "uv_root_candidate_token_ids": list(
                                    trace_record.get(
                                        "selective_uv_root_candidate_token_ids", []
                                    )
                                ),
                                "u_tree_nodes": u_nodes,
                                "gc_tree_nodes": gc_nodes,
                                "matched_tree_node_budget": int(
                                    trace_record.get(
                                        "selective_matched_tree_node_budget",
                                        min(u_nodes, gc_nodes),
                                    )
                                ),
                                "u_shadow_accept": u_accept,
                                "gc_shadow_accept": gc_accept,
                                "u_matched_accept": u_matched,
                                "gc_matched_accept": gc_matched,
                                "matched_gc_minus_u_accept": gc_matched - u_matched,
                                # Auxiliary compatibility alias. It is never
                                # the confirmatory utility estimand.
                                "gc_minus_u_shadow_accept": gc_accept - u_accept,
                                "target_token_id": trace_record.get(
                                    "selective_target_token_id"
                                ),
                                "root_transition_context_order": int(
                                    trace_record.get(
                                        "root_transition_context_order", 0
                                    )
                                ),
                                "root_transition_persistent": bool(
                                    trace_record.get(
                                        "root_transition_persistent", False
                                    )
                                ),
                                "root_transition_top_probability": (
                                    None
                                    if trace_record.get(
                                        "root_transition_top_probability"
                                    )
                                    is None
                                    else float(
                                        trace_record[
                                            "root_transition_top_probability"
                                        ]
                                    )
                                ),
                                "conditional_transition_count": int(
                                    trace_record.get(
                                        "conditional_transition_count", 0
                                    )
                                ),
                                "confidence": float(
                                    trace_record.get("confidence", 0.0)
                                ),
                                "next_confidence": float(
                                    trace_record.get("next_confidence", 0.0)
                                ),
                                "context_candidate_mode": str(
                                    trace_record.get(
                                        "context_candidate_mode", "unknown"
                                    )
                                ),
                                "transition_bin": int(
                                    trace_record.get("transition_bin", 0)
                                ),
                                "num_transition_bins": int(
                                    trace_record.get("num_transition_bins", 1)
                                ),
                                "persistent_unigram_enabled": bool(
                                    trace_record.get(
                                        "persistent_unigram_enabled", False
                                    )
                                ),
                                "persistent_unigram_mode": str(
                                    trace_record.get(
                                        "persistent_unigram_mode", "unknown"
                                    )
                                ),
                                "persistent_unigram_size_at_start": int(
                                    trace_record.get(
                                        "persistent_unigram_size_at_start", 0
                                    )
                                ),
                                "persistent_unigram_size": int(
                                    trace_record.get("persistent_unigram_size", 0)
                                ),
                                "persistent_context_size_at_start": int(
                                    trace_record.get(
                                        "persistent_context_size_at_start", 0
                                    )
                                ),
                                "persistent_context_size": int(
                                    trace_record.get("persistent_context_size", 0)
                                ),
                                "persistent_trigram_size_at_start": int(
                                    trace_record.get(
                                        "persistent_trigram_size_at_start", 0
                                    )
                                ),
                                "persistent_trigram_size": int(
                                    trace_record.get("persistent_trigram_size", 0)
                                ),
                                "persistent_shadow_size": int(
                                    trace_record.get("persistent_shadow_size", 0)
                                ),
                                "prompt_transition_row_count": int(
                                    trace_record.get(
                                        "prompt_transition_row_count", 0
                                    )
                                ),
                                "global_backoff_candidate_count": int(
                                    trace_record.get(
                                        "global_backoff_candidate_count", 0
                                    )
                                ),
                                "root_residual_candidate_count": int(
                                    trace_record.get(
                                        "root_residual_candidate_count", 0
                                    )
                                ),
                            }
                        )
    return records


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def assign_within_benchmark_deciles(
    records: List[dict], num_bins: int = 10
) -> List[dict]:
    """Assign tie-preserving ranks within each benchmark and protocol split."""

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["benchmark"], record.get("analysis_split", "all"))].append(
            index
        )
    for indices in groups.values():
        values = np.asarray(
            [records[index]["visual_probe_jsd"] for index in indices],
            dtype=np.float64,
        )
        ranks = _average_tie_ranks(values)
        bins = np.ceil(ranks * num_bins / len(indices)).astype(int)
        bins = np.clip(bins, 1, num_bins)
        percentiles = (ranks - 0.5) / max(len(indices), 1)
        for index, bin_index, percentile in zip(
            indices, bins.tolist(), percentiles.tolist()
        ):
            records[index]["visual_sensitivity_decile"] = int(bin_index)
            records[index]["visual_sensitivity_percentile"] = float(percentile)
    return records


def _rule(minimum_budget: int) -> dict:
    return {
        "rule_id": f"both_available_matched_budget_ge_{int(minimum_budget)}",
        "requires_u_available": True,
        "requires_gc_available": True,
        "minimum_matched_tree_node_budget": int(minimum_budget),
        "acceptance_metric": "equal-node-budget shadow acceptance",
    }


def _eligible(record: Mapping, rule: Mapping) -> bool:
    return bool(
        record.get("u_available", False)
        and record.get("gc_available", False)
        and int(record.get("matched_tree_node_budget", 0))
        >= int(rule["minimum_matched_tree_node_budget"])
    )


def _empty_stats() -> np.ndarray:
    return np.zeros((10, len(STAT_NAMES)), dtype=np.float64)


def _add_record(stats: np.ndarray, record: dict, rule: Mapping) -> None:
    decile = int(record["visual_sensitivity_decile"]) - 1
    values = np.zeros(len(STAT_NAMES), dtype=np.float64)
    values[:5] = (
        1.0,
        float(record["u_available"]),
        float(record["gc_available"]),
        float(record["u_top8_hit"] and record["u_available"]),
        float(record["gc_top8_hit"] and record["gc_available"]),
    )
    if _eligible(record, rule):
        values[5:] = (
            1.0,
            float(record["matched_gc_minus_u_accept"]),
            float(record["u_matched_accept"]),
            float(record["gc_matched_accept"]),
        )
    stats[decile] += values


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def _metrics_from_stats(stats: np.ndarray) -> np.ndarray:
    state_count = stats[..., 0]
    u_available = stats[..., 1]
    gc_available = stats[..., 2]
    eligible = stats[..., 5]
    return np.stack(
        (
            _safe_divide(u_available, state_count),
            _safe_divide(gc_available, state_count),
            _safe_divide(stats[..., 3], u_available),
            _safe_divide(stats[..., 4], gc_available),
            _safe_divide(stats[..., 6], eligible),
        ),
        axis=-1,
    )


def build_cluster_stats(records: Sequence[dict], rule: Mapping):
    grouped: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for record in records:
        stats = grouped[record["benchmark"]].setdefault(
            record["cluster_id"], _empty_stats()
        )
        _add_record(stats, record, rule)
    return {
        benchmark: np.stack(list(clusters.values()), axis=0)
        for benchmark, clusters in sorted(grouped.items())
    }


def _nanmean_stack(values: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(values, axis=0)
    valid = np.isfinite(stack)
    return _safe_divide(
        np.where(valid, stack, 0.0).sum(axis=0), valid.sum(axis=0)
    )


def point_estimates(cluster_stats: Dict[str, np.ndarray]):
    benchmark_stats = {
        benchmark: clusters.sum(axis=0)
        for benchmark, clusters in cluster_stats.items()
    }
    curves = {
        benchmark: _metrics_from_stats(stats)
        for benchmark, stats in benchmark_stats.items()
    }
    return benchmark_stats, curves, _nanmean_stack(list(curves.values()))


def _low_high_delta(stats: np.ndarray):
    low = stats[..., :2, :].sum(axis=-2)
    high = stats[..., 8:, :].sum(axis=-2)
    low_delta = _safe_divide(low[..., 6], low[..., 5])
    high_delta = _safe_divide(high[..., 6], high[..., 5])
    return low_delta, high_delta, high_delta - low_delta


def clustered_bootstrap(
    cluster_stats: Dict[str, np.ndarray],
    resamples: int = 50000,
    seed: int = 42,
    chunk_size: int = 500,
):
    """Resample image clusters within benchmark, then macro-average benches."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    curve_draws = np.empty(
        (resamples, 10, len(METRIC_NAMES)), dtype=np.float32
    )
    contrast_draws = np.empty((resamples, 3), dtype=np.float32)
    cursor = 0
    arrays = list(cluster_stats.values())
    while cursor < resamples:
        batch_size = min(chunk_size, resamples - cursor)
        batch_curves = []
        batch_contrasts = []
        for clusters in arrays:
            count = int(clusters.shape[0])
            weights = rng.multinomial(
                count, np.full(count, 1.0 / count), size=batch_size
            )
            sampled = np.einsum("bc,cds->bds", weights, clusters)
            batch_curves.append(_metrics_from_stats(sampled))
            batch_contrasts.append(np.stack(_low_high_delta(sampled), axis=-1))
        curve_draws[cursor : cursor + batch_size] = _nanmean_stack(
            batch_curves
        ).astype(np.float32)
        contrast_draws[cursor : cursor + batch_size] = _nanmean_stack(
            batch_contrasts
        ).astype(np.float32)
        cursor += batch_size
    return curve_draws, contrast_draws


def _ci(values: np.ndarray) -> List[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _position_adjusted_visual_slope(records: Sequence[dict], rule: Mapping):
    """OLS visual slope with output-position and benchmark controls."""

    selected = [row for row in records if _eligible(row, rule)]
    benchmarks = sorted({row["benchmark"] for row in selected})
    if len(selected) < 4 or not benchmarks:
        return {"estimate": None, "cluster_robust_95_ci": [None, None]}
    x_rows, y, cluster_ids = [], [], []
    for row in selected:
        x_rows.append(
            [
                1.0,
                float(row["visual_sensitivity_percentile"]),
                float(np.log1p(max(int(row.get("output_position", 0)), 0))),
                *[
                    float(row["benchmark"] == benchmark)
                    for benchmark in benchmarks[1:]
                ],
            ]
        )
        y.append(float(row["matched_gc_minus_u_accept"]))
        cluster_ids.append(row["cluster_id"])
    x = np.asarray(x_rows, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    beta = np.linalg.lstsq(x, y_array, rcond=None)[0]
    residual = y_array - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    clusters = sorted(set(cluster_ids))
    ids = np.asarray(cluster_ids)
    for cluster_id in clusters:
        score = x[ids == cluster_id].T @ residual[ids == cluster_id]
        meat += np.outer(score, score)
    n, p, g = len(y), x.shape[1], len(clusters)
    correction = (
        (g / (g - 1.0)) * ((n - 1.0) / (n - p))
        if g > 1 and n > p
        else 1.0
    )
    variance = correction * bread @ meat @ bread
    standard_error = float(np.sqrt(max(variance[1, 1], 0.0)))
    estimate = float(beta[1])
    return {
        "estimate": estimate,
        "cluster_robust_standard_error": standard_error,
        "cluster_robust_95_ci": [
            estimate - 1.96 * standard_error,
            estimate + 1.96 * standard_error,
        ],
        "num_states": int(n),
        "num_image_clusters": int(g),
        "controls": [
            "log1p(output_position)",
            f"benchmark fixed effects (reference={benchmarks[0]})",
        ],
    }


def summarize(
    records: List[dict],
    bootstrap_resamples: int = 50000,
    seed: int = 42,
    rule: Mapping | None = None,
    split_name: str = "all",
):
    """Summarize one preselected split under one fixed rule."""

    if not records:
        raise ValueError("cannot summarize an empty record set")
    for record in records:
        record.setdefault("analysis_split", split_name)
        record.setdefault("matched_tree_node_budget", 10**9)
        record.setdefault("u_matched_accept", record.get("u_shadow_accept", 0))
        record.setdefault("gc_matched_accept", record.get("gc_shadow_accept", 0))
        record.setdefault(
            "matched_gc_minus_u_accept",
            record.get(
                "gc_minus_u_shadow_accept",
                record["gc_matched_accept"] - record["u_matched_accept"],
            ),
        )
        record.setdefault("output_position", 0)
        record.setdefault("visual_probe_same_text_trajectory", True)
        record.setdefault("visual_probe_recomputed_vision_encoder", True)
        record.setdefault("visual_probe_full_top1_matches_target", True)
    if any("visual_sensitivity_decile" not in row for row in records):
        assign_within_benchmark_deciles(records)
    rule = dict(rule or _rule(0))
    cluster_stats = build_cluster_stats(records, rule)
    benchmark_stats, benchmark_curves, macro = point_estimates(cluster_stats)
    curve_draws, contrast_draws = clustered_bootstrap(
        cluster_stats, resamples=bootstrap_resamples, seed=seed
    )
    low_delta, high_delta, interaction = _nanmean_stack(
        [np.asarray(_low_high_delta(stats)) for stats in benchmark_stats.values()]
    )
    low_ci = _ci(contrast_draws[:, 0])
    high_ci = _ci(contrast_draws[:, 1])
    interaction_ci = _ci(contrast_draws[:, 2])
    adjusted = _position_adjusted_visual_slope(records, rule)
    adjusted_ci = adjusted.get("cluster_robust_95_ci", [None, None])
    adjusted_positive = bool(
        adjusted_ci[0] is not None and adjusted_ci[0] > 0.0
    )
    interaction_positive = bool(interaction_ci[0] > 0.0)
    true_crossover = bool(low_ci[1] < 0.0 and high_ci[0] > 0.0)
    if true_crossover and interaction_positive and adjusted_positive:
        verdict = "selective_reuse_dilemma_supported"
    elif interaction_positive and adjusted_positive:
        verdict = "positive_interaction_without_sign_crossover"
    else:
        verdict = "selective_reuse_dilemma_not_supported"

    overall_stats = sum(
        (stats.sum(axis=0) for stats in benchmark_stats.values()),
        np.zeros(len(STAT_NAMES), dtype=np.float64),
    )
    eligible = overall_stats[5]
    overall = {
        "eligible_states": int(eligible),
        "eligible_state_ratio": _json_number(eligible / overall_stats[0]),
        "u_matched_accept": _json_number(overall_stats[7] / eligible),
        "gc_matched_accept": _json_number(overall_stats[8] / eligible),
        "gc_minus_u_matched_accept": _json_number(overall_stats[6] / eligible),
    }

    macro_rows = []
    for decile in range(10):
        row = {
            "decile": decile + 1,
            "eligible_states": int(
                sum(stats[decile, 5] for stats in benchmark_stats.values())
            ),
        }
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            row[metric_name] = _json_number(macro[decile, metric_index])
            row[f"{metric_name}_ci"] = _ci(
                curve_draws[:, decile, metric_index]
            )
        macro_rows.append(row)

    by_benchmark = {}
    for benchmark, curve in benchmark_curves.items():
        stats = benchmark_stats[benchmark]
        by_benchmark[benchmark] = {
            "num_image_clusters": int(cluster_stats[benchmark].shape[0]),
            "num_states": int(stats[:, 0].sum()),
            "eligible_states": int(stats[:, 5].sum()),
            "deciles": [
                {
                    "decile": decile + 1,
                    "num_states": int(stats[decile, 0]),
                    "eligible_states": int(stats[decile, 5]),
                    **{
                        metric: _json_number(curve[decile, index])
                        for index, metric in enumerate(METRIC_NAMES)
                    },
                }
                for decile in range(10)
            ],
        }

    return {
        "schema_version": 2,
        "split": split_name,
        "rule": rule,
        "num_records": len(records),
        "num_image_clusters": len({row["cluster_id"] for row in records}),
        "num_benchmarks": len(cluster_stats),
        "benchmarks": sorted(cluster_stats),
        "visual_sensitivity": (
            "JSD(full image, four mean-patch-occluded images), recomputing "
            "the vision encoder and all text states on one fixed text path"
        ),
        "probe_audit": {
            "same_text_trajectory_ratio": float(
                np.mean(
                    [row["visual_probe_same_text_trajectory"] for row in records]
                )
            ),
            "vision_encoder_recomputed_ratio": float(
                np.mean(
                    [
                        row["visual_probe_recomputed_vision_encoder"]
                        for row in records
                    ]
                )
            ),
            "full_view_top1_matches_realized_token_ratio": float(
                np.mean(
                    [
                        row["visual_probe_full_top1_matches_target"]
                        for row in records
                    ]
                )
            ),
        },
        "deciles": "tie-preserving ranks within benchmark and split",
        "acceptance_estimand": (
            "GC-only minus U-only shadow acceptance, conditional on both "
            "sources and equal candidate-node budget"
        ),
        "bootstrap": {
            "unit": "image cluster",
            "stratification": "within benchmark",
            "macro_averaging": "equal benchmark weight",
            "resamples": int(bootstrap_resamples),
            "seed": int(seed),
        },
        "evidence_gate": {
            "verdict": verdict,
            "low_visual_delta": _json_number(low_delta),
            "low_visual_delta_95_ci": low_ci,
            "high_visual_delta": _json_number(high_delta),
            "high_visual_delta_95_ci": high_ci,
            "source_by_visual_interaction": _json_number(interaction),
            "source_by_visual_interaction_95_ci": interaction_ci,
            "position_adjusted_visual_slope": adjusted,
            "positive_interaction": bool(
                interaction_positive and adjusted_positive
            ),
            "sign_crossover": true_crossover,
            "strict_support_rule": (
                "held-out unadjusted interaction CI > 0, adjusted visual "
                "slope CI > 0, low-visual delta CI < 0, and high-visual "
                "delta CI > 0"
            ),
        },
        "overall_matched_acceptance": overall,
        "macro_deciles": macro_rows,
        "by_benchmark": by_benchmark,
    }


def select_discovery_rule(records: List[dict]) -> tuple[dict, dict]:
    """Choose once from a predeclared family, using discovery only."""

    candidate_rows = []
    benchmark_count = len({row["benchmark"] for row in records})
    minimum_states = max(20, 4 * benchmark_count)
    for threshold in PREDECLARED_MATCHED_BUDGETS:
        rule = _rule(threshold)
        eligible = [row for row in records if _eligible(row, rule)]
        grouped = defaultdict(list)
        for row in eligible:
            grouped[row["benchmark"]].append(row)
        low_values, high_values = [], []
        for rows in grouped.values():
            low = [
                row["matched_gc_minus_u_accept"]
                for row in rows
                if row["visual_sensitivity_decile"] <= 2
            ]
            high = [
                row["matched_gc_minus_u_accept"]
                for row in rows
                if row["visual_sensitivity_decile"] >= 9
            ]
            if low and high:
                low_values.append(float(np.mean(low)))
                high_values.append(float(np.mean(high)))
        interaction = (
            float(np.mean(high_values) - np.mean(low_values))
            if low_values and high_values
            else None
        )
        valid_benchmarks = len(low_values)
        valid = bool(
            len(eligible) >= minimum_states
            and valid_benchmarks >= max(2, benchmark_count // 2)
            and interaction is not None
        )
        candidate_rows.append(
            {
                **rule,
                "eligible_discovery_states": len(eligible),
                "eligible_discovery_state_ratio": len(eligible) / len(records),
                "valid_benchmarks": valid_benchmarks,
                "low_visual_macro_delta": (
                    float(np.mean(low_values)) if low_values else None
                ),
                "high_visual_macro_delta": (
                    float(np.mean(high_values)) if high_values else None
                ),
                "discovery_interaction": interaction,
                "eligible_for_selection": valid,
            }
        )
    selectable = [row for row in candidate_rows if row["eligible_for_selection"]]
    chosen = (
        max(
            selectable,
            key=lambda row: (
                float(row["discovery_interaction"]),
                -int(row["minimum_matched_tree_node_budget"]),
            ),
        )
        if selectable
        else candidate_rows[0]
    )
    chosen_rule = _rule(chosen["minimum_matched_tree_node_budget"])
    return chosen_rule, {
        "selection_scope": "discovery only",
        "selection_objective": (
            "maximize macro high-minus-low visual contrast over the "
            "predeclared matched-node-budget family"
        ),
        "minimum_eligible_discovery_states": minimum_states,
        "tie_breaker": "prefer the less restrictive node threshold",
        "candidate_rules": candidate_rows,
        "chosen_rule": chosen_rule,
        "heldout_access_during_selection": False,
    }


def select_discovery_exemplars(
    records: Sequence[dict], rule: Mapping, per_side: int = 12
) -> List[dict]:
    """Outcome-select discovery cases, permanently labeled non-inferential."""

    eligible = [row for row in records if _eligible(row, rule)]

    def choose(candidates, reverse):
        candidates = sorted(
            candidates,
            key=lambda row: (
                float(row["matched_gc_minus_u_accept"]),
                float(row["visual_probe_jsd"]),
            ),
            reverse=reverse,
        )
        output, used_clusters = [], set()
        for row in candidates:
            if row["cluster_id"] in used_clusters:
                continue
            copied = dict(row)
            copied["selection_reason"] = (
                "high-visual GC-favored discovery exemplar"
                if reverse
                else "low-visual U-favored discovery exemplar"
            )
            copied["statistical_evidence"] = False
            output.append(copied)
            used_clusters.add(row["cluster_id"])
            if len(output) >= int(per_side):
                break
        return output

    low = [
        row
        for row in eligible
        if row["visual_sensitivity_decile"] <= 2
        and row["matched_gc_minus_u_accept"] < 0
    ]
    high = [
        row
        for row in eligible
        if row["visual_sensitivity_decile"] >= 9
        and row["matched_gc_minus_u_accept"] > 0
    ]
    return choose(low, reverse=False) + choose(high, reverse=True)


def write_records(records: Sequence[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_benchmark_csv(summaries: Mapping[str, dict], output_path: Path) -> None:
    fieldnames = [
        "split",
        "benchmark",
        "decile",
        "num_states",
        "eligible_states",
        *METRIC_NAMES,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, summary in summaries.items():
            for benchmark, payload in summary["by_benchmark"].items():
                for row in payload["deciles"]:
                    writer.writerow({"split": split, "benchmark": benchmark, **row})


def plot_summary(summary: dict, output_stem: Path, *, exploratory: bool) -> None:
    rows = summary["macro_deciles"]
    x = np.arange(1, 11)
    colors = {"U": "#E67E22", "GC": "#2468B4"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.75), constrained_layout=True)

    def draw(axis, metric, label, color):
        y = np.asarray(
            [np.nan if row[metric] is None else row[metric] for row in rows]
        )
        ci = np.asarray([row[f"{metric}_ci"] for row in rows], dtype=float)
        axis.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
        axis.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.16)

    draw(axes[0], "u_coverage", "U", colors["U"])
    draw(axes[0], "gc_coverage", "G/C", colors["GC"])
    axes[0].set_title("(a) Source coverage (all states)")
    axes[0].set_ylabel("Available-source probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False)
    draw(axes[1], "u_conditional_top8_recall", "U", colors["U"])
    draw(axes[1], "gc_conditional_top8_recall", "G/C", colors["GC"])
    axes[1].set_title("(b) Recall when source exists")
    axes[1].set_ylabel("Target-token top-8 recall")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False)
    draw(
        axes[2],
        "matched_gc_minus_u_accept",
        "G/C minus U (equal nodes)",
        colors["GC"],
    )
    axes[2].axhline(0.0, color="#666666", linewidth=1.1, linestyle="--")
    axes[2].set_title("(c) Conditional source utility")
    axes[2].set_ylabel("Accepted-token difference per state")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Within-benchmark visual-sensitivity decile")
        axis.set_xticks(x)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    label = (
        "EXPLORATORY — rule selected on discovery; not confirmatory evidence"
        if exploratory
        else "CONFIRMATORY HELD-OUT — discovery rule frozen before analysis"
    )
    fig.suptitle(
        f"{label}\n{summary['rule']['rule_id']} · "
        f"{summary['num_benchmarks']} benchmarks",
        fontsize=11.5,
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_exemplars(
    discovery_records: Sequence[dict],
    exemplars: Sequence[dict],
    rule: Mapping,
    output_stem: Path,
) -> None:
    eligible = [row for row in discovery_records if _eligible(row, rule)]
    selected_keys = {
        (row["benchmark"], row["question_id"], row["turn_index"], row["iteration"])
        for row in exemplars
    }
    fig, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    axis.scatter(
        [row["visual_probe_jsd"] for row in eligible],
        [row["matched_gc_minus_u_accept"] for row in eligible],
        s=8,
        alpha=0.14,
        color="#777777",
        label="all eligible discovery states",
    )
    chosen = [
        row
        for row in eligible
        if (
            row["benchmark"],
            row["question_id"],
            row["turn_index"],
            row["iteration"],
        )
        in selected_keys
    ]
    axis.scatter(
        [row["visual_probe_jsd"] for row in chosen],
        [row["matched_gc_minus_u_accept"] for row in chosen],
        s=42,
        alpha=0.9,
        color="#C73E1D",
        label="outcome-selected exemplars",
    )
    axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axis.set_xlabel("True-occlusion visual JSD")
    axis.set_ylabel("G/C minus U matched acceptance")
    axis.set_title(
        "DISCOVERY CASE STUDY ONLY\nSelected exemplars are not statistical evidence"
    )
    axis.legend(frameon=False)
    axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no policy results.jsonl files were found")
    records = load_selective_records(paths)
    if not records:
        parser.error("no selective-reuse diagnostic states were found")
    protocols = sorted({row["visual_probe_protocol"] for row in records})
    if protocols != [TRUE_PROBE_PROTOCOL]:
        parser.error(
            "confirmatory analysis requires true teacher-forced pixel probes; "
            f"found protocols={protocols}"
        )
    splits = {row["analysis_split"] for row in records}
    if splits != {"discovery", "heldout"}:
        parser.error(
            "expected discovery and heldout records; found " + repr(sorted(splits))
        )

    assign_within_benchmark_deciles(records)
    discovery = [row for row in records if row["analysis_split"] == "discovery"]
    heldout = [row for row in records if row["analysis_split"] == "heldout"]
    frozen_rule, selection_log = select_discovery_rule(discovery)
    discovery_summary = summarize(
        discovery,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        rule=frozen_rule,
        split_name="discovery",
    )
    heldout_summary = summarize(
        heldout,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed + 1,
        rule=frozen_rule,
        split_name="heldout",
    )
    exemplars = select_discovery_exemplars(discovery, frozen_rule)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 2,
        "protocol": "discovery_rule_selection_then_single_heldout_analysis",
        "policy": args.policy,
        "input_paths": [str(path) for path in paths],
        "visual_probe_protocols": protocols,
        "frozen_rule": frozen_rule,
        "selection_log_path": str(
            (args.output_dir / "selection_log.json").resolve()
        ),
        "discovery": discovery_summary,
        "heldout": heldout_summary,
        "evidence_gate": heldout_summary["evidence_gate"],
        "reporting_policy": {
            "confirmatory_result": "heldout only",
            "discovery_figure": "exploratory, rule-selected",
            "selected_exemplars": (
                "outcome-selected discovery cases; never statistical evidence"
            ),
            "speed_metrics": (
                "invalid in this diagnostic run because untimed probes may "
                "affect the thermal/load state of later samples"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selection_log.json").write_text(
        json.dumps(selection_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "frozen_rule.json").write_text(
        json.dumps(frozen_rule, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_records(records, args.output_dir / "selective_reuse_records.jsonl")
    write_records(
        exemplars, args.output_dir / "discovery_selected_exemplars.jsonl"
    )
    write_benchmark_csv(
        {"discovery": discovery_summary, "heldout": heldout_summary},
        args.output_dir / "by_benchmark_decile.csv",
    )
    plot_summary(
        discovery_summary,
        args.output_dir / "discovery_exploratory_figure",
        exploratory=True,
    )
    plot_summary(
        heldout_summary,
        args.output_dir / "heldout_confirmatory_figure",
        exploratory=False,
    )
    # Historical filename now aliases the held-out figure, never discovery.
    plot_summary(
        heldout_summary,
        args.output_dir / "selective_reuse_figure",
        exploratory=False,
    )
    plot_exemplars(
        discovery,
        exemplars,
        frozen_rule,
        args.output_dir / "discovery_selected_exemplars",
    )
    print(json.dumps(summary["evidence_gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
