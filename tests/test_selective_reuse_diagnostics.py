import math

import torch

from evaluation.analyze_selective_reuse import (
    assign_within_benchmark_deciles,
    summarize,
)
from evaluation.analyze_selective_visual_injection import (
    _macro_tail_stats,
    assign_deciles as assign_visual_injection_deciles,
    clustered_bootstrap as bootstrap_visual_injection,
    validate_and_filter as validate_visual_injection,
)
from evaluation.analyze_selective_source_conflict import (
    assign_deciles as assign_source_conflict_deciles,
    point_estimates as source_conflict_point_estimates,
    validate_and_filter as validate_source_conflict,
)
from evaluation.analyze_wrong_image_control import summarize_pairs
from evaluation.eval_selective_reuse_wrong_image import build_pairing
from evaluation.eval_sam_grounded_fixed_multi import (
    _image_cluster_identity,
    build_parser,
)
from evaluation.eval_sam_grounded_mmspec import _policy_kwargs
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel


POLICY = (
    "context-score-trigram-fusion-persistent-suffix4-visualcache-"
    "hotpath-cpp-depth10-node63-wide-plus4"
)


def _synthetic_rows(low_delta=-2, high_delta=2):
    rows = []
    for benchmark in ("A", "B"):
        for cluster in range(20):
            for state in range(2):
                rank = cluster * 2 + state
                high = rank >= 20
                delta = high_delta if high else low_delta
                u_accept = max(-delta, 0)
                gc_accept = max(delta, 0)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "cluster_id": f"{benchmark}:{cluster}",
                        "visual_probe_jsd": float(rank),
                        "u_available": True,
                        "gc_available": True,
                        "u_top8_hit": not high,
                        "gc_top8_hit": high,
                        "u_shadow_accept": u_accept,
                        "gc_shadow_accept": gc_accept,
                        "fixed_accept": max(u_accept, gc_accept),
                        "oracle_accept": max(u_accept, gc_accept),
                        "gc_minus_u_shadow_accept": delta,
                    }
                )
    return rows


def test_longest_matching_shadow_path_uses_realized_continuation():
    paths = [[7, 8], [7, 9, 10], [6, 5, 4]]
    assert TreeRecyclingSpecModel._longest_matching_path(paths, [7, 9, 11]) == 2
    assert TreeRecyclingSpecModel._longest_matching_path(paths, [1, 2]) == 0
    assert TreeRecyclingSpecModel._longest_matching_path([], [7]) == 0


def test_visual_probe_metrics_report_jsd_union_and_top1_disagreement():
    logits = torch.tensor(
        [[5.0, 1.0, 0.0], [4.0, 2.0, 0.0], [0.0, 5.0, 1.0]]
    )
    metrics = TreeRecyclingSpecModel._visual_probe_metrics(logits, top_k=1)
    assert metrics["visual_probe_jsd"] > 0.0
    assert metrics["visual_probe_topk_union_size"] == 2
    assert math.isclose(
        metrics["visual_probe_top1_disagreement_rate"], 0.5
    )


def test_diagnostic_flag_is_candidate_only_and_preserves_policy_name():
    args = build_parser().parse_args(
        [
            "--base-model-path",
            "model",
            "--output-root",
            "output",
            "--manifest-dir",
            "manifest",
            "--policies",
            f"target,{POLICY}",
            "--candidate-policy",
            POLICY,
            "--selective-reuse-diagnostics",
        ]
    )
    target = _policy_kwargs(args, "target")
    candidate = _policy_kwargs(args, POLICY)
    assert target["selective_reuse_diagnostics"] is False
    assert candidate["selective_reuse_diagnostics"] is True
    assert candidate["draft_policy"] == POLICY


def test_tied_visual_sensitivity_values_are_not_split_across_deciles():
    rows = _synthetic_rows()
    for row in rows[:10]:
        row["visual_probe_jsd"] = 0.0
    assign_within_benchmark_deciles(rows)
    for benchmark in ("A", "B"):
        tied = [
            row["visual_sensitivity_decile"]
            for row in rows
            if row["benchmark"] == benchmark
            and row["visual_probe_jsd"] == 0.0
        ]
        assert len(set(tied)) == 1


def test_cluster_bootstrap_supports_true_crossover():
    payload = summarize(_synthetic_rows(), bootstrap_resamples=200, seed=3)
    gate = payload["evidence_gate"]
    assert gate["verdict"] == "selective_reuse_dilemma_supported"
    assert gate["positive_interaction"] is True
    assert gate["sign_crossover"] is True


def test_cluster_bootstrap_downgrades_interaction_without_crossover():
    payload = summarize(
        _synthetic_rows(low_delta=1, high_delta=2),
        bootstrap_resamples=200,
        seed=3,
    )
    gate = payload["evidence_gate"]
    assert gate["verdict"] == "positive_interaction_without_sign_crossover"
    assert gate["positive_interaction"] is True
    assert gate["sign_crossover"] is False


def test_equal_budget_visual_injection_detects_root_aligned_interaction():
    rows = []
    for benchmark in ("A", "B"):
        for cluster in range(20):
            high = cluster >= 16
            low = cluster < 4
            rows.append(
                {
                    "benchmark": benchmark,
                    "cluster_id": f"{benchmark}:{cluster}",
                    "visual_probe_jsd": float(cluster),
                    "u_available": True,
                    "v_available": True,
                    "v_source": "prompt_visual_max",
                    "uv_visual_slots": 1,
                    "uv_candidate_budget": 8,
                    "u_root_candidate_token_ids": list(range(8)),
                    "uv_root_candidate_token_ids": list(range(7)) + [9],
                    "u_top8_hit": low,
                    "uv_top8_hit": high,
                }
            )
    assign_visual_injection_deciles(rows)
    eligible, audit = validate_visual_injection(rows)
    point, per_benchmark = _macro_tail_stats(eligible)
    draws = bootstrap_visual_injection(eligible, resamples=100, seed=7)

    assert audit["eligible_states"] == 40
    assert set(per_benchmark) == {"A", "B"}
    assert point.tolist() == [-1.0, 1.0, 2.0]
    assert float(draws[:, 2].min()) == 2.0


def test_source_conflict_filter_is_candidate_only_and_root_aligned():
    rows = []
    for benchmark in ("A", "B"):
        for cluster in range(20):
            high = cluster >= 16
            low = cluster < 4
            target_token_id = 0 if low else (8 if high else 99)
            rows.append(
                {
                    "benchmark": benchmark,
                    "cluster_id": f"{benchmark}:{cluster}",
                    "visual_probe_jsd": float(cluster),
                    "u_available": True,
                    "gc_available": True,
                    "u_root_candidate_token_ids": list(range(8)),
                    "gc_root_candidate_token_ids": list(range(7, 15)),
                    "target_token_id": target_token_id,
                    "u_top8_hit": target_token_id in range(8),
                    "gc_top8_hit": target_token_id in range(7, 15),
                }
            )
    assign_source_conflict_deciles(rows)
    eligible, audit = validate_source_conflict(rows)
    point, by_benchmark = source_conflict_point_estimates(eligible)

    assert audit["eligible_states"] == 40
    assert all(row["candidate_set_overlap"] == 1 for row in eligible)
    assert set(by_benchmark) == {"A", "B"}
    assert point.tolist() == [-1.0, 1.0, 2.0]


def test_wrong_image_pairing_is_distinct_and_same_category_when_possible():
    rows = [
        {"_fixed_index": 0, "category": "x", "image_id": "a"},
        {"_fixed_index": 1, "category": "x", "image_id": "b"},
        {"_fixed_index": 2, "category": "y", "image_id": "c"},
    ]
    pairs = build_pairing(rows, None, count=2, seed=42, name="toy")
    assert [pair["target_position"] for pair in pairs] == [0, 1]
    assert all(pair["target_image_identity"] != pair["source_image_identity"] for pair in pairs)
    assert all(not pair["used_category_fallback"] for pair in pairs)


def test_image_cluster_identity_prefers_bytes_over_generic_archive_path():
    first = _image_cluster_identity(
        "ScienceQA",
        {"image": {"path": "image.png", "bytes": b"first"}},
        {"source_index": 1},
    )
    second = _image_cluster_identity(
        "ScienceQA",
        {"image": {"path": "image.png", "bytes": b"second"}},
        {"source_index": 2},
    )

    assert first != second
    assert ":bytes:" in first


def test_wrong_image_summary_macro_averages_benchmarks():
    pairs = []
    for benchmark, changed in (("A", True), ("B", False)):
        for index in range(4):
            pairs.append(
                {
                    "benchmark": benchmark,
                    "question_id": f"{benchmark}:{index}",
                    "output_changed": changed,
                    "original_mean_visual_jsd": 0.1,
                    "wrong_mean_visual_jsd": 0.2,
                    "wrong_minus_original_mean_visual_jsd": 0.1,
                    "original_mean_gc_minus_u_accept": -0.5,
                    "wrong_mean_gc_minus_u_accept": 0.5,
                    "original_num_states": 3,
                    "wrong_num_states": 3,
                    "same_category_pair": True,
                }
            )
    payload = summarize_pairs(pairs, bootstrap_resamples=100, seed=4)
    assert payload["num_benchmarks"] == 2
    assert payload["metrics"]["output_changed"]["estimate"] == 0.5
    assert payload["same_category_pair_ratio"] == 1.0
