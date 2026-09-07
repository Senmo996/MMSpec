import json
from types import SimpleNamespace

from evaluation.promote_validated_continuous_figure import (
    EXPECTED_BENCHMARKS,
    evaluate_gates,
)


PROTOCOL = "teacher_forced_mean_patch_occlusion_v1"
METRIC = "visual_target_drop_fraction"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_fixture(tmp_path):
    root = tmp_path / "run"
    analysis_dir = root / "analysis"
    fixed_manifest = tmp_path / "fixed_summary.json"
    mmspec_audit = tmp_path / "mmspec_audit.json"
    support = {benchmark: 30 for benchmark in EXPECTED_BENCHMARKS}
    write_json(
        analysis_dir / "summary.json",
        {
            "analysis_role": "outcome_blind_secondary_validation",
            "analysis_split": "all",
            "benchmarks": sorted(EXPECTED_BENCHMARKS),
            "excluded_benchmarks": ["MME"],
            "visual_probe_protocols": [PROTOCOL],
            "probe_audit": {
                "same_text_trajectory_ratio": 1.0,
                "recomputed_vision_encoder_ratio": 1.0,
            },
            "frozen_rule": {
                "visual_metric": METRIC,
                "visual_percentile_reference_population": "eligible_conflicts",
            },
            "visual_percentile_audit": {
                "reference_population": "eligible_conflicts",
                "uses_source_hit_outcomes": False,
                "states_by_benchmark": support,
                "num_reference_states": sum(support.values()),
            },
            "continuous_by_benchmark": {
                benchmark: {} for benchmark in EXPECTED_BENCHMARKS
            },
            "empirical_conflict_tail_by_benchmark": {
                benchmark: {} for benchmark in EXPECTED_BENCHMARKS
            },
            "empirical_conflict_tail_valid_all_benchmark_bootstrap_draws": 9500,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 161803,
            "evidence_gate": {
                "positive_continuous_interaction": True,
                "p10_u_favored": True,
                "p90_gc_favored": True,
                "strict_continuous_crossover": True,
                "observed_low_u_favored": True,
                "observed_high_gc_favored": True,
                "observed_positive_interaction": True,
                "strict_observed_tail_crossover": True,
                "strict_joint_crossover": True,
            },
            "cluster_bootstrap_95_ci": {
                "delta_slope": [0.1, 0.3],
                "delta_p10": [-0.2, -0.01],
                "delta_p90": [0.01, 0.2],
            },
            "empirical_conflict_tail_cluster_bootstrap_95_ci": {
                "low_delta": [-0.2, -0.01],
                "high_delta": [0.01, 0.2],
                "interaction": [0.1, 0.3],
            },
        },
    )
    write_json(root / "completion_audit.json", {"status": "passed", "failures": []})
    write_json(
        fixed_manifest,
        {
            "all_source_index_overlaps_zero": True,
            "all_image_cluster_overlaps_zero": True,
        },
    )
    write_json(
        mmspec_audit,
        {
            "all_unique_within_range": True,
            "all_pairwise_overlaps_zero": True,
        },
    )
    args = SimpleNamespace(
        output_root=root,
        analysis_dir="analysis",
        fixed_manifest_summary=fixed_manifest,
        mmspec_range_audit=mmspec_audit,
        expected_probe_protocol=PROTOCOL,
        expected_visual_metric=METRIC,
    )
    return args, analysis_dir / "summary.json"


def test_continuous_promotion_accepts_only_complete_strict_result(tmp_path):
    args, _summary_path = valid_fixture(tmp_path)

    failures, _evidence = evaluate_gates(args)

    assert failures == []


def test_continuous_promotion_rejects_nonsignificant_low_endpoint(tmp_path):
    args, summary_path = valid_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidence_gate"]["p10_u_favored"] = False
    summary["evidence_gate"]["strict_continuous_crossover"] = False
    summary["cluster_bootstrap_95_ci"]["delta_p10"] = [-0.2, 0.01]
    write_json(summary_path, summary)

    failures, _evidence = evaluate_gates(args)

    assert "evidence gate failed: p10_u_favored" in failures
    assert "strict_continuous_crossover is not true" in failures
    assert "delta_p10 interval is not strictly negative" in failures


def test_continuous_promotion_rejects_sparse_or_outcome_based_ranks(tmp_path):
    args, summary_path = valid_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["visual_percentile_audit"]["uses_source_hit_outcomes"] = True
    summary["visual_percentile_audit"]["states_by_benchmark"]["OCRBench"] = 29
    summary["visual_percentile_audit"]["num_reference_states"] -= 1
    write_json(summary_path, summary)

    failures, _evidence = evaluate_gates(args)

    assert "visual percentile assignment is not outcome-independent" in failures
    assert "fewer than 30 eligible conflicts for benchmarks: OCRBench" in failures


def test_continuous_promotion_rejects_fitted_crossover_without_observed_crossover(
    tmp_path,
):
    args, summary_path = valid_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidence_gate"]["observed_low_u_favored"] = False
    summary["evidence_gate"]["strict_observed_tail_crossover"] = False
    summary["evidence_gate"]["strict_joint_crossover"] = False
    summary["empirical_conflict_tail_cluster_bootstrap_95_ci"]["low_delta"] = [
        -0.1,
        0.05,
    ]
    write_json(summary_path, summary)

    failures, _evidence = evaluate_gates(args)

    assert "evidence gate failed: observed_low_u_favored" in failures
    assert "strict_observed_tail_crossover is not true" in failures
    assert "strict_joint_crossover is not true" in failures
    assert "empirical low_delta interval is not strictly negative" in failures
