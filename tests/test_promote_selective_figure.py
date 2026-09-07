import json
from types import SimpleNamespace

from evaluation.promote_validated_selective_figure import (
    EXPECTED_BENCHMARKS,
    evaluate_gates,
)


PROTOCOL = "teacher_forced_mean_patch_occlusion_v1"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_fixture(tmp_path):
    root = tmp_path / "run"
    analysis_dir = root / "analysis"
    fixed_manifest = tmp_path / "fixed_summary.json"
    mmspec_audit = tmp_path / "mmspec_audit.json"
    write_json(
        analysis_dir / "summary.json",
        {
            "analysis_role": "new_validation",
            "analysis_split": "all",
            "benchmarks": sorted(EXPECTED_BENCHMARKS),
            "excluded_benchmarks": ["MME"],
            "visual_probe_protocols": [PROTOCOL],
            "frozen_rule": {
                "visual_metric": "visual_target_drop_fraction",
            },
            "evidence_gate": {
                "low_u_favored": True,
                "high_gc_favored": True,
                "positive_interaction": True,
                "strict_crossover": True,
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
    )
    return args, analysis_dir / "summary.json"


def test_promotion_gate_accepts_complete_strict_validation(tmp_path):
    args, _summary_path = valid_fixture(tmp_path)

    failures, evidence = evaluate_gates(args)

    assert failures == []
    assert evidence["summary"]["evidence_gate"]["strict_crossover"] is True


def test_promotion_gate_rejects_non_strict_crossover(tmp_path):
    args, summary_path = valid_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidence_gate"]["low_u_favored"] = False
    summary["evidence_gate"]["strict_crossover"] = False
    write_json(summary_path, summary)

    failures, _evidence = evaluate_gates(args)

    assert "evidence gate failed: low_u_favored" in failures
    assert "strict_crossover is not true" in failures


def test_promotion_gate_rejects_visual_metric_mismatch(tmp_path):
    args, summary_path = valid_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["frozen_rule"]["visual_metric"] = (
        "local2_mean_target_drop_fraction"
    )
    write_json(summary_path, summary)

    failures, _evidence = evaluate_gates(args)

    assert "visual metric does not match the frozen rule" in failures
