"""Promote a selective-reuse triptych only after all evidence gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED_BENCHMARKS = {
    "MMSpec",
    "MMT-Bench",
    "SEEDBench",
    "ScienceQA",
    "OCRBench",
    "ChartQA",
    "MathVista",
    "TextVQA",
}
DEFAULT_EXPECTED_VISUAL_METRIC = "visual_target_drop_fraction"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_gates(args) -> tuple[list[str], dict]:
    failures = []
    try:
        summary = load_json(args.output_root / args.analysis_dir / "summary.json")
        completion = load_json(args.output_root / "completion_audit.json")
        fixed_manifest = load_json(args.fixed_manifest_summary)
        mmspec_audit = load_json(args.mmspec_range_audit)
    except ValueError as error:
        return [str(error)], {}

    if summary.get("analysis_role") != "new_validation":
        failures.append("analysis_role is not new_validation")
    if summary.get("analysis_split") != "all":
        failures.append("analysis_split is not all")
    if set(summary.get("benchmarks", [])) != EXPECTED_BENCHMARKS:
        failures.append("analysis does not contain the expected eight benchmarks")
    if summary.get("excluded_benchmarks") != ["MME"]:
        failures.append("analysis does not explicitly exclude MME")
    if summary.get("visual_probe_protocols") != [args.expected_probe_protocol]:
        failures.append("visual probe protocol does not match the frozen protocol")
    expected_visual_metric = getattr(
        args, "expected_visual_metric", DEFAULT_EXPECTED_VISUAL_METRIC
    )
    if summary.get("frozen_rule", {}).get("visual_metric") != expected_visual_metric:
        failures.append("visual metric does not match the frozen rule")
    evidence = summary.get("evidence_gate", {})
    for gate in ("low_u_favored", "high_gc_favored", "positive_interaction"):
        if evidence.get(gate) is not True:
            failures.append(f"evidence gate failed: {gate}")
    if evidence.get("strict_crossover") is not True:
        failures.append("strict_crossover is not true")
    if completion.get("status") != "passed" or completion.get("failures"):
        failures.append("completion audit did not pass cleanly")
    if fixed_manifest.get("all_source_index_overlaps_zero") is not True:
        failures.append("fixed manifest source-index disjointness failed")
    if fixed_manifest.get("all_image_cluster_overlaps_zero") is not True:
        failures.append("fixed manifest image-cluster disjointness failed")
    if mmspec_audit.get("all_unique_within_range") is not True:
        failures.append("MMSpec within-range uniqueness failed")
    if mmspec_audit.get("all_pairwise_overlaps_zero") is not True:
        failures.append("MMSpec pairwise range disjointness failed")
    return failures, {
        "summary": summary,
        "completion": completion,
        "fixed_manifest": fixed_manifest,
        "mmspec_audit": mmspec_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--fixed-manifest-summary", type=Path, required=True)
    parser.add_argument("--mmspec-range-audit", type=Path, required=True)
    parser.add_argument("--expected-probe-protocol", required=True)
    parser.add_argument(
        "--expected-visual-metric",
        default=DEFAULT_EXPECTED_VISUAL_METRIC,
    )
    parser.add_argument(
        "--promotion-dir", type=Path, default=Path("publication_ready")
    )
    args = parser.parse_args()

    failures, evidence = evaluate_gates(args)
    promotion_dir = args.output_root / args.promotion_dir
    decision_path = args.output_root / "promotion_decision.json"
    payload = {
        "schema_version": 1,
        "status": "rejected" if failures else "promoted",
        "output_root": str(args.output_root.resolve()),
        "analysis_dir": str((args.output_root / args.analysis_dir).resolve()),
        "expected_probe_protocol": args.expected_probe_protocol,
        "expected_visual_metric": args.expected_visual_metric,
        "fixed_manifest_summary": str(args.fixed_manifest_summary.resolve()),
        "mmspec_range_audit": str(args.mmspec_range_audit.resolve()),
        "failures": failures,
    }
    if not failures:
        source_dir = args.output_root / args.analysis_dir
        source_figures = [
            source_dir / "selective_conflict_triptych.pdf",
            source_dir / "selective_conflict_triptych.png",
        ]
        missing_figures = [
            path for path in source_figures if not path.is_file() or path.stat().st_size == 0
        ]
        if missing_figures:
            failures.extend(f"missing source figure: {path}" for path in missing_figures)
            payload["status"] = "rejected"
            payload["failures"] = failures
    if not failures:
        promotion_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for source in source_figures:
            suffix = source.suffix
            destination = promotion_dir / f"validated_selective_reuse_triptych{suffix}"
            shutil.copy2(source, destination)
            artifacts.append(
                {
                    "path": str(destination.resolve()),
                    "sha256": sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        payload["artifacts"] = artifacts
        payload["point_estimates"] = evidence["summary"].get("point_estimates")
        payload["confidence_intervals_95"] = evidence["summary"].get(
            "cluster_bootstrap_95_ci"
        )
        payload["evidence_gate"] = evidence["summary"].get("evidence_gate")
        (promotion_dir / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    decision_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "num_failures": len(payload["failures"]),
                "decision_path": str(decision_path),
            },
            indent=2,
        )
    )
    if payload["status"] != "promoted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
