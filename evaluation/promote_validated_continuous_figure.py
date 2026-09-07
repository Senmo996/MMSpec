"""Promote the conflict-conditioned triptych only after strict audit gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
EXPECTED_ROLE = "outcome_blind_secondary_validation"
EXPECTED_REFERENCE_POPULATION = "eligible_conflicts"
EXPECTED_BOOTSTRAP_RESAMPLES = 10000
EXPECTED_BOOTSTRAP_SEED = 161803
MIN_REFERENCE_STATES_PER_BENCHMARK = 30
MIN_EMPIRICAL_TAIL_VALID_DRAW_FRACTION = 0.90


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


def _valid_interval(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        and value[0] <= value[1]
    )


def evaluate_gates(args) -> tuple[list[str], dict]:
    failures = []
    try:
        summary = load_json(args.output_root / args.analysis_dir / "summary.json")
        completion = load_json(args.output_root / "completion_audit.json")
        fixed_manifest = load_json(args.fixed_manifest_summary)
        mmspec_audit = load_json(args.mmspec_range_audit)
    except ValueError as error:
        return [str(error)], {}

    if summary.get("analysis_role") != EXPECTED_ROLE:
        failures.append(f"analysis_role is not {EXPECTED_ROLE}")
    if summary.get("analysis_split") != "all":
        failures.append("analysis_split is not all")
    if set(summary.get("benchmarks", [])) != EXPECTED_BENCHMARKS:
        failures.append("analysis does not contain the expected eight benchmarks")
    if summary.get("excluded_benchmarks") != ["MME"]:
        failures.append("analysis does not explicitly exclude MME")
    if summary.get("visual_probe_protocols") != [args.expected_probe_protocol]:
        failures.append("visual probe protocol does not match the frozen protocol")

    rule = summary.get("frozen_rule", {})
    if rule.get("visual_metric") != args.expected_visual_metric:
        failures.append("visual metric does not match the frozen rule")
    if (
        rule.get("visual_percentile_reference_population")
        != EXPECTED_REFERENCE_POPULATION
    ):
        failures.append("visual percentile is not conflict-conditioned")

    probe = summary.get("probe_audit", {})
    if probe.get("same_text_trajectory_ratio") != 1.0:
        failures.append("same-text trajectory audit failed")
    if probe.get("recomputed_vision_encoder_ratio") != 1.0:
        failures.append("vision-encoder recomputation audit failed")

    percentile = summary.get("visual_percentile_audit", {})
    if percentile.get("reference_population") != EXPECTED_REFERENCE_POPULATION:
        failures.append("visual percentile audit has the wrong population")
    if percentile.get("uses_source_hit_outcomes") is not False:
        failures.append("visual percentile assignment is not outcome-independent")
    states = percentile.get("states_by_benchmark", {})
    if set(states) != EXPECTED_BENCHMARKS:
        failures.append("visual percentile audit is missing benchmark support")
    else:
        sparse = sorted(
            benchmark
            for benchmark, count in states.items()
            if not isinstance(count, int)
            or count < MIN_REFERENCE_STATES_PER_BENCHMARK
        )
        if sparse:
            failures.append(
                "fewer than 30 eligible conflicts for benchmarks: "
                + ", ".join(sparse)
            )
        if percentile.get("num_reference_states") != sum(states.values()):
            failures.append("visual percentile state total is inconsistent")

    if set(summary.get("continuous_by_benchmark", {})) != EXPECTED_BENCHMARKS:
        failures.append("continuous model is missing one or more benchmarks")
    if set(summary.get("empirical_conflict_tail_by_benchmark", {})) != EXPECTED_BENCHMARKS:
        failures.append("empirical conflict tails are missing one or more benchmarks")
    if summary.get("bootstrap_resamples") != EXPECTED_BOOTSTRAP_RESAMPLES:
        failures.append("bootstrap resample count does not match the frozen rule")
    if summary.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED:
        failures.append("bootstrap seed does not match the frozen rule")

    evidence = summary.get("evidence_gate", {})
    for gate in (
        "positive_continuous_interaction",
        "p10_u_favored",
        "p90_gc_favored",
    ):
        if evidence.get(gate) is not True:
            failures.append(f"evidence gate failed: {gate}")
    if evidence.get("strict_continuous_crossover") is not True:
        failures.append("strict_continuous_crossover is not true")
    for gate in (
        "observed_low_u_favored",
        "observed_high_gc_favored",
        "observed_positive_interaction",
    ):
        if evidence.get(gate) is not True:
            failures.append(f"evidence gate failed: {gate}")
    if evidence.get("strict_observed_tail_crossover") is not True:
        failures.append("strict_observed_tail_crossover is not true")
    if evidence.get("strict_joint_crossover") is not True:
        failures.append("strict_joint_crossover is not true")

    intervals = summary.get("cluster_bootstrap_95_ci", {})
    slope = intervals.get("delta_slope")
    low = intervals.get("delta_p10")
    high = intervals.get("delta_p90")
    for name, interval in (("delta_slope", slope), ("delta_p10", low), ("delta_p90", high)):
        if not _valid_interval(interval):
            failures.append(f"invalid 95% interval: {name}")
    if _valid_interval(slope) and slope[0] <= 0.0:
        failures.append("delta_slope interval is not strictly positive")
    if _valid_interval(low) and low[1] >= 0.0:
        failures.append("delta_p10 interval is not strictly negative")
    if _valid_interval(high) and high[0] <= 0.0:
        failures.append("delta_p90 interval is not strictly positive")

    empirical_intervals = summary.get(
        "empirical_conflict_tail_cluster_bootstrap_95_ci", {}
    )
    empirical_low = empirical_intervals.get("low_delta")
    empirical_high = empirical_intervals.get("high_delta")
    empirical_interaction = empirical_intervals.get("interaction")
    for name, interval in (
        ("empirical low_delta", empirical_low),
        ("empirical high_delta", empirical_high),
        ("empirical interaction", empirical_interaction),
    ):
        if not _valid_interval(interval):
            failures.append(f"invalid 95% interval: {name}")
    if _valid_interval(empirical_low) and empirical_low[1] >= 0.0:
        failures.append("empirical low_delta interval is not strictly negative")
    if _valid_interval(empirical_high) and empirical_high[0] <= 0.0:
        failures.append("empirical high_delta interval is not strictly positive")
    if _valid_interval(empirical_interaction) and empirical_interaction[0] <= 0.0:
        failures.append("empirical interaction interval is not strictly positive")
    valid_empirical_draws = summary.get(
        "empirical_conflict_tail_valid_all_benchmark_bootstrap_draws"
    )
    minimum_empirical_draws = int(
        EXPECTED_BOOTSTRAP_RESAMPLES * MIN_EMPIRICAL_TAIL_VALID_DRAW_FRACTION
    )
    if (
        not isinstance(valid_empirical_draws, int)
        or valid_empirical_draws < minimum_empirical_draws
    ):
        failures.append(
            "fewer than 90% of empirical-tail bootstrap draws retain all benchmarks"
        )

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
    parser.add_argument("--expected-visual-metric", required=True)
    parser.add_argument(
        "--promotion-dir", type=Path, default=Path("publication_ready")
    )
    args = parser.parse_args()

    failures, evidence = evaluate_gates(args)
    source_dir = args.output_root / args.analysis_dir
    promotion_dir = args.output_root / args.promotion_dir
    decision_path = args.output_root / "continuous_promotion_decision.json"
    payload = {
        "schema_version": 1,
        "status": "rejected" if failures else "promoted",
        "output_root": str(args.output_root.resolve()),
        "analysis_dir": str(source_dir.resolve()),
        "expected_probe_protocol": args.expected_probe_protocol,
        "expected_visual_metric": args.expected_visual_metric,
        "expected_reference_population": EXPECTED_REFERENCE_POPULATION,
        "failures": failures,
    }
    source_figures = [
        source_dir / "continuous_visual_interaction_triptych.pdf",
        source_dir / "continuous_visual_interaction_triptych.png",
    ]
    if not failures:
        missing = [
            path for path in source_figures if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            failures.extend(f"missing source figure: {path}" for path in missing)
            payload["status"] = "rejected"
            payload["failures"] = failures
    if not failures:
        promotion_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for source in source_figures:
            destination = promotion_dir / (
                "validated_conflict_conditioned_visual_triptych" + source.suffix
            )
            shutil.copy2(source, destination)
            artifacts.append(
                {
                    "path": str(destination.resolve()),
                    "sha256": sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        payload["artifacts"] = artifacts
        payload["point_estimates"] = evidence["summary"].get(
            "continuous_point_estimates"
        )
        payload["confidence_intervals_95"] = evidence["summary"].get(
            "cluster_bootstrap_95_ci"
        )
        payload["evidence_gate"] = evidence["summary"].get("evidence_gate")
        (promotion_dir / "continuous_manifest.json").write_text(
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
