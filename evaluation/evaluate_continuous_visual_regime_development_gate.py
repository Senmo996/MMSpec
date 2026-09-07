"""Gate a costly confirmation run from a continuous development diagnostic.

This script implements the directional development gate frozen in
``docs/selective_reuse_visual_regime_v3_spec.md``.  It deliberately does not
test confidence-interval significance: only the later image-disjoint
confirmation may support that claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
EXPECTED_PROTOCOL = "teacher_forced_whole_image_mean_ablation_v1"
EXPECTED_VISUAL_METRIC = "span2_mean_target_drop_fraction"
ALLOWED_VISUAL_METRICS = (
    EXPECTED_VISUAL_METRIC,
    "span2_mean_jsd",
)
EXPECTED_REFERENCE_POPULATION = "eligible_conflicts"
EXPECTED_BOOTSTRAP_RESAMPLES = 5000
EXPECTED_BOOTSTRAP_SEED = 161803


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate(
    summary: dict, *, expected_visual_metric: str = EXPECTED_VISUAL_METRIC
) -> dict:
    if expected_visual_metric not in ALLOWED_VISUAL_METRICS:
        raise ValueError(
            f"unsupported development visual metric: {expected_visual_metric}"
        )
    failures: list[str] = []
    if summary.get("analysis_role") != "development":
        failures.append("analysis_role is not development")
    if summary.get("analysis_split") != "all":
        failures.append("analysis_split is not all")
    if set(summary.get("benchmarks", [])) != EXPECTED_BENCHMARKS:
        failures.append("expected exactly the eight non-MME benchmarks")
    if summary.get("excluded_benchmarks") != ["MME"]:
        failures.append("MME exclusion is absent or malformed")
    if summary.get("visual_probe_protocols") != [EXPECTED_PROTOCOL]:
        failures.append("whole-image content-ablation protocol mismatch")

    rule = summary.get("frozen_rule", {})
    if rule.get("visual_metric") != expected_visual_metric:
        failures.append("exact two-token visual metric mismatch")
    if (
        rule.get("visual_percentile_reference_population")
        != EXPECTED_REFERENCE_POPULATION
    ):
        failures.append("visual percentile is not conflict-conditioned")

    probe = summary.get("probe_audit", {})
    for key in ("same_text_trajectory_ratio", "recomputed_vision_encoder_ratio"):
        if probe.get(key) != 1.0:
            failures.append(f"probe audit failed: {key}")

    percentile = summary.get("visual_percentile_audit", {})
    if percentile.get("reference_population") != EXPECTED_REFERENCE_POPULATION:
        failures.append("visual percentile audit has the wrong population")
    if percentile.get("uses_source_hit_outcomes") is not False:
        failures.append("visual percentile assignment used a source-hit outcome")
    support = percentile.get("states_by_benchmark", {})
    if set(support) != EXPECTED_BENCHMARKS:
        failures.append("visual percentile audit is missing benchmark support")
    elif any(not isinstance(value, int) or value < 3 for value in support.values()):
        failures.append("one or more benchmarks have fewer than three conflicts")
    if support and percentile.get("num_reference_states") != sum(support.values()):
        failures.append("visual percentile state total is inconsistent")

    by_benchmark = summary.get("continuous_by_benchmark", {})
    if set(by_benchmark) != EXPECTED_BENCHMARKS:
        failures.append("continuous fit is missing one or more benchmarks")
    finite_benchmarks = {
        name
        for name, row in by_benchmark.items()
        if name in EXPECTED_BENCHMARKS
        and all(
            _finite(row.get(key))
            for key in ("delta_slope", "delta_p10", "delta_p90")
        )
    }
    if finite_benchmarks != EXPECTED_BENCHMARKS:
        failures.append("one or more benchmark estimates are missing/non-finite")

    point = summary.get("continuous_point_estimates", {})
    values = {
        key: point.get(key) for key in ("delta_slope", "delta_p10", "delta_p90")
    }
    if not all(_finite(value) for value in values.values()):
        failures.append("one or more macro continuous estimates are missing/non-finite")
    else:
        if not float(values["delta_slope"]) > 0.0:
            failures.append("development source-preference slope is not positive")
        if not float(values["delta_p10"]) < 0.0:
            failures.append("development percentile-0.10 endpoint does not favor U")
        if not float(values["delta_p90"]) > 0.0:
            failures.append("development percentile-0.90 endpoint does not favor G/C")

    if summary.get("bootstrap_resamples") != EXPECTED_BOOTSTRAP_RESAMPLES:
        failures.append("development bootstrap resample count mismatch")
    if summary.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED:
        failures.append("development bootstrap seed mismatch")

    positive_slopes = sum(
        float(row["delta_slope"]) > 0.0
        for name, row in by_benchmark.items()
        if name in finite_benchmarks
    )
    point_crossovers = sum(
        float(row["delta_p10"]) < 0.0 < float(row["delta_p90"])
        for name, row in by_benchmark.items()
        if name in finite_benchmarks
    )
    return {
        "schema_version": 1,
        "decision": "go" if not failures else "no_go",
        "role": "development_launch_gate_not_confirmatory_evidence",
        "expected_benchmarks": sorted(EXPECTED_BENCHMARKS),
        "expected_protocol": EXPECTED_PROTOCOL,
        "expected_visual_metric": expected_visual_metric,
        "expected_reference_population": EXPECTED_REFERENCE_POPULATION,
        "macro_point_estimates": values,
        "states_by_benchmark": support,
        "positive_slope_benchmarks": positive_slopes,
        "point_crossover_benchmarks": point_crossovers,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-visual-metric",
        choices=ALLOWED_VISUAL_METRICS,
        default=EXPECTED_VISUAL_METRIC,
    )
    args = parser.parse_args()
    try:
        payload = evaluate(
            load_json(args.summary),
            expected_visual_metric=args.expected_visual_metric,
        )
    except ValueError as error:
        payload = {
            "schema_version": 1,
            "decision": "no_go",
            "role": "development_launch_gate_not_confirmatory_evidence",
            "failures": [str(error)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
