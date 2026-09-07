"""Outcome-independent launch gate for exact-span visual-regime confirmation.

This gate is deliberately weaker than a confirmation test: a small development
run only decides whether a costly image-disjoint confirmation is warranted.
The confirmation itself must pass cluster-bootstrap confidence-interval gates.
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
MIN_VALID_BENCHMARKS = 6
MIN_TOTAL_STATES_PER_TAIL = 18
MIN_CROSSOVER_BENCHMARKS = 2


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate(summary: dict) -> dict:
    failures = []
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
    if rule.get("visual_metric") != EXPECTED_VISUAL_METRIC:
        failures.append("exact two-token visual metric mismatch")
    if rule.get("uses_future_generated_token") is not True:
        failures.append("future-token diagnostic flag is not explicit")
    if rule.get("online_routing_feature") is not False:
        failures.append("offline diagnostic is incorrectly marked online")
    probe_audit = summary.get("probe_audit", {})
    for key in ("same_text_trajectory_ratio", "recomputed_vision_encoder_ratio"):
        if probe_audit.get(key) != 1.0:
            failures.append(f"probe audit failed: {key}")

    point = summary.get("point_estimates", {})
    values = {key: point.get(key) for key in ("low_delta", "high_delta", "interaction")}
    if not all(_finite(value) for value in values.values()):
        failures.append("one or more macro point estimates are missing/non-finite")
    else:
        if not float(values["low_delta"]) < 0.0:
            failures.append("development low tail does not favor U")
        if not float(values["high_delta"]) > 0.0:
            failures.append("development high tail does not favor G/C")
        if not float(values["interaction"]) > 0.0:
            failures.append("development interaction is not positive")

    by_benchmark = summary.get("by_benchmark", {})
    valid = {
        name: row
        for name, row in by_benchmark.items()
        if name in EXPECTED_BENCHMARKS
        and _finite(row.get("low_delta"))
        and _finite(row.get("high_delta"))
        and _finite(row.get("interaction"))
    }
    num_valid = len(valid)
    low_states = sum(int(row.get("low_eligible_states", 0)) for row in valid.values())
    high_states = sum(int(row.get("high_eligible_states", 0)) for row in valid.values())
    positive_interactions = sum(float(row["interaction"]) > 0.0 for row in valid.values())
    crossover_benchmarks = sum(
        float(row["low_delta"]) < 0.0 < float(row["high_delta"])
        for row in valid.values()
    )
    required_positive = max(4, math.ceil(0.625 * max(num_valid, 1)))
    if num_valid < MIN_VALID_BENCHMARKS:
        failures.append(
            f"only {num_valid} benchmarks have both eligible visual tails"
        )
    if low_states < MIN_TOTAL_STATES_PER_TAIL:
        failures.append(f"low tail has only {low_states} eligible states")
    if high_states < MIN_TOTAL_STATES_PER_TAIL:
        failures.append(f"high tail has only {high_states} eligible states")
    if positive_interactions < required_positive:
        failures.append(
            f"only {positive_interactions}/{num_valid} benchmark interactions "
            f"are positive; require {required_positive}"
        )
    if crossover_benchmarks < MIN_CROSSOVER_BENCHMARKS:
        failures.append(
            f"only {crossover_benchmarks} benchmarks show a point crossover"
        )

    return {
        "schema_version": 1,
        "decision": "go" if not failures else "no_go",
        "role": "development_launch_gate_not_confirmatory_evidence",
        "expected_benchmarks": sorted(EXPECTED_BENCHMARKS),
        "expected_protocol": EXPECTED_PROTOCOL,
        "expected_visual_metric": EXPECTED_VISUAL_METRIC,
        "macro_point_estimates": values,
        "num_valid_benchmarks": num_valid,
        "total_low_eligible_states": low_states,
        "total_high_eligible_states": high_states,
        "positive_interaction_benchmarks": positive_interactions,
        "required_positive_interaction_benchmarks": required_positive,
        "crossover_benchmarks": crossover_benchmarks,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = load_json(args.summary)
        payload = evaluate(summary)
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

