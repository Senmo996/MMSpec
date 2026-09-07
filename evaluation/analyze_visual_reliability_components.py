"""Prespecified component audit for visual–reliability arbitration.

This analysis compares visual-only, reliability-only, and joint endpoint
strata without altering the v11 confirmation decision.  All labels are formed
from variables available before target verification.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    average_tie_percentiles,
    prepare,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)
from evaluation.analyze_visual_reliability_arbitration import (  # noqa: E402
    EXPECTED_BENCHMARKS,
    MIN_DEVELOPMENT_STATES_PER_STRATUM,
    MIN_VALIDATION_STATES_PER_STRATUM,
    TAIL_FRACTION,
    assign_arbitration_strata,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    COUNTERFACTUAL_BANK_PROTOCOL,
)


ROUTE_FIELDS = {
    "visual_only": "arbitration_visual_rank",
    "reliability_only": "arbitration_gc_confidence_advantage_rank",
    "joint": "arbitration_score",
}
STRATIFIED_METRICS = (
    "low_root_delta",
    "high_root_delta",
    "root_interaction",
    "low_accept_delta",
    "high_accept_delta",
    "accept_interaction",
)
REGRESSION_TERMS = ("intercept", "visual", "reliability", "interaction")


def assign_component_strata(
    rows: Sequence[dict], *, route_name: str, tail_fraction: float = TAIL_FRACTION
) -> dict:
    """Assign bottom/top score tails within each benchmark."""

    if route_name not in ROUTE_FIELDS:
        raise ValueError(f"unsupported component route: {route_name}")
    score_field = ROUTE_FIELDS[route_name]
    label_field = f"component_{route_name}_stratum"
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row[label_field] = "ineligible"
        if row.get("arbitration_score") is not None:
            grouped[row["benchmark"]].append(row)

    by_benchmark = {}
    for benchmark, benchmark_rows in sorted(grouped.items()):
        score_percentiles = average_tie_percentiles(
            np.asarray([float(row[score_field]) for row in benchmark_rows])
        )
        counts: dict[str, int] = defaultdict(int)
        for row, percentile in zip(benchmark_rows, score_percentiles.tolist()):
            if float(percentile) < float(tail_fraction):
                stratum = "low"
            elif float(percentile) >= 1.0 - float(tail_fraction):
                stratum = "high"
            else:
                stratum = "ambiguous"
            row[label_field] = stratum
            counts[stratum] += 1
        counts["valid"] = len(benchmark_rows)
        by_benchmark[benchmark] = dict(counts)
    return {
        "route_name": route_name,
        "score_field": score_field,
        "label_field": label_field,
        "tail_fraction": float(tail_fraction),
        "uses_source_hit_outcomes": False,
        "uses_accepted_length_outcomes": False,
        "by_benchmark": by_benchmark,
    }


def summarize_component_benchmark(
    rows: Sequence[dict], *, route_name: str
) -> dict | None:
    label_field = f"component_{route_name}_stratum"
    low = [row for row in rows if row.get(label_field) == "low"]
    high = [row for row in rows if row.get(label_field) == "high"]
    if not low or not high:
        return None

    def mean(group: Sequence[dict], field: str) -> float:
        return float(np.mean([float(row[field]) for row in group]))

    output = {
        "low_states": len(low),
        "high_states": len(high),
        "low_root_delta": mean(low, "frozen_delta"),
        "high_root_delta": mean(high, "frozen_delta"),
        "low_accept_delta": float(
            np.mean(
                [
                    float(row["gc_matched_accept"])
                    - float(row["u_matched_accept"])
                    for row in low
                ]
            )
        ),
        "high_accept_delta": float(
            np.mean(
                [
                    float(row["gc_matched_accept"])
                    - float(row["u_matched_accept"])
                    for row in high
                ]
            )
        ),
    }
    output["root_interaction"] = output["high_root_delta"] - output["low_root_delta"]
    output["accept_interaction"] = (
        output["high_accept_delta"] - output["low_accept_delta"]
    )
    return output


def summarize_component_route(
    rows: Sequence[dict], *, route_name: str, minimum_states_per_stratum: int
) -> tuple[dict, dict, list[str]]:
    by_benchmark = {}
    insufficient = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_component_benchmark(
            [row for row in rows if row["benchmark"] == benchmark],
            route_name=route_name,
        )
        if (
            summary is None
            or summary["low_states"] < minimum_states_per_stratum
            or summary["high_states"] < minimum_states_per_stratum
        ):
            insufficient.append(benchmark)
        else:
            by_benchmark[benchmark] = summary
    if not by_benchmark:
        return {}, {}, insufficient
    point = {
        metric: float(np.mean([summary[metric] for summary in by_benchmark.values()]))
        for metric in STRATIFIED_METRICS
    }
    return point, by_benchmark, insufficient


def weighted_factorial_regression(rows: Sequence[dict], *, outcome: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("arbitration_score") is not None:
            grouped[row["benchmark"]].append(row)
    design = []
    response = []
    weights = []
    for benchmark_rows in grouped.values():
        for row in benchmark_rows:
            visual = float(row["arbitration_visual_rank"]) - 0.5
            reliability = (
                float(row["arbitration_gc_confidence_advantage_rank"]) - 0.5
            )
            design.append([1.0, visual, reliability, visual * reliability])
            response.append(float(row[outcome]))
            weights.append(1.0 / len(benchmark_rows))
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    w = np.sqrt(np.asarray(weights, dtype=np.float64))
    coefficients = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]
    return {
        term: float(value)
        for term, value in zip(REGRESSION_TERMS, coefficients.tolist())
    }


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int
) -> tuple[dict, dict, int]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    valid_rows = [row for row in rows if row.get("arbitration_score") is not None]
    for row in valid_rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    route_draws = {
        route: {metric: [] for metric in STRATIFIED_METRICS}
        for route in ROUTE_FIELDS
    }
    comparison_names = (
        "joint_minus_visual_root_interaction",
        "joint_minus_reliability_root_interaction",
        "joint_minus_visual_accept_interaction",
        "joint_minus_reliability_accept_interaction",
    )
    comparison_draws = {name: [] for name in comparison_names}
    regression_draws = {
        outcome: {term: [] for term in REGRESSION_TERMS}
        for outcome in ("frozen_delta", "matched_gc_minus_u_accept")
    }
    rng = np.random.default_rng(int(seed))
    valid_draws = 0
    for _ in range(int(resamples)):
        sampled_by_benchmark = {}
        sampled_all = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            sampled_indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in sampled_indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            sampled_by_benchmark[benchmark] = sampled
            sampled_all.extend(sampled)

        route_summaries = {}
        failed = False
        for route_name in ROUTE_FIELDS:
            benchmark_summaries = []
            for benchmark in sorted(sampled_by_benchmark):
                summary = summarize_component_benchmark(
                    sampled_by_benchmark[benchmark], route_name=route_name
                )
                if summary is None:
                    failed = True
                    break
                benchmark_summaries.append(summary)
            if failed:
                break
            route_summaries[route_name] = {
                metric: float(
                    np.mean([summary[metric] for summary in benchmark_summaries])
                )
                for metric in STRATIFIED_METRICS
            }
        if failed:
            continue

        valid_draws += 1
        for route_name, summary in route_summaries.items():
            for metric, value in summary.items():
                route_draws[route_name][metric].append(value)
        comparison_draws["joint_minus_visual_root_interaction"].append(
            route_summaries["joint"]["root_interaction"]
            - route_summaries["visual_only"]["root_interaction"]
        )
        comparison_draws["joint_minus_reliability_root_interaction"].append(
            route_summaries["joint"]["root_interaction"]
            - route_summaries["reliability_only"]["root_interaction"]
        )
        comparison_draws["joint_minus_visual_accept_interaction"].append(
            route_summaries["joint"]["accept_interaction"]
            - route_summaries["visual_only"]["accept_interaction"]
        )
        comparison_draws["joint_minus_reliability_accept_interaction"].append(
            route_summaries["joint"]["accept_interaction"]
            - route_summaries["reliability_only"]["accept_interaction"]
        )
        for outcome in regression_draws:
            coefficients = weighted_factorial_regression(sampled_all, outcome=outcome)
            for term, value in coefficients.items():
                regression_draws[outcome][term].append(value)

    def intervals(draws):
        return {
            name: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for name, values in draws.items()
            if values
        }

    return (
        {
            "routes": {
                route: intervals(metrics) for route, metrics in route_draws.items()
            },
            "comparisons": intervals(comparison_draws),
        },
        {
            outcome: intervals(terms)
            for outcome, terms in regression_draws.items()
        },
        valid_draws,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--analysis-role", choices=("development", "new_validation"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=91211)
    args = parser.parse_args()

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != EXPECTED_BENCHMARKS:
        parser.error(
            "expected exactly the eight non-MME benchmarks, got " + repr(benchmarks)
        )
    protocols = sorted({str(row["visual_probe_protocol"]) for row in rows})
    if protocols != [COUNTERFACTUAL_BANK_PROTOCOL]:
        parser.error(f"unexpected visual probe protocols: {protocols}")

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    arbitration_audit = assign_arbitration_strata(rows)
    minimum = (
        MIN_DEVELOPMENT_STATES_PER_STRATUM
        if args.analysis_role == "development"
        else MIN_VALIDATION_STATES_PER_STRATUM
    )
    assignment_audits = {}
    point_estimates = {}
    by_benchmark = {}
    insufficient = {}
    for route_name in ROUTE_FIELDS:
        assignment_audits[route_name] = assign_component_strata(
            rows, route_name=route_name
        )
        point, per_benchmark, missing = summarize_component_route(
            rows,
            route_name=route_name,
            minimum_states_per_stratum=minimum,
        )
        point_estimates[route_name] = point
        by_benchmark[route_name] = per_benchmark
        insufficient[route_name] = missing

    regression = {
        outcome: weighted_factorial_regression(rows, outcome=outcome)
        for outcome in ("frozen_delta", "matched_gc_minus_u_accept")
    }
    bootstrap, regression_intervals, valid_draws = clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    comparisons = {}
    if all(point_estimates.values()):
        comparisons = {
            "joint_minus_visual_root_interaction": (
                point_estimates["joint"]["root_interaction"]
                - point_estimates["visual_only"]["root_interaction"]
            ),
            "joint_minus_reliability_root_interaction": (
                point_estimates["joint"]["root_interaction"]
                - point_estimates["reliability_only"]["root_interaction"]
            ),
            "joint_minus_visual_accept_interaction": (
                point_estimates["joint"]["accept_interaction"]
                - point_estimates["visual_only"]["accept_interaction"]
            ),
            "joint_minus_reliability_accept_interaction": (
                point_estimates["joint"]["accept_interaction"]
                - point_estimates["reliability_only"]["accept_interaction"]
            ),
        }

    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "warning": (
            "This component audit cannot replace or rescue the frozen v11 "
            "confirmation decision."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "minimum_states_per_stratum": minimum,
        "route_score_fields": ROUTE_FIELDS,
        "tail_fraction": TAIL_FRACTION,
        "eligibility_audit": eligibility_audit,
        "arbitration_audit": arbitration_audit,
        "assignment_audits": assignment_audits,
        "point_estimates": point_estimates,
        "comparisons": comparisons,
        "by_benchmark": by_benchmark,
        "insufficient_support_benchmarks": insufficient,
        "factorial_regression": regression,
        "cluster_bootstrap_95_ci": bootstrap,
        "factorial_regression_cluster_bootstrap_95_ci": regression_intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_route_all_benchmark_draws": valid_draws,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "component_audit.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "point_estimates": point_estimates,
                "comparisons": comparisons,
                "factorial_regression": regression,
                "insufficient_support_benchmarks": insufficient,
                "bootstrap_valid_draws": valid_draws,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
