"""Descriptive tie-aware pairwise win rates for candidate-only validation.

For each state, a source receives one point when its matched-budget accepted
prefix is longer, zero when it is shorter, and one half for a tie.  The metric
therefore answers how often each source wins, while the frozen accepted-token
delta continues to measure by how much.  This panel is descriptive and does
not replace the frozen confirmatory gate.
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

from evaluation.analyze_candidate_visual_alignment_only import (  # noqa: E402
    DEFAULT_EXPECTED_BENCHMARKS,
    assign_candidate_only_strata,
)
from evaluation.analyze_frozen_selective_conflict_regime import prepare  # noqa: E402
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)


METRICS = (
    "low_u_pairwise_win",
    "low_gc_pairwise_win",
    "high_u_pairwise_win",
    "high_gc_pairwise_win",
)


def _u_tie_aware_win(row: dict) -> float:
    u_accept = float(row["u_matched_accept"])
    gc_accept = float(row["gc_matched_accept"])
    if u_accept > gc_accept:
        return 1.0
    if u_accept < gc_accept:
        return 0.0
    return 0.5


def summarize_pairwise_win(rows: Sequence[dict]) -> tuple[dict, dict] | None:
    """Return equal-benchmark pairwise win probabilities in frozen strata."""

    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        low = [
            row
            for row in benchmark_rows
            if row.get("arbitration_stratum") == "low"
        ]
        high = [
            row
            for row in benchmark_rows
            if row.get("arbitration_stratum") == "high"
        ]
        if not low or not high:
            return None
        low_u = [_u_tie_aware_win(row) for row in low]
        high_u = [_u_tie_aware_win(row) for row in high]
        by_benchmark[benchmark] = {
            "low_states": len(low),
            "high_states": len(high),
            "low_tie_rate": float(
                np.mean(
                    [
                        float(row["u_matched_accept"])
                        == float(row["gc_matched_accept"])
                        for row in low
                    ]
                )
            ),
            "high_tie_rate": float(
                np.mean(
                    [
                        float(row["u_matched_accept"])
                        == float(row["gc_matched_accept"])
                        for row in high
                    ]
                )
            ),
            "low_u_pairwise_win": float(np.mean(low_u)),
            "low_gc_pairwise_win": float(
                np.mean([1.0 - value for value in low_u])
            ),
            "high_u_pairwise_win": float(np.mean(high_u)),
            "high_gc_pairwise_win": float(
                np.mean([1.0 - value for value in high_u])
            ),
        }
    point = {
        metric: float(
            np.mean([summary[metric] for summary in by_benchmark.values()])
        )
        for metric in METRICS
    }
    return point, by_benchmark


def clustered_bootstrap_pairwise_win(
    rows: Sequence[dict], *, resamples: int, seed: int
) -> tuple[dict, int]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    draws = {metric: [] for metric in METRICS}
    valid_draws = 0
    for _ in range(int(resamples)):
        sampled_rows = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled_rows.extend(
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            )
        summary = summarize_pairwise_win(sampled_rows)
        if summary is None:
            continue
        point, _ = summary
        valid_draws += 1
        for metric in METRICS:
            draws[metric].append(point[metric])
    intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
    }
    return intervals, valid_draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=223607)
    parser.add_argument(
        "--expected-benchmarks", default=DEFAULT_EXPECTED_BENCHMARKS
    )
    args = parser.parse_args()

    primary = json.loads(args.primary_summary.read_text(encoding="utf-8"))
    if primary.get("decision") not in {"validated", "sensitivity_pass"}:
        parser.error("primary candidate-only result did not pass")
    tail_fraction = float(primary["frozen_rule"]["tail_fraction"])
    expected = {
        item.strip()
        for item in str(args.expected_benchmarks).split(",")
        if item.strip()
    }
    paths = discover_result_paths(args.results_roots, args.policy)
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != expected:
        parser.error(
            f"benchmark mismatch: expected {sorted(expected)}, got {benchmarks}"
        )
    prepare(rows, visual_metric="span2_mean_target_drop_fraction")
    assignment_audit = assign_candidate_only_strata(
        rows, tail_fraction=tail_fraction
    )
    result = summarize_pairwise_win(rows)
    if result is None:
        parser.error("one or more benchmarks lack a frozen low/high stratum")
    point, by_benchmark = result
    intervals, valid_draws = clustered_bootstrap_pairwise_win(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    payload = {
        "schema_version": 1,
        "analysis_role": "descriptive_pairwise_win_panel",
        "not_a_validation_gate": True,
        "tail_fraction": tail_fraction,
        "primary_summary": str(args.primary_summary.resolve()),
        "benchmarks": benchmarks,
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "outcome": "matched-budget accepted-prefix length",
        "tie_handling": "each source receives one half for an equal length",
        "assignment_audit": assignment_audit,
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "by_benchmark": by_benchmark,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
