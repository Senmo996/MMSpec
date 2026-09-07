"""Descriptive four-bar visual-support panel for candidate-only validation.

The confirmatory candidate-only labels and outcomes remain unchanged.  This
secondary analysis exposes the two source-specific terms whose difference
defines candidate alignment:

    E_s = min(log P_s(true) - log P_s(mean),
              log P_s(true) - log P_s(wrong)).

It is used only to render panel (a), not as an additional validation gate.
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

from evaluation.analyze_candidate_visual_alignment import (  # noqa: E402
    GC_SUPPORT_FIELD,
    U_SUPPORT_FIELD,
)
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
    "low_u_support",
    "low_gc_support",
    "high_u_support",
    "high_gc_support",
    "low_u_relative_support",
    "low_gc_relative_support",
    "high_u_relative_support",
    "high_gc_relative_support",
)


def _u_relative_support(row: dict) -> float:
    """Return the parameter-free softmax share assigned to U."""

    delta = float(row[GC_SUPPORT_FIELD]) - float(row[U_SUPPORT_FIELD])
    delta = float(np.clip(delta, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(delta)))


def summarize_support(rows: Sequence[dict]) -> tuple[dict, dict] | None:
    """Return equal-benchmark source support for both frozen strata."""

    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        low = [row for row in benchmark_rows if row.get("arbitration_stratum") == "low"]
        high = [
            row for row in benchmark_rows if row.get("arbitration_stratum") == "high"
        ]
        if not low or not high:
            return None
        low_u_relative = [_u_relative_support(row) for row in low]
        high_u_relative = [_u_relative_support(row) for row in high]
        by_benchmark[benchmark] = {
            "low_states": len(low),
            "high_states": len(high),
            "low_u_support": float(np.mean([row[U_SUPPORT_FIELD] for row in low])),
            "low_gc_support": float(
                np.mean([row[GC_SUPPORT_FIELD] for row in low])
            ),
            "high_u_support": float(
                np.mean([row[U_SUPPORT_FIELD] for row in high])
            ),
            "high_gc_support": float(
                np.mean([row[GC_SUPPORT_FIELD] for row in high])
            ),
            "low_u_relative_support": float(np.mean(low_u_relative)),
            "low_gc_relative_support": float(
                np.mean([1.0 - value for value in low_u_relative])
            ),
            "high_u_relative_support": float(np.mean(high_u_relative)),
            "high_gc_relative_support": float(
                np.mean([1.0 - value for value in high_u_relative])
            ),
        }
    point = {
        metric: float(
            np.mean([summary[metric] for summary in by_benchmark.values()])
        )
        for metric in METRICS
    }
    return point, by_benchmark


def clustered_bootstrap_support(
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
        summary = summarize_support(sampled_rows)
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
    result = summarize_support(rows)
    if result is None:
        parser.error("one or more benchmarks lack a frozen low/high stratum")
    point, by_benchmark = result
    intervals, valid_draws = clustered_bootstrap_support(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    payload = {
        "schema_version": 1,
        "analysis_role": "descriptive_source_support_panel",
        "not_a_validation_gate": True,
        "tail_fraction": tail_fraction,
        "primary_summary": str(args.primary_summary.resolve()),
        "benchmarks": benchmarks,
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "support_definition": (
            "minimum true-minus-mean and true-minus-matched-wrong "
            "candidate-set log-probability-mass gain"
        ),
        "relative_support_definition": (
            "parameter-free two-source softmax of the per-state conservative "
            "supports; U and G/C shares sum to one before aggregation"
        ),
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
