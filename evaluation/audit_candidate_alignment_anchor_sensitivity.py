"""Post-hoc sensitivity audit for the candidate-only validation.

The frozen primary analysis requires the true-image full-model top-1 token to
match the teacher token before a state can enter an alignment stratum.  This
audit removes only that anchor-fidelity filter while preserving every other
eligibility rule, the candidate-only score, the 30% tails, and the clustered
bootstrap.  It is a robustness check, not a replacement confirmatory test.
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
    GC_CONFIDENCE_FIELD,
    GC_SUPPORT_FIELD,
    U_CONFIDENCE_FIELD,
    U_SUPPORT_FIELD,
    VISUAL_ALIGNMENT_FIELD,
)
from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    average_tie_percentiles,
    prepare,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)
from evaluation.analyze_visual_reliability_arbitration import (  # noqa: E402
    MIN_VALIDATION_STATES_PER_STRATUM,
    TAIL_FRACTION,
    clustered_bootstrap,
    macro_summary,
)


DEFAULT_EXPECTED_BENCHMARKS = (
    "MMT-Bench,SEEDBench,ScienceQA,OCRBench,ChartQA,MathVista,TextVQA"
)


def assign_without_anchor_filter(
    rows: Sequence[dict], *, tail_fraction: float = TAIL_FRACTION
) -> dict:
    """Assign candidate-only tails without consulting the teacher-token anchor."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    required = (
        VISUAL_ALIGNMENT_FIELD,
        U_SUPPORT_FIELD,
        GC_SUPPORT_FIELD,
        U_CONFIDENCE_FIELD,
        GC_CONFIDENCE_FIELD,
    )
    for row in rows:
        row["arbitration_stratum"] = "ineligible"
        row["arbitration_visual_rank"] = None
        row["arbitration_gc_confidence_advantage_rank"] = None
        row["arbitration_score"] = None
        row["arbitration_score_percentile"] = None
        if not row.get("frozen_eligible", False):
            excluded["frozen_ineligible"] += 1
            continue
        if not row.get("visual_probe_candidate_alignment_available", False):
            excluded["missing_candidate_alignment"] += 1
            continue
        if row.get("visual_probe_candidate_alignment_uses_target_outcome", False):
            raise ValueError("candidate alignment unexpectedly uses target outcome")
        if any(row.get(field) is None for field in required):
            excluded["missing_signal"] += 1
            continue
        if int(row.get("visual_probe_candidate_alignment_budget", 0)) != int(
            row.get("frozen_root_budget", -1)
        ):
            excluded["candidate_budget_mismatch"] += 1
            continue
        if int(row.get("root_transition_context_order", 0)) not in (2, 3):
            excluded["invalid_gc_context_order"] += 1
            continue
        u_probability = float(row[U_CONFIDENCE_FIELD])
        gc_probability = float(row[GC_CONFIDENCE_FIELD])
        if not (0.0 <= u_probability <= 1.0 and 0.0 <= gc_probability <= 1.0):
            excluded["invalid_probability"] += 1
            continue
        grouped[row["benchmark"]].append(row)

    by_benchmark = {}
    for benchmark, benchmark_rows in sorted(grouped.items()):
        visual_values = np.asarray(
            [float(row[VISUAL_ALIGNMENT_FIELD]) for row in benchmark_rows]
        )
        reliability_values = np.asarray(
            [
                float(row[GC_CONFIDENCE_FIELD])
                - float(row[U_CONFIDENCE_FIELD])
                for row in benchmark_rows
            ]
        )
        visual_ranks = average_tie_percentiles(visual_values)
        reliability_ranks = average_tie_percentiles(reliability_values)
        counts: dict[str, int] = defaultdict(int)
        for row, visual_rank, reliability_rank in zip(
            benchmark_rows,
            visual_ranks.tolist(),
            reliability_ranks.tolist(),
        ):
            percentile = float(visual_rank)
            row["arbitration_visual_rank"] = percentile
            row["arbitration_gc_confidence_advantage_rank"] = float(
                reliability_rank
            )
            row["arbitration_score"] = percentile
            row["arbitration_score_percentile"] = percentile
            if percentile < float(tail_fraction):
                stratum = "low"
            elif percentile >= 1.0 - float(tail_fraction):
                stratum = "high"
            else:
                stratum = "ambiguous"
            row["arbitration_stratum"] = stratum
            counts[stratum] += 1
        counts["reference_states"] = len(benchmark_rows)
        counts["failed_full_anchor_states_included"] = sum(
            not row.get("visual_probe_full_top1_matches_target", False)
            for row in benchmark_rows
        )
        by_benchmark[benchmark] = dict(counts)

    return {
        "uses_source_reliability_for_assignment": False,
        "uses_source_hit_outcomes_for_assignment": False,
        "uses_accepted_length_outcomes_for_assignment": False,
        "uses_target_token_id_for_assignment_or_eligibility": False,
        "tail_fraction": float(tail_fraction),
        "excluded": dict(excluded),
        "by_benchmark": by_benchmark,
    }


def _strict_gate(point: dict, intervals: dict, valid_draws: int, required: int) -> bool:
    gates = (
        point["low_root_delta"] < 0.0
        and intervals["low_root_delta"][1] < 0.0,
        point["high_root_delta"] > 0.0
        and intervals["high_root_delta"][0] > 0.0,
        point["root_interaction"] > 0.0
        and intervals["root_interaction"][0] > 0.0,
        point["low_accept_delta"] < 0.0
        and intervals["low_accept_delta"][1] < 0.0,
        point["high_accept_delta"] > 0.0
        and intervals["high_accept_delta"][0] > 0.0,
        point["accept_interaction"] > 0.0
        and intervals["accept_interaction"][0] > 0.0,
    )
    return bool(valid_draws >= required and all(gates))


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

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    assignment_audit = assign_without_anchor_filter(rows)
    point, by_benchmark, insufficient = macro_summary(
        rows, minimum_states_per_stratum=MIN_VALIDATION_STATES_PER_STRATUM
    )
    intervals, valid_draws = clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    required_draws = max(100, int(0.90 * args.bootstrap_resamples))
    primary = json.loads(args.primary_summary.read_text(encoding="utf-8"))
    comparison_metrics = (
        "low_root_delta",
        "high_root_delta",
        "root_interaction",
        "low_accept_delta",
        "high_accept_delta",
        "accept_interaction",
    )
    payload = {
        "schema_version": 1,
        "analysis_role": "post_hoc_anchor_filter_sensitivity",
        "warning": "Robustness audit only; the frozen primary remains authoritative.",
        "benchmarks": benchmarks,
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "eligibility_audit": eligibility_audit,
        "assignment_audit": assignment_audit,
        "failed_full_anchor_states_included": sum(
            not row.get("visual_probe_full_top1_matches_target", False)
            and row.get("arbitration_stratum") in {"low", "high", "ambiguous"}
            for row in rows
        ),
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "by_benchmark": by_benchmark,
        "insufficient_support_benchmarks": insufficient,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "bootstrap_required_valid_draws": required_draws,
        "strict_double_crossover_persists": _strict_gate(
            point, intervals, valid_draws, required_draws
        ),
        "difference_from_primary": {
            metric: float(point[metric] - primary["point_estimates"][metric])
            for metric in comparison_metrics
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
