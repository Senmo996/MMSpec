"""Analyze source-relative candidate visual alignment.

The frozen v14 score asks which reuse source's candidate set gains more target-
model probability mass from the true image than from both counterfactual views.
Candidate-set IDs and row probabilities are available before verification;
source hits and accepted lengths are outcomes only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    BLUE,
    INK,
    MUTED,
    ORANGE,
    average_tie_percentiles,
    prepare,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)
from evaluation.analyze_visual_reliability_arbitration import (  # noqa: E402
    EXPECTED_BENCHMARKS,
    METRICS,
    MIN_DEVELOPMENT_STATES_PER_STRATUM,
    MIN_VALIDATION_STATES_PER_STRATUM,
    TAIL_FRACTION,
    clustered_bootstrap as primary_clustered_bootstrap,
    macro_summary,
)
from evaluation.analyze_visual_reliability_components import (  # noqa: E402
    ROUTE_FIELDS,
    assign_component_strata,
    clustered_bootstrap as component_clustered_bootstrap,
    summarize_component_route,
    weighted_factorial_regression,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    COUNTERFACTUAL_BANK_PROTOCOL,
)


VISUAL_ALIGNMENT_FIELD = "visual_probe_gc_minus_u_candidate_visual_support"
U_SUPPORT_FIELD = "visual_probe_u_candidate_consensus_support"
GC_SUPPORT_FIELD = "visual_probe_gc_candidate_consensus_support"
U_CONFIDENCE_FIELD = "u_row_top_probability"
GC_CONFIDENCE_FIELD = "root_transition_top_probability"
VISUAL_WEIGHT = 0.50
DEFAULT_EXPECTED_BENCHMARKS = ",".join(sorted(EXPECTED_BENCHMARKS))


def assign_candidate_alignment_strata(
    rows: Sequence[dict],
    *,
    visual_weight: float = VISUAL_WEIGHT,
    tail_fraction: float = TAIL_FRACTION,
) -> dict:
    """Assign frozen U-aligned and G/C-aligned arbitration tails."""

    if not 0.0 <= float(visual_weight) <= 1.0:
        raise ValueError("visual_weight must lie in [0, 1]")
    if not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must lie strictly between 0 and 0.5")
    grouped: dict[str, list[dict]] = defaultdict(list)
    invalid_reasons: dict[str, int] = defaultdict(int)
    score_fields = (
        "arbitration_visual_rank",
        "arbitration_gc_confidence_advantage_rank",
        "arbitration_score",
        "arbitration_score_percentile",
    )
    for row in rows:
        row["arbitration_stratum"] = "ineligible"
        for field in score_fields:
            row[field] = None
        if not row.get("frozen_eligible", False):
            continue
        if not row.get("visual_probe_full_top1_matches_target", False):
            row["arbitration_stratum"] = "invalid_full_anchor"
            invalid_reasons["invalid_full_anchor"] += 1
            continue
        if not row.get("visual_probe_candidate_alignment_available", False):
            row["arbitration_stratum"] = "missing_candidate_alignment"
            invalid_reasons["missing_candidate_alignment"] += 1
            continue
        if row.get("visual_probe_candidate_alignment_uses_target_outcome", False):
            raise ValueError("candidate alignment unexpectedly uses target outcome")
        required = (
            VISUAL_ALIGNMENT_FIELD,
            U_SUPPORT_FIELD,
            GC_SUPPORT_FIELD,
            U_CONFIDENCE_FIELD,
            GC_CONFIDENCE_FIELD,
        )
        if any(row.get(field) is None for field in required):
            row["arbitration_stratum"] = "missing_signal"
            invalid_reasons["missing_signal"] += 1
            continue
        if int(row.get("visual_probe_candidate_alignment_budget", 0)) != int(
            row.get("frozen_root_budget", -1)
        ):
            row["arbitration_stratum"] = "candidate_budget_mismatch"
            invalid_reasons["candidate_budget_mismatch"] += 1
            continue
        if int(row.get("root_transition_context_order", 0)) not in (2, 3):
            row["arbitration_stratum"] = "invalid_gc_context_order"
            invalid_reasons["invalid_gc_context_order"] += 1
            continue
        u_probability = float(row[U_CONFIDENCE_FIELD])
        gc_probability = float(row[GC_CONFIDENCE_FIELD])
        if not (0.0 <= u_probability <= 1.0 and 0.0 <= gc_probability <= 1.0):
            row["arbitration_stratum"] = "invalid_probability"
            invalid_reasons["invalid_probability"] += 1
            continue
        grouped[row["benchmark"]].append(row)

    by_benchmark = {}
    for benchmark, benchmark_rows in sorted(grouped.items()):
        visual_values = np.asarray(
            [float(row[VISUAL_ALIGNMENT_FIELD]) for row in benchmark_rows]
        )
        visual_ranks = average_tie_percentiles(visual_values)
        reliability_values = np.asarray(
            [
                float(row[GC_CONFIDENCE_FIELD])
                - float(row[U_CONFIDENCE_FIELD])
                for row in benchmark_rows
            ]
        )
        reliability_ranks = average_tie_percentiles(reliability_values)
        scores = []
        for row, visual_rank, reliability_rank in zip(
            benchmark_rows,
            visual_ranks.tolist(),
            reliability_ranks.tolist(),
        ):
            row["arbitration_visual_rank"] = float(visual_rank)
            row["arbitration_gc_confidence_advantage_rank"] = float(
                reliability_rank
            )
            score = (
                float(visual_weight) * float(visual_rank)
                + (1.0 - float(visual_weight)) * float(reliability_rank)
            )
            row["arbitration_score"] = score
            scores.append(score)

        score_percentiles = average_tie_percentiles(np.asarray(scores))
        counts: dict[str, int] = defaultdict(int)
        for row, percentile in zip(benchmark_rows, score_percentiles.tolist()):
            row["arbitration_score_percentile"] = float(percentile)
            if float(percentile) < float(tail_fraction):
                stratum = "low"
            elif float(percentile) >= 1.0 - float(tail_fraction):
                stratum = "high"
            else:
                stratum = "ambiguous"
            row["arbitration_stratum"] = stratum
            counts[stratum] += 1
        counts["reference_states"] = len(benchmark_rows)
        by_benchmark[benchmark] = {
            **dict(counts),
            "candidate_alignment_min": float(visual_values.min()),
            "candidate_alignment_max": float(visual_values.max()),
            "candidate_alignment_mean": float(visual_values.mean()),
        }

    return {
        "uses_source_hit_outcomes": False,
        "uses_accepted_length_outcomes": False,
        "uses_target_token_id": False,
        "visual_weight": float(visual_weight),
        "source_reliability_weight": 1.0 - float(visual_weight),
        "tail_fraction": float(tail_fraction),
        "visual_signal": (
            "within-benchmark rank of consensus true-image candidate-set "
            "support for G/C minus U"
        ),
        "source_reliability_signal": (
            "within-benchmark rank of G/C row top probability minus U row "
            "top probability"
        ),
        "valid_states": sum(
            counts.get("reference_states", 0)
            for counts in by_benchmark.values()
        ),
        "invalid_reasons": dict(invalid_reasons),
        "by_benchmark": by_benchmark,
    }


def _bar_errors(values, names, intervals):
    return np.asarray(
        [
            [value - intervals[name][0] for value, name in zip(values, names)],
            [intervals[name][1] - value for value, name in zip(values, names)],
        ]
    )


def plot(payload: dict, output_stem: Path) -> None:
    point = payload["point_estimates"]
    intervals = payload["cluster_bootstrap_95_ci"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    x = np.arange(2)
    width = 0.32
    labels = ["U-aligned\nregime", "G/C-aligned\nregime"]

    for offset, names, color, label in (
        (
            -width / 2,
            ("low_visual_rank", "high_visual_rank"),
            "#4C9FBE",
            "Candidate visual alignment",
        ),
        (
            width / 2,
            (
                "low_gc_confidence_advantage_rank",
                "high_gc_confidence_advantage_rank",
            ),
            "#8064A2",
            "G/C reliability advantage",
        ),
    ):
        values = [point[name] for name in names]
        axes[0].bar(
            x + offset,
            values,
            width,
            yerr=_bar_errors(values, names, intervals),
            capsize=2.4,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("Within-benchmark rank")
    axes[0].set_title("(a) Source evidence", loc="left", weight="bold")
    axes[0].legend(
        loc="upper left",
        frameon=False,
        fontsize=4.6,
        handlelength=1.2,
        labelspacing=0.25,
    )

    for offset, prefix, color, label in (
        (-width / 2, "u", ORANGE, "Persistent U"),
        (width / 2, "gc", BLUE, "Request-local G/C"),
    ):
        names = (f"low_{prefix}_recall", f"high_{prefix}_recall")
        values = [point[name] for name in names]
        axes[1].bar(
            x + offset,
            values,
            width,
            yerr=_bar_errors(values, names, intervals),
            capsize=2.4,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_ylabel("Equal-budget root recall")
    axes[1].set_title("(b) Root recall", loc="left", weight="bold")

    names = ("low_accept_delta", "high_accept_delta")
    values = [point[name] for name in names]
    axes[2].bar(
        x,
        values,
        0.48,
        yerr=_bar_errors(values, names, intervals),
        capsize=2.8,
        color=[ORANGE, BLUE],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[2].axhline(0.0, color="#5D6670", linestyle=(0, (3, 2)), lw=0.9)
    bound = max(
        0.25,
        max(
            abs(intervals[name][edge])
            for name in names
            for edge in (0, 1)
        )
        * 1.15,
    )
    axes[2].set_ylim(-bound, bound)
    axes[2].set_ylabel("G/C - U accepted tokens")
    axes[2].set_title("(c) Accepted-token gain", loc="left", weight="bold")
    for index, value in enumerate(values):
        offset = 0.045 * bound
        axes[2].text(
            index,
            value + (offset if value >= 0.0 else -offset),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=6.2,
            weight="bold",
            color=INK,
        )

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.title.set_fontsize(7.5)
        axis.yaxis.label.set_size(6.3)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.955),
        ncol=2,
        frameon=False,
        fontsize=6.1,
        handlelength=1.5,
        columnspacing=1.0,
    )
    fig.suptitle(
        "Score visual evidence over reuse candidates, not only the target token",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=8.8,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.905,
        f"{payload['analysis_role'].upper().replace('_', ' ')} · "
        f"{len(payload['benchmarks'])} benchmarks · MME excluded · "
        "50% candidate alignment + 50% row reliability",
        ha="left",
        va="center",
        fontsize=5.8,
        weight="bold",
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "Outcome-blind strata; equal benchmark weight; matched root/tree "
        "budget; image-cluster bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=5.1,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.77, bottom=0.25, wspace=0.48)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--analysis-role",
        choices=("development", "new_validation"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=141421)
    parser.add_argument(
        "--expected-benchmarks",
        default=DEFAULT_EXPECTED_BENCHMARKS,
        help="Comma-separated exact benchmark set; MME is forbidden.",
    )
    args = parser.parse_args()

    expected_benchmarks = {
        name.strip()
        for name in str(args.expected_benchmarks).split(",")
        if name.strip()
    }
    if not expected_benchmarks:
        parser.error("--expected-benchmarks must not be empty")
    if "MME" in expected_benchmarks or "MME_Benchmark" in expected_benchmarks:
        parser.error("MME is excluded from this diagnostic")

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != expected_benchmarks:
        parser.error(
            "benchmark set mismatch: expected "
            + repr(sorted(expected_benchmarks))
            + ", got "
            + repr(benchmarks)
        )
    protocols = sorted({str(row["visual_probe_protocol"]) for row in rows})
    if protocols != [COUNTERFACTUAL_BANK_PROTOCOL]:
        parser.error(f"unexpected visual probe protocols: {protocols}")
    if not all(row.get("visual_probe_same_text_trajectory") for row in rows):
        parser.error("counterfactual probes did not preserve every trajectory")
    if not all(row.get("visual_probe_recomputed_vision_encoder") for row in rows):
        parser.error("counterfactual probes did not recompute every vision input")

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    arbitration_audit = assign_candidate_alignment_strata(rows)
    minimum = (
        MIN_DEVELOPMENT_STATES_PER_STRATUM
        if args.analysis_role == "development"
        else MIN_VALIDATION_STATES_PER_STRATUM
    )
    try:
        point, by_benchmark, insufficient = macro_summary(
            rows, minimum_states_per_stratum=minimum
        )
    except ValueError:
        point, by_benchmark, insufficient = {}, {}, benchmarks
    support_complete = (
        not insufficient and set(by_benchmark) == expected_benchmarks
    )
    intervals, valid_draws = primary_clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )

    component_assignments = {}
    component_points = {}
    component_by_benchmark = {}
    component_insufficient = {}
    for route_name in ROUTE_FIELDS:
        component_assignments[route_name] = assign_component_strata(
            rows, route_name=route_name
        )
        route_point, route_by_benchmark, route_missing = summarize_component_route(
            rows,
            route_name=route_name,
            minimum_states_per_stratum=minimum,
        )
        component_points[route_name] = route_point
        component_by_benchmark[route_name] = route_by_benchmark
        component_insufficient[route_name] = route_missing
    component_intervals, regression_intervals, component_valid_draws = (
        component_clustered_bootstrap(
            rows,
            resamples=args.bootstrap_resamples,
            seed=args.seed + 1,
        )
    )
    regressions = {
        outcome: weighted_factorial_regression(rows, outcome=outcome)
        for outcome in ("frozen_delta", "matched_gc_minus_u_accept")
    }
    component_comparisons = {}
    if all(component_points.values()):
        component_comparisons = {
            "joint_minus_alignment_root_interaction": (
                component_points["joint"]["root_interaction"]
                - component_points["visual_only"]["root_interaction"]
            ),
            "joint_minus_reliability_root_interaction": (
                component_points["joint"]["root_interaction"]
                - component_points["reliability_only"]["root_interaction"]
            ),
            "joint_minus_alignment_accept_interaction": (
                component_points["joint"]["accept_interaction"]
                - component_points["visual_only"]["accept_interaction"]
            ),
            "joint_minus_reliability_accept_interaction": (
                component_points["joint"]["accept_interaction"]
                - component_points["reliability_only"]["accept_interaction"]
            ),
        }

    component_support_complete = all(
        not missing and set(component_by_benchmark[route]) == expected_benchmarks
        for route, missing in component_insufficient.items()
    )
    directional = bool(
        support_complete
        and point.get("low_root_delta", 0.0) < 0.0
        and point.get("high_root_delta", 0.0) > 0.0
        and point.get("root_interaction", 0.0) > 0.0
        and point.get("low_accept_delta", 0.0) < 0.0
        and point.get("high_accept_delta", 0.0) > 0.0
        and point.get("accept_interaction", 0.0) > 0.0
    )
    development_component_gate = bool(
        component_support_complete
        and component_points.get("visual_only", {}).get("root_interaction", 0.0)
        > 0.0
        and component_points.get("visual_only", {}).get("accept_interaction", 0.0)
        > 0.0
        and component_comparisons.get(
            "joint_minus_reliability_root_interaction", 0.0
        )
        > 0.0
        and component_comparisons.get(
            "joint_minus_reliability_accept_interaction", 0.0
        )
        > 0.0
    )
    required_draws = max(100, int(0.90 * args.bootstrap_resamples))
    strict_metrics = (
        ("low_root_delta", 1, "negative"),
        ("high_root_delta", 0, "positive"),
        ("root_interaction", 0, "positive"),
        ("low_accept_delta", 1, "negative"),
        ("high_accept_delta", 0, "positive"),
        ("accept_interaction", 0, "positive"),
    )
    strict_directional = bool(
        directional
        and valid_draws >= required_draws
        and all(
            (
                intervals.get(metric, [0.0, 0.0])[edge] < 0.0
                if direction == "negative"
                else intervals.get(metric, [0.0, 0.0])[edge] > 0.0
            )
            for metric, edge, direction in strict_metrics
        )
    )
    comparison_ci = component_intervals.get("comparisons", {})
    strict_incremental_visual = bool(
        component_valid_draws >= required_draws
        and comparison_ci.get(
            "joint_minus_reliability_root_interaction", [0.0, 0.0]
        )[0]
        > 0.0
        and comparison_ci.get(
            "joint_minus_reliability_accept_interaction", [0.0, 0.0]
        )[0]
        > 0.0
    )
    route_ci = component_intervals.get("routes", {})
    strict_candidate_alignment = bool(
        component_valid_draws >= required_draws
        and route_ci.get("visual_only", {})
        .get("root_interaction", [0.0, 0.0])[0]
        > 0.0
        and route_ci.get("visual_only", {})
        .get("accept_interaction", [0.0, 0.0])[0]
        > 0.0
    )
    if args.analysis_role == "development":
        decision = (
            "go_to_fresh_independent_confirmation"
            if directional and development_component_gate
            else "development_no_go"
        )
    else:
        decision = (
            "validated"
            if (
                component_support_complete
                and strict_directional
                and strict_candidate_alignment
                and strict_incremental_visual
            )
            else "validation_no_go"
        )

    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "warning": (
            "Candidate-relative scores were introduced after v11 outcomes were "
            "known. A development pass still requires image-disjoint confirmation."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "frozen_rule": {
            "candidate_alignment_field": VISUAL_ALIGNMENT_FIELD,
            "visual_weight": VISUAL_WEIGHT,
            "source_reliability_weight": 1.0 - VISUAL_WEIGHT,
            "tail_fraction": TAIL_FRACTION,
            "candidate_set_support": (
                "minimum of true-minus-mean and true-minus-wrong candidate-set "
                "log probability mass"
            ),
            "uses_source_hit_outcomes": False,
            "uses_accepted_length_outcomes": False,
            "uses_target_token_id": False,
            "minimum_states_per_stratum": minimum,
        },
        "eligibility_audit": eligibility_audit,
        "arbitration_audit": arbitration_audit,
        "point_estimates": point,
        "by_benchmark": by_benchmark,
        "insufficient_support_benchmarks": insufficient,
        "support_complete": support_complete,
        "cluster_bootstrap_95_ci": intervals,
        "component_point_estimates": component_points,
        "component_comparisons": component_comparisons,
        "component_by_benchmark": component_by_benchmark,
        "component_insufficient_support_benchmarks": component_insufficient,
        "component_support_complete": component_support_complete,
        "component_cluster_bootstrap_95_ci": component_intervals,
        "factorial_regression": regressions,
        "factorial_regression_cluster_bootstrap_95_ci": regression_intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "component_bootstrap_valid_all_route_all_benchmark_draws": (
            component_valid_draws
        ),
        "bootstrap_required_valid_draws": required_draws,
        "development_directional_double_crossover": directional,
        "development_component_gate": development_component_gate,
        "strict_confirmatory_double_crossover": strict_directional,
        "strict_candidate_alignment_gate": strict_candidate_alignment,
        "strict_incremental_visual_gate": strict_incremental_visual,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if support_complete and valid_draws:
        plot(payload, args.output_dir / "candidate_visual_alignment_triptych")
    print(
        json.dumps(
            {
                "point_estimates": point,
                "component_point_estimates": component_points,
                "component_comparisons": component_comparisons,
                "insufficient_support_benchmarks": insufficient,
                "development_directional_double_crossover": directional,
                "development_component_gate": development_component_gate,
                "strict_confirmatory_double_crossover": strict_directional,
                "strict_candidate_alignment_gate": strict_candidate_alignment,
                "strict_incremental_visual_gate": strict_incremental_visual,
                "decision": decision,
                "output": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
