"""Analyze a frozen visual-demand × source-reliability arbitration score.

This is an offline mechanistic diagnostic.  Its strata use only counterfactual
visual responses and source-row probabilities available before target
verification; U/G/C hits and accepted lengths are outcomes only.
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
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    COUNTERFACTUAL_BANK_PROTOCOL,
)


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
VISUAL_WEIGHT = 0.50
TAIL_FRACTION = 0.30
MIN_DEVELOPMENT_STATES_PER_STRATUM = 5
MIN_VALIDATION_STATES_PER_STRATUM = 15
MEAN_VISUAL_FIELD = "visual_probe_span2_mean_mean_target_margin_drop"
WRONG_VISUAL_FIELD = "visual_probe_span2_wrong_mean_target_margin_drop"
U_CONFIDENCE_FIELD = "u_row_top_probability"
GC_CONFIDENCE_FIELD = "root_transition_top_probability"
METRICS = (
    "low_visual_rank",
    "high_visual_rank",
    "low_gc_confidence_advantage_rank",
    "high_gc_confidence_advantage_rank",
    "low_u_recall",
    "low_gc_recall",
    "high_u_recall",
    "high_gc_recall",
    "low_root_delta",
    "high_root_delta",
    "root_interaction",
    "low_u_matched_accept",
    "low_gc_matched_accept",
    "high_u_matched_accept",
    "high_gc_matched_accept",
    "low_accept_delta",
    "high_accept_delta",
    "accept_interaction",
    "low_fraction",
    "high_fraction",
    "ambiguous_fraction",
)


def assign_arbitration_strata(
    rows: Sequence[dict],
    *,
    visual_weight: float = VISUAL_WEIGHT,
    tail_fraction: float = TAIL_FRACTION,
) -> dict:
    """Assign outcome-blind low/high arbitration regimes within benchmark."""

    if not 0.0 <= float(visual_weight) <= 1.0:
        raise ValueError("visual_weight must lie in [0, 1]")
    if not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must lie strictly between 0 and 0.5")
    grouped: dict[str, list[dict]] = defaultdict(list)
    invalid_reasons = defaultdict(int)
    for row in rows:
        row["arbitration_stratum"] = "ineligible"
        for field in (
            "arbitration_visual_rank",
            "arbitration_gc_confidence_advantage_rank",
            "arbitration_score",
            "arbitration_score_percentile",
        ):
            row[field] = None
        if not row.get("frozen_eligible", False):
            continue
        if not row.get("visual_probe_full_top1_matches_target", False):
            row["arbitration_stratum"] = "invalid_full_anchor"
            invalid_reasons["invalid_full_anchor"] += 1
            continue
        required = (
            MEAN_VISUAL_FIELD,
            WRONG_VISUAL_FIELD,
            U_CONFIDENCE_FIELD,
            GC_CONFIDENCE_FIELD,
        )
        if any(row.get(field) is None for field in required):
            row["arbitration_stratum"] = "missing_signal"
            invalid_reasons["missing_signal"] += 1
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
        mean_ranks = average_tie_percentiles(
            np.asarray(
                [float(row[MEAN_VISUAL_FIELD]) for row in benchmark_rows]
            )
        )
        wrong_ranks = average_tie_percentiles(
            np.asarray(
                [float(row[WRONG_VISUAL_FIELD]) for row in benchmark_rows]
            )
        )
        confidence_advantages = np.asarray(
            [
                float(row[GC_CONFIDENCE_FIELD])
                - float(row[U_CONFIDENCE_FIELD])
                for row in benchmark_rows
            ]
        )
        confidence_ranks = average_tie_percentiles(confidence_advantages)
        scores = []
        for row, mean_rank, wrong_rank, confidence_rank in zip(
            benchmark_rows,
            mean_ranks.tolist(),
            wrong_ranks.tolist(),
            confidence_ranks.tolist(),
        ):
            visual_rank = 0.5 * (float(mean_rank) + float(wrong_rank))
            score = (
                float(visual_weight) * visual_rank
                + (1.0 - float(visual_weight)) * float(confidence_rank)
            )
            row["arbitration_visual_rank"] = visual_rank
            row["arbitration_gc_confidence_advantage_rank"] = float(
                confidence_rank
            )
            row["arbitration_score"] = score
            scores.append(score)
        score_ranks = average_tie_percentiles(np.asarray(scores))
        counts = defaultdict(int)
        for row, score_rank in zip(benchmark_rows, score_ranks.tolist()):
            row["arbitration_score_percentile"] = float(score_rank)
            if float(score_rank) < float(tail_fraction):
                stratum = "low"
            elif float(score_rank) >= 1.0 - float(tail_fraction):
                stratum = "high"
            else:
                stratum = "ambiguous"
            row["arbitration_stratum"] = stratum
            counts[stratum] += 1
        counts["reference_states"] = len(benchmark_rows)
        by_benchmark[benchmark] = dict(counts)

    return {
        "uses_source_hit_outcomes": False,
        "uses_accepted_length_outcomes": False,
        "visual_weight": float(visual_weight),
        "source_reliability_weight": 1.0 - float(visual_weight),
        "tail_fraction": float(tail_fraction),
        "visual_signal": (
            "mean of within-benchmark mean-image and matched-wrong-image "
            "span-2 target-margin ranks"
        ),
        "source_reliability_signal": (
            "within-benchmark rank of G/C row top probability minus U row "
            "top probability"
        ),
        "u_confidence_field": U_CONFIDENCE_FIELD,
        "gc_confidence_field": GC_CONFIDENCE_FIELD,
        "valid_states": sum(
            counts.get("reference_states", 0)
            for counts in by_benchmark.values()
        ),
        "invalid_reasons": dict(invalid_reasons),
        "by_benchmark": by_benchmark,
    }


def summarize_benchmark(rows: Sequence[dict]) -> dict | None:
    low = [row for row in rows if row.get("arbitration_stratum") == "low"]
    high = [row for row in rows if row.get("arbitration_stratum") == "high"]
    ambiguous = [
        row for row in rows if row.get("arbitration_stratum") == "ambiguous"
    ]
    valid = low + high + ambiguous
    if not low or not high or not valid:
        return None

    def mean(group: Sequence[dict], field: str) -> float:
        return float(np.mean([float(row[field]) for row in group]))

    output = {
        "valid_states": len(valid),
        "low_states": len(low),
        "high_states": len(high),
        "ambiguous_states": len(ambiguous),
        "low_visual_rank": mean(low, "arbitration_visual_rank"),
        "high_visual_rank": mean(high, "arbitration_visual_rank"),
        "low_gc_confidence_advantage_rank": mean(
            low, "arbitration_gc_confidence_advantage_rank"
        ),
        "high_gc_confidence_advantage_rank": mean(
            high, "arbitration_gc_confidence_advantage_rank"
        ),
        "low_u_recall": mean(low, "frozen_u_hit"),
        "low_gc_recall": mean(low, "frozen_gc_hit"),
        "high_u_recall": mean(high, "frozen_u_hit"),
        "high_gc_recall": mean(high, "frozen_gc_hit"),
        "low_u_matched_accept": mean(low, "u_matched_accept"),
        "low_gc_matched_accept": mean(low, "gc_matched_accept"),
        "high_u_matched_accept": mean(high, "u_matched_accept"),
        "high_gc_matched_accept": mean(high, "gc_matched_accept"),
        "low_fraction": len(low) / len(valid),
        "high_fraction": len(high) / len(valid),
        "ambiguous_fraction": len(ambiguous) / len(valid),
    }
    output["low_root_delta"] = output["low_gc_recall"] - output["low_u_recall"]
    output["high_root_delta"] = (
        output["high_gc_recall"] - output["high_u_recall"]
    )
    output["root_interaction"] = (
        output["high_root_delta"] - output["low_root_delta"]
    )
    output["low_accept_delta"] = (
        output["low_gc_matched_accept"] - output["low_u_matched_accept"]
    )
    output["high_accept_delta"] = (
        output["high_gc_matched_accept"] - output["high_u_matched_accept"]
    )
    output["accept_interaction"] = (
        output["high_accept_delta"] - output["low_accept_delta"]
    )
    return output


def macro_summary(
    rows: Sequence[dict], *, minimum_states_per_stratum: int
) -> tuple[dict, dict, list[str]]:
    by_benchmark = {}
    insufficient = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_benchmark(
            [row for row in rows if row["benchmark"] == benchmark]
        )
        if (
            summary is None
            or summary["low_states"] < int(minimum_states_per_stratum)
            or summary["high_states"] < int(minimum_states_per_stratum)
        ):
            insufficient.append(benchmark)
            continue
        by_benchmark[benchmark] = summary
    if not by_benchmark:
        raise ValueError("no benchmark has supported arbitration strata")
    point = {
        metric: float(
            np.mean([summary[metric] for summary in by_benchmark.values()])
        )
        for metric in METRICS
    }
    return point, by_benchmark, insufficient


def clustered_bootstrap(
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
        summaries = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            sampled_indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in sampled_indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            summary = summarize_benchmark(sampled)
            if summary is not None:
                summaries.append(summary)
        if len(summaries) != len(grouped):
            continue
        valid_draws += 1
        for metric in METRICS:
            draws[metric].append(
                float(np.mean([summary[metric] for summary in summaries]))
            )
    intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
        if values
    }
    return intervals, valid_draws


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
    labels = ["Low\nscore", "High\nscore"]

    for offset, names, color, label in (
        (
            -width / 2,
            ("low_visual_rank", "high_visual_rank"),
            "#4C9FBE",
            "Visual demand",
        ),
        (
            width / 2,
            (
                "low_gc_confidence_advantage_rank",
                "high_gc_confidence_advantage_rank",
            ),
            "#8064A2",
            "G/C confidence advantage",
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
    axes[0].set_title("(a) Evidence", loc="left", weight="bold")
    axes[0].legend(
        loc="upper left",
        frameon=False,
        fontsize=4.7,
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
    axes[2].set_ylabel("G/C − U accepted tokens")
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
        "Multimodal reuse needs visual–reliability arbitration",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.905,
        f"{payload['analysis_role'].upper().replace('_', ' ')} · "
        f"{len(payload['benchmarks'])} benchmarks · MME excluded · "
        "50% visual + 50% source reliability",
        ha="left",
        va="center",
        fontsize=5.9,
        weight="bold",
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "Outcome-blind strata; equal benchmark weight; matched tree budget; "
        "image-cluster bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=5.3,
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
    parser.add_argument("--bootstrap-resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=57721)
    args = parser.parse_args()

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != EXPECTED_BENCHMARKS:
        parser.error(
            "expected exactly the eight non-MME benchmarks, got "
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
    stratum_audit = assign_arbitration_strata(rows)
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
    support_complete = not insufficient and set(by_benchmark) == EXPECTED_BENCHMARKS
    intervals, valid_draws = clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    required_draws = max(100, int(0.90 * args.bootstrap_resamples))

    directional = bool(
        support_complete
        and point.get("low_root_delta", 0.0) < 0.0
        and point["high_root_delta"] > 0.0
        and point["root_interaction"] > 0.0
        and point["low_accept_delta"] < 0.0
        and point["high_accept_delta"] > 0.0
        and point["accept_interaction"] > 0.0
    )
    strict_metrics = (
        ("low_root_delta", 1, "negative"),
        ("high_root_delta", 0, "positive"),
        ("root_interaction", 0, "positive"),
        ("low_accept_delta", 1, "negative"),
        ("high_accept_delta", 0, "positive"),
        ("accept_interaction", 0, "positive"),
    )
    strict = bool(
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
    decision = (
        "go_to_fresh_independent_confirmation"
        if args.analysis_role == "development" and directional
        else "development_no_go"
        if args.analysis_role == "development"
        else "validated"
        if strict
        else "validation_no_go"
    )

    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "warning": (
            "The score weights and tails were selected on development data; "
            "only a new image-disjoint validation can support the claim."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "frozen_rule": {
            "visual_weight": VISUAL_WEIGHT,
            "source_reliability_weight": 1.0 - VISUAL_WEIGHT,
            "tail_fraction": TAIL_FRACTION,
            "mean_visual_field": MEAN_VISUAL_FIELD,
            "wrong_visual_field": WRONG_VISUAL_FIELD,
            "u_confidence_field": U_CONFIDENCE_FIELD,
            "gc_confidence_field": GC_CONFIDENCE_FIELD,
            "minimum_states_per_stratum": minimum,
            "uses_source_hit_outcomes": False,
            "uses_accepted_length_outcomes": False,
            "primary_utility": "matched G/C-minus-U accepted length",
            "mechanism_outcome": "equal-root-budget G/C-minus-U target hit",
        },
        "eligibility_audit": eligibility_audit,
        "stratum_audit": stratum_audit,
        "point_estimates": point,
        "by_benchmark": by_benchmark,
        "insufficient_support_benchmarks": insufficient,
        "support_complete": support_complete,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "bootstrap_required_valid_draws": required_draws,
        "development_directional_double_crossover": directional,
        "strict_confirmatory_double_crossover": strict,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if support_complete and valid_draws:
        plot(payload, args.output_dir / "visual_reliability_triptych")
    print(
        json.dumps(
            {
                "point_estimates": point,
                "insufficient_support_benchmarks": insufficient,
                "development_directional_double_crossover": directional,
                "strict_confirmatory_double_crossover": strict,
                "decision": decision,
                "output": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
