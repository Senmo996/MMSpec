"""Analyze source preference under consensus visual-reliance strata.

Low/high labels are assigned without U/G/C hit outcomes.  Each state must show
agreement between an equal-shape mean-image intervention and a matched,
in-distribution wrong-image intervention.  Target-margin ranks are computed
within benchmark over full-view-valid eligible source conflicts.  The middle
and counterfactual-discordant states remain explicitly ambiguous.
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
CONSENSUS_TAIL_FRACTION = 0.40
MIN_DEVELOPMENT_STATES_PER_STRATUM = 3
MIN_VALIDATION_STATES_PER_STRATUM = 15
U_READY_PROBABILITY = 0.50
SOURCE_POPULATIONS = ("all_conflicts", "pre_request_confident_u")
VISUAL_ALIGNMENTS = ("current_span2", "grounded_context")
VISUAL_SIGNALS = ("target_margin_drop", "distribution_jsd")
MEAN_MARGIN_FIELD = "visual_probe_span2_mean_mean_target_margin_drop"
WRONG_MARGIN_FIELD = "visual_probe_span2_wrong_mean_target_margin_drop"
MEAN_JSD_FIELD = "visual_probe_span2_mean_mean_jsd"
WRONG_JSD_FIELD = "visual_probe_span2_wrong_mean_jsd"
MEAN_TOP1_FIELD = "visual_probe_span2_mean_top1_change_rate"
WRONG_TOP1_FIELD = "visual_probe_span2_wrong_top1_change_rate"
CONTEXT_MEAN_MARGIN_FIELD = "contextual_mean_margin_drop"
CONTEXT_WRONG_MARGIN_FIELD = "contextual_wrong_margin_drop"
CONTEXT_MEAN_TOP1_FIELD = "contextual_mean_top1_change_rate"
CONTEXT_WRONG_TOP1_FIELD = "contextual_wrong_top1_change_rate"
METRICS = (
    "low_mean_margin_drop",
    "low_wrong_margin_drop",
    "high_mean_margin_drop",
    "high_wrong_margin_drop",
    "mean_margin_separation",
    "wrong_margin_separation",
    "low_u_recall",
    "low_gc_recall",
    "high_u_recall",
    "high_gc_recall",
    "low_delta",
    "high_delta",
    "interaction",
    "low_fraction",
    "high_fraction",
    "ambiguous_fraction",
)


def is_u_ready(row: dict) -> bool:
    """Return an outcome-blind pre-request reliability gate for persistent U."""

    return bool(row.get("u_available_before_request", False)) and (
        row.get("u_row_top_probability_before_request") is not None
        and float(row["u_row_top_probability_before_request"])
        >= U_READY_PROBABILITY
    )


def add_grounded_context_features(rows: Sequence[dict]) -> None:
    """Align visual evidence with the generated tokens that key G/C reuse."""

    for row in rows:
        required_order = 3 if row.get("g_available", False) else 2
        row["contextual_required_order"] = required_order
        row["contextual_full_anchor_valid"] = False
        row[CONTEXT_MEAN_MARGIN_FIELD] = None
        row[CONTEXT_WRONG_MARGIN_FIELD] = None
        row[CONTEXT_MEAN_TOP1_FIELD] = None
        row[CONTEXT_WRONG_TOP1_FIELD] = None
        mean_drops = list(
            row.get("visual_probe_context3_mean_target_margin_drops", [])
        )
        wrong_drops = list(
            row.get("visual_probe_context3_wrong_target_margin_drops", [])
        )
        mean_changes = list(
            row.get("visual_probe_context3_mean_top1_changed", [])
        )
        wrong_changes = list(
            row.get("visual_probe_context3_wrong_top1_changed", [])
        )
        targets = list(row.get("visual_probe_context3_target_token_ids", []))
        full_top1 = list(
            row.get("visual_probe_context3_full_top1_token_ids", [])
        )
        lengths = {
            len(mean_drops),
            len(wrong_drops),
            len(mean_changes),
            len(wrong_changes),
            len(targets),
            len(full_top1),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < required_order:
            continue
        mean_drops = mean_drops[-required_order:]
        wrong_drops = wrong_drops[-required_order:]
        mean_changes = mean_changes[-required_order:]
        wrong_changes = wrong_changes[-required_order:]
        targets = targets[-required_order:]
        full_top1 = full_top1[-required_order:]
        row["contextual_full_anchor_valid"] = bool(full_top1 == targets)
        row[CONTEXT_MEAN_MARGIN_FIELD] = float(max(mean_drops))
        row[CONTEXT_WRONG_MARGIN_FIELD] = float(max(wrong_drops))
        row[CONTEXT_MEAN_TOP1_FIELD] = float(np.mean(mean_changes))
        row[CONTEXT_WRONG_TOP1_FIELD] = float(np.mean(wrong_changes))


def alignment_fields(
    visual_alignment: str,
    visual_signal: str = "target_margin_drop",
) -> tuple[str, str, str, str]:
    if visual_signal not in VISUAL_SIGNALS:
        raise ValueError(f"unsupported visual signal: {visual_signal}")
    if visual_signal == "distribution_jsd":
        if visual_alignment != "current_span2":
            raise ValueError(
                "distribution_jsd is available only for current_span2"
            )
        return (
            MEAN_JSD_FIELD,
            WRONG_JSD_FIELD,
            MEAN_TOP1_FIELD,
            WRONG_TOP1_FIELD,
        )
    if visual_alignment == "current_span2":
        return (
            MEAN_MARGIN_FIELD,
            WRONG_MARGIN_FIELD,
            MEAN_TOP1_FIELD,
            WRONG_TOP1_FIELD,
        )
    if visual_alignment == "grounded_context":
        return (
            CONTEXT_MEAN_MARGIN_FIELD,
            CONTEXT_WRONG_MARGIN_FIELD,
            CONTEXT_MEAN_TOP1_FIELD,
            CONTEXT_WRONG_TOP1_FIELD,
        )
    raise ValueError(f"unsupported visual alignment: {visual_alignment}")


def assign_consensus_visual_strata(
    rows: Sequence[dict],
    *,
    tail_fraction: float = CONSENSUS_TAIL_FRACTION,
    source_population: str = "all_conflicts",
    visual_alignment: str = "current_span2",
    visual_signal: str = "target_margin_drop",
) -> dict:
    """Assign strict low/high strata from two source-outcome-blind probes."""

    if not 0.0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must lie strictly between 0 and 0.5")
    if source_population not in SOURCE_POPULATIONS:
        raise ValueError(f"unsupported source population: {source_population}")
    if visual_alignment not in VISUAL_ALIGNMENTS:
        raise ValueError(f"unsupported visual alignment: {visual_alignment}")
    if visual_signal not in VISUAL_SIGNALS:
        raise ValueError(f"unsupported visual signal: {visual_signal}")
    if visual_alignment == "grounded_context":
        add_grounded_context_features(rows)
    mean_margin_field, wrong_margin_field, mean_top1_field, wrong_top1_field = (
        alignment_fields(visual_alignment, visual_signal)
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["consensus_visual_stratum"] = "ineligible"
        row["mean_margin_percentile"] = None
        row["wrong_margin_percentile"] = None
        if not row.get("frozen_eligible", False):
            continue
        if (
            source_population == "pre_request_confident_u"
            and not is_u_ready(row)
        ):
            row["consensus_visual_stratum"] = "u_not_ready"
            continue
        if not row.get("visual_probe_full_top1_matches_target", False):
            row["consensus_visual_stratum"] = "invalid_full_anchor"
            continue
        if visual_alignment == "grounded_context" and not row.get(
            "contextual_full_anchor_valid", False
        ):
            row["consensus_visual_stratum"] = "invalid_or_missing_context_anchor"
            continue
        if (
            row.get(mean_margin_field) is None
            or row.get(wrong_margin_field) is None
            or row.get(mean_top1_field) is None
            or row.get(wrong_top1_field) is None
        ):
            if visual_alignment == "grounded_context":
                row["consensus_visual_stratum"] = "insufficient_grounded_context"
                continue
            raise ValueError("counterfactual visual fields are missing")
        grouped[row["benchmark"]].append(row)

    counts = {}
    for benchmark, benchmark_rows in sorted(grouped.items()):
        mean_ranks = average_tie_percentiles(
            np.asarray([float(row[mean_margin_field]) for row in benchmark_rows])
        )
        wrong_ranks = average_tie_percentiles(
            np.asarray([float(row[wrong_margin_field]) for row in benchmark_rows])
        )
        benchmark_counts = defaultdict(int)
        for row, mean_rank, wrong_rank in zip(
            benchmark_rows, mean_ranks.tolist(), wrong_ranks.tolist()
        ):
            row["mean_margin_percentile"] = float(mean_rank)
            row["wrong_margin_percentile"] = float(wrong_rank)
            mean_top1_rate = float(row[mean_top1_field])
            wrong_top1_rate = float(row[wrong_top1_field])
            low_rank = (
                float(mean_rank) < float(tail_fraction)
                and float(wrong_rank) < float(tail_fraction)
            )
            high_rank = (
                float(mean_rank) >= 1.0 - float(tail_fraction)
                and float(wrong_rank) >= 1.0 - float(tail_fraction)
            )
            if low_rank and mean_top1_rate == 0.0 and wrong_top1_rate == 0.0:
                stratum = "low"
            elif high_rank and mean_top1_rate > 0.0 and wrong_top1_rate > 0.0:
                stratum = "high"
            else:
                stratum = "ambiguous"
            row["consensus_visual_stratum"] = stratum
            benchmark_counts[stratum] += 1
        benchmark_counts["reference_states"] = len(benchmark_rows)
        counts[benchmark] = dict(benchmark_counts)

    valid_anchors = sum(
        count.get("reference_states", 0) for count in counts.values()
    )
    population_rows = [
        row
        for row in rows
        if bool(row.get("frozen_eligible", False))
        and (
            source_population == "all_conflicts"
            or is_u_ready(row)
        )
    ]
    invalid_anchors = sum(
        not bool(row.get("visual_probe_full_top1_matches_target", False))
        for row in population_rows
    )
    return {
        "uses_source_hit_outcomes": False,
        "source_population": source_population,
        "visual_alignment": visual_alignment,
        "visual_signal": visual_signal,
        "u_ready_rule": (
            "U row existed before request and its pre-request top-candidate "
            "probability was >= 0.50"
            if source_population == "pre_request_confident_u"
            else None
        ),
        "reference_population": (
            "eligible source conflicts with full-view target top-1 match"
        ),
        "tail_fraction": float(tail_fraction),
        "low_rule": (
            "both margin ranks < 0.40 and both span-2 top-1 change rates = 0"
        ),
        "high_rule": (
            "both margin ranks >= 0.60 and both span-2 top-1 change rates > 0"
        ),
        "valid_anchor_states": int(valid_anchors),
        "invalid_full_anchor_states": int(invalid_anchors),
        "invalid_or_missing_context_anchor_states": int(
            sum(
                row.get("consensus_visual_stratum")
                == "invalid_or_missing_context_anchor"
                for row in rows
            )
        ),
        "source_population_states": len(population_rows),
        "u_not_ready_states": int(
            sum(
                bool(row.get("frozen_eligible", False)) and not is_u_ready(row)
                for row in rows
            )
            if source_population == "pre_request_confident_u"
            else 0
        ),
        "by_benchmark": counts,
    }


def summarize_consensus_benchmark(
    rows: Sequence[dict],
    *,
    source_population: str = "all_conflicts",
    visual_alignment: str = "current_span2",
    visual_signal: str = "target_margin_drop",
) -> dict | None:
    if source_population not in SOURCE_POPULATIONS:
        raise ValueError(f"unsupported source population: {source_population}")
    mean_margin_field, wrong_margin_field, _mean_top1, _wrong_top1 = (
        alignment_fields(visual_alignment, visual_signal)
    )
    eligible = [
        row
        for row in rows
        if row.get("frozen_eligible", False)
        and (
            source_population == "all_conflicts"
            or is_u_ready(row)
        )
    ]
    low = [row for row in rows if row.get("consensus_visual_stratum") == "low"]
    high = [row for row in rows if row.get("consensus_visual_stratum") == "high"]
    ambiguous = [
        row for row in rows if row.get("consensus_visual_stratum") == "ambiguous"
    ]
    if not eligible or not low or not high:
        return None
    output = {
        "eligible_states": len(eligible),
        "low_states": len(low),
        "high_states": len(high),
        "ambiguous_states": len(ambiguous),
        "low_mean_margin_drop": float(
            np.mean([row[mean_margin_field] for row in low])
        ),
        "low_wrong_margin_drop": float(
            np.mean([row[wrong_margin_field] for row in low])
        ),
        "high_mean_margin_drop": float(
            np.mean([row[mean_margin_field] for row in high])
        ),
        "high_wrong_margin_drop": float(
            np.mean([row[wrong_margin_field] for row in high])
        ),
        "low_u_recall": float(np.mean([row["frozen_u_hit"] for row in low])),
        "low_gc_recall": float(np.mean([row["frozen_gc_hit"] for row in low])),
        "high_u_recall": float(np.mean([row["frozen_u_hit"] for row in high])),
        "high_gc_recall": float(np.mean([row["frozen_gc_hit"] for row in high])),
        "low_delta": float(np.mean([row["frozen_delta"] for row in low])),
        "high_delta": float(np.mean([row["frozen_delta"] for row in high])),
        "low_fraction": len(low) / len(eligible),
        "high_fraction": len(high) / len(eligible),
        "ambiguous_fraction": len(ambiguous) / len(eligible),
    }
    output["mean_margin_separation"] = (
        output["high_mean_margin_drop"] - output["low_mean_margin_drop"]
    )
    output["wrong_margin_separation"] = (
        output["high_wrong_margin_drop"] - output["low_wrong_margin_drop"]
    )
    output["interaction"] = output["high_delta"] - output["low_delta"]
    return output


def consensus_macro_summary(
    rows: Sequence[dict],
    *,
    minimum_states_per_stratum: int,
    source_population: str = "all_conflicts",
    visual_alignment: str = "current_span2",
    visual_signal: str = "target_margin_drop",
) -> tuple[dict, dict, list[str]]:
    by_benchmark = {}
    insufficient = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_consensus_benchmark(
            [row for row in rows if row["benchmark"] == benchmark],
            source_population=source_population,
            visual_alignment=visual_alignment,
            visual_signal=visual_signal,
        )
        if (
            summary is None
            or int(summary["low_states"]) < int(minimum_states_per_stratum)
            or int(summary["high_states"]) < int(minimum_states_per_stratum)
        ):
            insufficient.append(benchmark)
            continue
        by_benchmark[benchmark] = summary
    if not by_benchmark:
        raise ValueError("no benchmark has supported consensus low/high strata")
    macro = {
        metric: float(np.mean([row[metric] for row in by_benchmark.values()]))
        for metric in METRICS
    }
    return macro, by_benchmark, insufficient


def clustered_bootstrap(
    rows: Sequence[dict],
    *,
    resamples: int,
    seed: int,
    source_population: str = "all_conflicts",
    visual_alignment: str = "current_span2",
    visual_signal: str = "target_margin_drop",
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
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            summary = summarize_consensus_benchmark(
                sampled,
                source_population=source_population,
                visual_alignment=visual_alignment,
                visual_signal=visual_signal,
            )
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


def _errors(value: float, interval: Sequence[float]) -> np.ndarray:
    return np.asarray([[value - interval[0]], [interval[1] - value]])


def plot(payload: dict, output_stem: Path) -> None:
    point = payload["point_estimates"]
    intervals = payload["cluster_bootstrap_95_ci"]
    visual_signal = payload.get("visual_signal", "target_margin_drop")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    x = np.arange(2)
    width = 0.32
    labels = (
        ["Low grounded\ncontext", "High grounded\ncontext"]
        if payload.get("visual_alignment", "current_span2") == "grounded_context"
        else ["Low visual\n(consensus)", "High visual\n(consensus)"]
    )

    for offset, prefix, color, label in (
        (-width / 2, "mean", ORANGE, "Mean image"),
        (width / 2, "wrong", BLUE, "Matched wrong image"),
    ):
        metrics = (f"low_{prefix}_margin_drop", f"high_{prefix}_margin_drop")
        values = [point[metric] for metric in metrics]
        errors = np.asarray(
            [
                [values[i] - intervals[metric][0] for i, metric in enumerate(metrics)],
                [intervals[metric][1] - values[i] for i, metric in enumerate(metrics)],
            ]
        )
        axes[0].bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=2.4,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axes[0].axhline(0.0, color="#5D6670", lw=0.8)
    axes[0].set_ylabel(
        "Full-distribution JSD"
        if visual_signal == "distribution_jsd"
        else "Target-margin drop"
    )
    axes[0].set_title("(a) Counterfactual check", loc="left", weight="bold")

    for offset, prefix, color, label in (
        (-width / 2, "u", ORANGE, "Persistent U"),
        (width / 2, "gc", BLUE, "Request-local G/C"),
    ):
        metrics = (f"low_{prefix}_recall", f"high_{prefix}_recall")
        values = [point[metric] for metric in metrics]
        errors = np.asarray(
            [
                [values[i] - intervals[metric][0] for i, metric in enumerate(metrics)],
                [intervals[metric][1] - values[i] for i, metric in enumerate(metrics)],
            ]
        )
        axes[1].bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=2.4,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_ylabel("Equal-budget root recall")
    axes[1].set_title("(b) Candidate recall", loc="left", weight="bold")

    delta_metrics = ("low_delta", "high_delta")
    delta_values = [point[metric] for metric in delta_metrics]
    delta_errors = np.asarray(
        [
            [
                delta_values[i] - intervals[metric][0]
                for i, metric in enumerate(delta_metrics)
            ],
            [
                intervals[metric][1] - delta_values[i]
                for i, metric in enumerate(delta_metrics)
            ],
        ]
    )
    axes[2].bar(
        x,
        delta_values,
        0.48,
        yerr=delta_errors,
        capsize=2.8,
        color=[ORANGE, BLUE],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[2].axhline(0.0, color="#5D6670", linestyle=(0, (3, 2)), lw=0.9)
    bound = max(
        0.15,
        max(abs(intervals[metric][edge]) for metric in delta_metrics for edge in (0, 1))
        * 1.15,
    )
    axes[2].set_ylim(-bound, bound)
    axes[2].set_ylabel("Δ root recall (G/C − U)")
    axes[2].set_title("(c) Source preference", loc="left", weight="bold")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.title.set_fontsize(8.0)

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
        (
            "Context-aligned counterfactual visual-demand test"
            if payload.get("visual_alignment", "current_span2")
            == "grounded_context"
            else (
                "Distributional counterfactual visual-demand test"
                if visual_signal == "distribution_jsd"
                else "Consensus counterfactual visual-demand test"
            )
        ),
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
        f"{payload['source_population'].replace('_', ' ')}",
        ha="left",
        va="center",
        fontsize=5.9,
        weight="bold",
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "Same text trajectory; mean-image and matched wrong-image agreement; "
        + (
            "full-vocabulary JSD; "
            if visual_signal == "distribution_jsd"
            else "target-margin loss; "
        )
        + "equal benchmark weight; image-cluster bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=5.3,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.77, bottom=0.25, wspace=0.56)
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
        choices=("development", "new_validation", "outcome_blind_validation"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--source-population",
        choices=SOURCE_POPULATIONS,
        default="all_conflicts",
    )
    parser.add_argument(
        "--visual-alignment",
        choices=VISUAL_ALIGNMENTS,
        default="current_span2",
    )
    parser.add_argument(
        "--visual-signal",
        choices=VISUAL_SIGNALS,
        default="target_margin_drop",
    )
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
    same_text_ratio = float(
        np.mean([bool(row["visual_probe_same_text_trajectory"]) for row in rows])
    )
    recomputed_vision_ratio = float(
        np.mean([bool(row["visual_probe_recomputed_vision_encoder"]) for row in rows])
    )
    if same_text_ratio != 1.0 or recomputed_vision_ratio != 1.0:
        parser.error("visual probes failed same-path/recomputed-vision audit")

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    stratum_audit = assign_consensus_visual_strata(
        rows,
        source_population=args.source_population,
        visual_alignment=args.visual_alignment,
        visual_signal=args.visual_signal,
    )
    minimum = (
        MIN_DEVELOPMENT_STATES_PER_STRATUM
        if args.analysis_role == "development"
        else MIN_VALIDATION_STATES_PER_STRATUM
    )
    try:
        point, by_benchmark, insufficient = consensus_macro_summary(
            rows,
            minimum_states_per_stratum=minimum,
            source_population=args.source_population,
            visual_alignment=args.visual_alignment,
            visual_signal=args.visual_signal,
        )
    except ValueError:
        point = {}
        by_benchmark = {}
        insufficient = benchmarks
    support_complete = not insufficient and set(by_benchmark) == EXPECTED_BENCHMARKS
    intervals, valid_draws = clustered_bootstrap(
        rows,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
        source_population=args.source_population,
        visual_alignment=args.visual_alignment,
        visual_signal=args.visual_signal,
    )
    required_valid_draws = max(100, int(0.90 * args.bootstrap_resamples))

    directional = bool(
        support_complete
        and point.get("low_delta", 0.0) < 0.0
        and point["high_delta"] > 0.0
        and point["interaction"] > 0.0
    )
    strict = bool(
        directional
        and valid_draws >= required_valid_draws
        and intervals.get("low_delta", [0.0, 0.0])[1] < 0.0
        and intervals.get("high_delta", [0.0, 0.0])[0] > 0.0
        and intervals.get("interaction", [0.0, 0.0])[0] > 0.0
    )
    if args.analysis_role == "development":
        decision = "go_to_independent_confirmation" if directional else "development_no_go"
    else:
        decision = "validated" if strict else "validation_no_go"

    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": "all",
        "source_population": args.source_population,
        "visual_alignment": args.visual_alignment,
        "visual_signal": args.visual_signal,
        "warning": (
            "Confirmatory only when the counterfactual rule and analysis gate "
            "were frozen before inspecting image-disjoint validation outcomes."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "probe_audit": {
            "same_text_trajectory_ratio": same_text_ratio,
            "recomputed_vision_encoder_ratio": recomputed_vision_ratio,
            "wrong_image_category_fallback_ratio": float(
                np.mean(
                    [
                        bool(row["visual_probe_wrong_image_used_category_fallback"])
                        for row in rows
                    ]
                )
            ),
        },
        "frozen_rule": {
            "mean_margin_field": alignment_fields(
                args.visual_alignment, args.visual_signal
            )[0],
            "wrong_margin_field": alignment_fields(
                args.visual_alignment, args.visual_signal
            )[1],
            "mean_top1_field": alignment_fields(
                args.visual_alignment, args.visual_signal
            )[2],
            "wrong_top1_field": alignment_fields(
                args.visual_alignment, args.visual_signal
            )[3],
            "visual_alignment": args.visual_alignment,
            "visual_signal": args.visual_signal,
            "consensus_tail_fraction": CONSENSUS_TAIL_FRACTION,
            "u_ready_probability_threshold": (
                U_READY_PROBABILITY
                if args.source_population == "pre_request_confident_u"
                else None
            ),
            "u_ready_rule": stratum_audit["u_ready_rule"],
            "minimum_states_per_stratum": minimum,
            "uses_source_hit_outcomes": False,
            "outcome": "equal-root-budget G/C-minus-U target-token hit",
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
        "bootstrap_required_valid_draws": required_valid_draws,
        "development_directional_crossover": directional,
        "strict_confirmatory_crossover": strict,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if support_complete and valid_draws:
        plot(payload, args.output_dir / "consensus_visual_triptych")
    print(
        json.dumps(
            {
                "point_estimates": point,
                "insufficient_support_benchmarks": insufficient,
                "development_directional_crossover": directional,
                "strict_confirmatory_crossover": strict,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
