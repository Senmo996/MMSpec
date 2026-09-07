"""Describe candidate-source behavior across alignment-score quantiles.

This is a post-hoc visualization of the frozen candidate-alignment signal.  It
does not replace the preregistered low/high validation result.
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_candidate_pairwise_win_panel import (  # noqa: E402
    _u_tie_aware_win,
)
from evaluation.analyze_candidate_visual_alignment_only import (  # noqa: E402
    assign_candidate_only_strata,
)
from evaluation.analyze_candidate_visual_support_panel import (  # noqa: E402
    _u_relative_support,
)
from evaluation.analyze_frozen_selective_conflict_regime import prepare  # noqa: E402
from evaluation.analyze_selective_reuse import load_selective_records  # noqa: E402
from evaluation.render_validated_candidate_alignment_triptych import (  # noqa: E402
    INK,
)


U_FILL = "#DCEAF3"
U_EDGE = "#176FA6"
GC_FILL = "#E5E2F2"
GC_EDGE = "#6558B1"
GAIN_COLOR = "#159A9C"
LEFT_AXIS_COLOR = "#42657A"
RIGHT_AXIS_COLOR = "#117A7D"
FONT_FAMILY = "Liberation Sans"


METRICS = (
    "u_relative_support",
    "gc_relative_support",
    "u_pairwise_win",
    "gc_pairwise_win",
    "accept_delta",
)


def assign_quantile_bins(rows: Sequence[dict], num_bins: int) -> int:
    """Assign disjoint within-benchmark bins from U-most to G/C-most."""

    if num_bins < 2:
        raise ValueError("num_bins must be at least two")
    assigned = 0
    for row in rows:
        rank = row.get("arbitration_visual_rank")
        row["candidate_alignment_bin"] = None
        if rank is not None:
            row["candidate_alignment_bin"] = min(int(float(rank) * num_bins), num_bins - 1)
            assigned += 1
    return assigned


def summarize(rows: Sequence[dict], num_bins: int) -> tuple[dict, dict]:
    """Return equal-benchmark means for every alignment bin."""

    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        result = {metric: [] for metric in METRICS}
        result["states"] = []
        for bin_index in range(num_bins):
            current = [
                row
                for row in benchmark_rows
                if row.get("candidate_alignment_bin") == bin_index
            ]
            if not current:
                raise ValueError(f"{benchmark} has no states in bin {bin_index + 1}")
            u_support = float(np.mean([_u_relative_support(row) for row in current]))
            u_win = float(np.mean([_u_tie_aware_win(row) for row in current]))
            result["states"].append(len(current))
            result["u_relative_support"].append(u_support)
            result["gc_relative_support"].append(1.0 - u_support)
            result["u_pairwise_win"].append(u_win)
            result["gc_pairwise_win"].append(1.0 - u_win)
            result["accept_delta"].append(
                float(
                    np.mean(
                        [
                            float(row["gc_matched_accept"])
                            - float(row["u_matched_accept"])
                            for row in current
                        ]
                    )
                )
            )
        by_benchmark[benchmark] = result
    point = {
        metric: np.mean(
            [summary[metric] for summary in by_benchmark.values()], axis=0
        ).tolist()
        for metric in METRICS
    }
    return point, by_benchmark


def clustered_bootstrap(
    rows: Sequence[dict], *, num_bins: int, resamples: int, seed: int
) -> tuple[dict, int]:
    """Bootstrap image clusters using pre-aggregated bin sufficient statistics."""

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)

    packed = {}
    for benchmark, clusters in grouped.items():
        values = np.zeros((len(clusters), num_bins, 4), dtype=np.float64)
        for cluster_index, cluster in enumerate(sorted(clusters)):
            for row in clusters[cluster]:
                bin_index = row.get("candidate_alignment_bin")
                if bin_index is None:
                    continue
                values[cluster_index, bin_index, 0] += 1.0
                values[cluster_index, bin_index, 1] += _u_relative_support(row)
                values[cluster_index, bin_index, 2] += _u_tie_aware_win(row)
                values[cluster_index, bin_index, 3] += (
                    float(row["gc_matched_accept"])
                    - float(row["u_matched_accept"])
                )
        packed[benchmark] = values

    rng = np.random.default_rng(int(seed))
    draws = {metric: [] for metric in METRICS}
    for _ in range(int(resamples)):
        benchmark_draws = {metric: [] for metric in METRICS}
        valid = True
        for benchmark in sorted(packed):
            values = packed[benchmark]
            sampled = values[
                rng.integers(0, len(values), size=len(values))
            ].sum(axis=0)
            counts = sampled[:, 0]
            if np.any(counts == 0):
                valid = False
                break
            u_support = sampled[:, 1] / counts
            u_win = sampled[:, 2] / counts
            benchmark_draws["u_relative_support"].append(u_support)
            benchmark_draws["gc_relative_support"].append(1.0 - u_support)
            benchmark_draws["u_pairwise_win"].append(u_win)
            benchmark_draws["gc_pairwise_win"].append(1.0 - u_win)
            benchmark_draws["accept_delta"].append(sampled[:, 3] / counts)
        if not valid:
            continue
        for metric in METRICS:
            draws[metric].append(np.mean(benchmark_draws[metric], axis=0))

    intervals = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[metric] = np.percentile(array, [2.5, 97.5], axis=0).T.tolist()
    return intervals, len(draws[METRICS[0]])


def _errors(values: np.ndarray, intervals: Sequence[Sequence[float]]) -> np.ndarray:
    bounds = np.asarray(intervals, dtype=float)
    return np.vstack((values - bounds[:, 0], bounds[:, 1] - values))


def _rounded_bars(
    axis,
    positions,
    values,
    width,
    facecolor,
    edgecolor,
    label,
    rounding_size=0.035,
    linewidth=0.95,
) -> None:
    for position, value in zip(positions, values):
        axis.add_patch(
            FancyBboxPatch(
                (position - width / 2, 0.0),
                width,
                value,
                boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=label,
                zorder=3,
            )
        )
        label = "_nolegend_"


def signed_support_margin(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert U support shares and intervals to signed G/C-minus-U margins."""

    u_values = np.asarray(payload["point_estimates"]["u_relative_support"], dtype=float)
    u_intervals = np.asarray(
        payload["cluster_bootstrap_95_ci"]["u_relative_support"], dtype=float
    )
    return 1.0 - 2.0 * u_values, np.column_stack(
        (1.0 - 2.0 * u_intervals[:, 1], 1.0 - 2.0 * u_intervals[:, 0])
    )


def render(payload: dict, output_stem: Path, *, signed_margin: bool = False) -> None:
    point = payload["point_estimates"]
    intervals = payload["cluster_bootstrap_95_ci"]
    num_bins = int(payload["num_bins"])
    x = np.arange(num_bins)
    labels = [f"Q{index + 1}" for index in range(num_bins)]
    width = min(0.34, 0.76 / 2)
    fig, axis = plt.subplots(figsize=(4.70, 2.65))

    if signed_margin:
        values, support_intervals = signed_support_margin(payload)
        axis.bar(
            x,
            values,
            0.54,
            yerr=_errors(values, support_intervals),
            color=[U_FILL if value < 0.0 else GC_FILL for value in values],
            edgecolor=[U_EDGE if value < 0.0 else GC_EDGE for value in values],
            linewidth=0.9,
            capsize=2.0,
            error_kw={"ecolor": INK, "elinewidth": 0.7, "capthick": 0.7},
            zorder=3,
        )
        axis.set_ylim(-1.08, 1.08)
        axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.set_ylabel("Visual-support margin (G/C − U)")
        axis.axhline(0.0, color=INK, linewidth=0.85, zorder=2)
    else:
        for offset, prefix, facecolor, edgecolor, label in (
            (-width / 2, "u", U_FILL, U_EDGE, "Token-only reuse"),
            (width / 2, "gc", GC_FILL, GC_EDGE, "Context-aware reuse"),
        ):
            values = np.asarray(point[f"{prefix}_relative_support"], dtype=float)
            _rounded_bars(
                axis,
                x + offset,
                values,
                width,
                facecolor,
                edgecolor,
                label,
            )
        axis.set_ylim(0.0, 1.08)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_ylabel("Candidate source share")

    gain_values = np.asarray(point["accept_delta"], dtype=float)
    gain_intervals = intervals["accept_delta"]
    raw_bound = max(abs(value) for interval in gain_intervals for value in interval)
    bound = max(0.5, raw_bound * 1.18)
    gain_axis = axis.twinx()
    gain_axis.errorbar(
        x,
        gain_values,
        yerr=_errors(gain_values, gain_intervals),
        color=GAIN_COLOR,
        marker="D",
        markerfacecolor="white",
        markeredgecolor=GAIN_COLOR,
        markeredgewidth=1.2,
        markersize=5.6,
        linewidth=2.35,
        elinewidth=1.0,
        capsize=2.8,
        solid_capstyle="round",
        zorder=6,
    )
    if not signed_margin:
        gain_axis.axhline(0.0, color=GAIN_COLOR, linestyle=(0, (3, 2)), lw=0.7)
    gain_axis.set_ylim(-bound, bound)
    gain_axis.set_ylabel(
        "Accepted-token gain vs. token-only",
        color=RIGHT_AXIS_COLOR,
        fontsize=7.6,
        fontweight="bold",
    )
    gain_axis.tick_params(axis="y", colors=RIGHT_AXIS_COLOR, labelsize=7.0)
    gain_axis.spines[["top", "left", "bottom"]].set_visible(False)
    gain_axis.spines["right"].set_color(RIGHT_AXIS_COLOR)
    gain_axis.spines["right"].set_linewidth(1.15)
    gain_axis.patch.set_visible(False)

    axis.set_xticks(x, labels)
    axis.set_xlim(-0.58, num_bins - 0.42)
    axis.grid(
        axis="y",
        color="#DDE4F0",
        linewidth=0.75,
        linestyle=(0, (1.5, 2.5)),
        zorder=0,
    )
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_color(LEFT_AXIS_COLOR)
    axis.spines["left"].set_linewidth(1.15)
    axis.spines["bottom"].set_color(INK)
    axis.spines["bottom"].set_linewidth(1.1)
    axis.tick_params(
        axis="x", length=3.0, width=0.9, color=INK, pad=4, labelsize=7.5, colors=INK
    )
    axis.tick_params(
        axis="y", length=3.5, width=0.9, labelsize=7.0, colors=LEFT_AXIS_COLOR
    )
    axis.yaxis.label.set_size(7.8)
    axis.yaxis.label.set_weight("bold")
    axis.yaxis.label.set_color(LEFT_AXIS_COLOR)
    tick_labels = axis.get_xticklabels()
    for tick in tick_labels:
        tick.set_weight("bold")
    for tick in (*axis.get_yticklabels(), *gain_axis.get_yticklabels()):
        tick.set_weight("bold")
    axis.text(
        x[0],
        -0.13,
        "Text-dominant",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color=U_EDGE,
        clip_on=False,
    )
    axis.text(
        x[-1],
        -0.13,
        "Vision-dominant",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.7,
        fontweight="bold",
        color=GC_EDGE,
        clip_on=False,
    )

    if signed_margin:
        handles = [
            Patch(facecolor=U_FILL, edgecolor=U_EDGE, linewidth=0.9),
            Patch(facecolor=GC_FILL, edgecolor=GC_EDGE, linewidth=0.9),
        ]
        legend_labels = ["U-supported", "G/C-supported"]
    else:
        handles, legend_labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color=GAIN_COLOR,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=GAIN_COLOR,
            linewidth=2.35,
            markersize=5.2,
        )
    )
    legend_labels.append("Accepted-token gain")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
        prop={"family": FONT_FAMILY, "size": 7.2, "weight": "bold"},
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.115, right=0.865, top=0.78, bottom=0.20)
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT_FAMILY)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=223607)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").resolve() == args.primary_summary.resolve():
        parser.error("output-dir would overwrite primary-summary; use a subdirectory")

    primary = json.loads(args.primary_summary.read_text(encoding="utf-8"))
    if primary.get("decision") != "validated":
        parser.error("primary candidate-only result is not validated")
    benchmarks = list(primary["benchmarks"])
    if any(name in {"MME", "MMSpec"} for name in benchmarks):
        parser.error("MME and MMSpec must remain excluded")
    rows = load_selective_records([Path(path) for path in primary["input_paths"]])
    prepare(rows, visual_metric="span2_mean_target_drop_fraction")
    assign_candidate_only_strata(
        rows, tail_fraction=float(primary["frozen_rule"]["tail_fraction"])
    )
    assigned = assign_quantile_bins(rows, args.num_bins)
    point, by_benchmark = summarize(rows, args.num_bins)
    intervals, valid_draws = clustered_bootstrap(
        rows,
        num_bins=args.num_bins,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    if valid_draws < max(100, int(0.9 * args.bootstrap_resamples)):
        parser.error(f"only {valid_draws} valid bootstrap draws")

    payload = {
        "schema_version": 1,
        "analysis_role": "post_hoc_candidate_alignment_quantile_trend",
        "not_a_validation_gate": True,
        "primary_summary": str(args.primary_summary.resolve()),
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME", "MMSpec"],
        "num_bins": args.num_bins,
        "bin_edges": np.linspace(0.0, 1.0, args.num_bins + 1).tolist(),
        "bin_definition": (
            "disjoint within-benchmark quantiles of candidate visual alignment; "
            "lowest is U-most and highest is G/C-most"
        ),
        "equal_benchmark_weight": True,
        "num_input_states": len(rows),
        "num_candidate_alignment_states": assigned,
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "by_benchmark": by_benchmark,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "monotonic_transition": {
            "u_relative_support_decreases": bool(
                np.all(np.diff(point["u_relative_support"]) < 0.0)
            ),
            "u_pairwise_win_decreases": bool(
                np.all(np.diff(point["u_pairwise_win"]) < 0.0)
            ),
            "gc_minus_u_accept_increases": bool(
                np.all(np.diff(point["accept_delta"]) > 0.0)
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render(payload, args.output_dir / "candidate_alignment_quantile_trend")
    render(
        payload,
        args.output_dir / "candidate_alignment_signed_margin",
        signed_margin=True,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "figure": str(args.output_dir / "candidate_alignment_quantile_trend.pdf"),
                "point_estimates": point,
                "monotonic_transition": payload["monotonic_transition"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
