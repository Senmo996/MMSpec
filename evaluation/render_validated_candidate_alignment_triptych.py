"""Render a publication-ready triptych from a validated frozen summary.

This renderer changes presentation only.  It refuses non-validated summaries
and records hashes of both the source summary and rendered artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ORANGE = "#D08A3E"
BLUE = "#5E62A9"
INK = "#25323B"
MUTED = "#66737D"
GRID = "#DCE4E8"
GAIN_COLOR = "#2E7D6E"
REGIME_LABELS = ("U-aligned", "G/C-aligned")
SOURCE_STYLES = (
    ("u", ORANGE, "Persistent U"),
    ("gc", BLUE, "Request-local G/C"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _errors(values, names, intervals):
    return np.asarray(
        [
            [value - intervals[name][0] for value, name in zip(values, names)],
            [intervals[name][1] - value for value, name in zip(values, names)],
        ]
    )


def _plot_grouped_bars(
    axis, points, intervals, suffix, ylabel, *, labels_inside=False
) -> None:
    x = np.arange(2)
    width = 0.32
    for offset, (prefix, color, label) in zip((-width / 2, width / 2), SOURCE_STYLES):
        names = (f"low_{prefix}_{suffix}", f"high_{prefix}_{suffix}")
        values = [points[name] for name in names]
        axis.bar(
            x + offset,
            values,
            width,
            yerr=_errors(values, names, intervals),
            color=color,
            edgecolor="none",
            capsize=2.4,
            error_kw={"ecolor": INK, "elinewidth": 0.8, "capthick": 0.8},
            label=label,
            zorder=3,
        )
        for position, name, value in zip(x + offset, names, values):
            if labels_inside:
                axis.text(
                    position,
                    0.025,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    weight="semibold",
                    color="white",
                    zorder=5,
                )
            else:
                axis.annotate(
                    f"{value:.2f}",
                    xy=(position, intervals[name][1]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    weight="semibold",
                    color=INK,
                    annotation_clip=False,
                    zorder=5,
                )
    axis.set_ylim(0.0, 1.08)
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.set_ylabel(ylabel)


def render(
    payload: dict,
    support_payload: dict,
    win_payload: dict,
    output_stem: Path,
) -> None:
    point = payload["point_estimates"]
    intervals = payload["cluster_bootstrap_95_ci"]
    support = support_payload["point_estimates"]
    support_intervals = support_payload["cluster_bootstrap_95_ci"]
    wins = win_payload["point_estimates"]
    win_intervals = win_payload["cluster_bootstrap_95_ci"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    x = np.arange(2)

    _plot_grouped_bars(
        axes[0],
        support,
        support_intervals,
        "relative_support",
        "Relative visual support",
    )
    _plot_grouped_bars(
        axes[1],
        wins,
        win_intervals,
        "pairwise_win",
        "Matched-prefix win probability",
        labels_inside=True,
    )

    gain_names = ("low_accept_delta", "high_accept_delta")
    gain_values = [point[name] for name in gain_names]
    raw_bound = max(
        abs(intervals[name][edge])
        for name in gain_names
        for edge in (0, 1)
    )
    bound = max(0.5, raw_bound * 1.32)
    gain_axis = axes[1].twinx()
    gain_axis.errorbar(
        x,
        gain_values,
        yerr=_errors(gain_values, gain_names, intervals),
        color=GAIN_COLOR,
        marker="D",
        markerfacecolor="white",
        markeredgecolor=GAIN_COLOR,
        markeredgewidth=1.0,
        markersize=5.2,
        linewidth=1.8,
        elinewidth=0.9,
        capsize=3.0,
        capthick=0.9,
        zorder=6,
    )
    gain_axis.axhline(0.0, color=GAIN_COLOR, linestyle=(0, (3, 2)), lw=0.75)
    gain_axis.set_ylim(-bound, bound)
    gain_axis.set_ylabel("Accepted tokens (G/C − U)", color=GAIN_COLOR)
    gain_axis.tick_params(axis="y", colors=GAIN_COLOR, labelsize=5.8)
    gain_axis.spines[["top", "left", "bottom"]].set_visible(False)
    gain_axis.spines["right"].set_color(GAIN_COLOR)
    gain_axis.spines["right"].set_linewidth(0.8)
    gain_axis.yaxis.label.set_size(5.9)
    gain_axis.patch.set_visible(False)
    label_pad = 0.055 * bound
    for index, (name, value) in enumerate(zip(gain_names, gain_values)):
        if value >= 0.0:
            label_y = intervals[name][1] + label_pad
            vertical = "bottom"
        else:
            label_y = intervals[name][0] - label_pad
            vertical = "top"
        gain_axis.annotate(
            f"{value:+.2f}",
            xy=(index, intervals[name][1] if value >= 0.0 else intervals[name][0]),
            xytext=(0, 4 if value >= 0.0 else -5),
            textcoords="offset points",
            ha="center",
            va=vertical,
            fontsize=6.1,
            weight="semibold",
            color=GAIN_COLOR,
            annotation_clip=False,
            zorder=7,
        )

    panel_titles = (
        "(a) Visual support",
        "(b) Pairwise win and accepted-token gain",
    )
    for axis, title in zip(axes, panel_titles):
        axis.set_xticks(x, REGIME_LABELS)
        axis.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#87949D")
        axis.spines[["left", "bottom"]].set_linewidth(0.7)
        axis.tick_params(axis="x", length=0, pad=3, labelsize=6.2, colors=INK)
        axis.tick_params(axis="y", length=3, width=0.7, labelsize=5.8, colors=INK)
        axis.yaxis.label.set_size(5.9)
        axis.yaxis.label.set_color(INK)
        axis.text(
            0.5,
            -0.28,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.0,
            weight="semibold",
            color=INK,
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color=GAIN_COLOR,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=GAIN_COLOR,
            markeredgewidth=1.0,
            linewidth=1.8,
            markersize=4.8,
        )
    )
    legend_labels.append("Accepted-token gain")
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=6.3,
        handlelength=1.5,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.82, bottom=0.28, wspace=0.32)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--support-summary", type=Path, required=True)
    parser.add_argument("--win-summary", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    analysis_role = payload.get("analysis_role")
    if analysis_role == "hierarchical_new_validation":
        if payload.get("decision") != "validated":
            parser.error("refusing to render a non-validated summary")
        if payload.get("strict_confirmatory_double_crossover") is not True:
            parser.error("strict confirmatory crossover is not true")
    elif analysis_role == "post_hoc_tail_fraction_sensitivity":
        if payload.get("decision") != "sensitivity_pass":
            parser.error("tail-fraction sensitivity did not pass")
        if payload.get("strict_double_crossover") is not True:
            parser.error("strict sensitivity crossover is not true")
    else:
        parser.error(f"unexpected primary analysis role: {analysis_role}")
    if payload.get("bootstrap_valid_all_benchmark_draws") != 10000:
        parser.error("expected exactly 10,000 valid bootstrap draws")
    if payload.get("frozen_rule", {}).get("source_reliability_weight") != 0.0:
        parser.error("summary is not the candidate-only route")
    tail_fraction = float(payload["frozen_rule"]["tail_fraction"])

    support_payload = json.loads(args.support_summary.read_text(encoding="utf-8"))
    if support_payload.get("analysis_role") != "descriptive_source_support_panel":
        parser.error("unexpected source-support summary role")
    if support_payload.get("not_a_validation_gate") is not True:
        parser.error("source-support panel must be marked descriptive")
    if support_payload.get("bootstrap_valid_all_benchmark_draws") != 10000:
        parser.error("source-support panel lacks 10,000 valid bootstrap draws")
    if float(support_payload.get("tail_fraction", tail_fraction)) != tail_fraction:
        parser.error("source-support tail fraction does not match")

    win_payload = json.loads(args.win_summary.read_text(encoding="utf-8"))
    if win_payload.get("analysis_role") != "descriptive_pairwise_win_panel":
        parser.error("unexpected pairwise-win summary role")
    if win_payload.get("not_a_validation_gate") is not True:
        parser.error("pairwise-win panel must be marked descriptive")
    if win_payload.get("bootstrap_valid_all_benchmark_draws") != 10000:
        parser.error("pairwise-win panel lacks 10,000 valid bootstrap draws")
    if float(win_payload.get("tail_fraction", tail_fraction)) != tail_fraction:
        parser.error("pairwise-win tail fraction does not match")

    render(payload, support_payload, win_payload, args.output_stem)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    manifest = {
        "schema_version": 1,
        "render_only": True,
        "analysis_role": analysis_role,
        "tail_fraction": tail_fraction,
        "source_summary": str(args.summary.resolve()),
        "source_summary_sha256": _sha256(args.summary),
        "source_support_summary": str(args.support_summary.resolve()),
        "source_support_summary_sha256": _sha256(args.support_summary),
        "pairwise_win_summary": str(args.win_summary.resolve()),
        "pairwise_win_summary_sha256": _sha256(args.win_summary),
        "png": str(png_path.resolve()),
        "png_sha256": _sha256(png_path),
        "pdf": str(pdf_path.resolve()),
        "pdf_sha256": _sha256(pdf_path),
    }
    manifest_path = args.output_stem.with_name(
        args.output_stem.name + "_render_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
