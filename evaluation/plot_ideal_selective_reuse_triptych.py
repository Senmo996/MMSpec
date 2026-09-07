"""Plot a clearly labelled synthetic target pattern for selective reuse.

The input shipped beside this script contains hand-authored illustrative
values.  This utility exists only to preview the evidence pattern that a future
experiment would need to support.  It must never be used as an empirical plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ORANGE = "#E8792E"
BLUE = "#2864DC"
INK = "#17212B"
MUTED = "#66717D"
GRID = "#D9DEE4"


def _errors(value: float, interval: Sequence[float]) -> np.ndarray:
    return np.asarray(
        [[value - float(interval[0])], [float(interval[1]) - value]],
        dtype=float,
    )


def _grouped_bars(
    axis,
    point: dict,
    intervals: dict,
    *,
    u_metric: str,
    gc_metric: str,
    ylabel: str,
) -> None:
    x = np.arange(2)
    width = 0.32
    for offset, source, metric, color in (
        (-width / 2, "U", u_metric, ORANGE),
        (width / 2, "G/C", gc_metric, BLUE),
    ):
        values = [float(point[tail][metric]) for tail in ("low", "high")]
        lows = [
            values[index] - float(intervals[tail][metric][0])
            for index, tail in enumerate(("low", "high"))
        ]
        highs = [
            float(intervals[tail][metric][1]) - values[index]
            for index, tail in enumerate(("low", "high"))
        ]
        bars = axis.bar(
            x + offset,
            values,
            width,
            yerr=np.asarray([lows, highs]),
            capsize=2.8,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": INK},
            zorder=3,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.065,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=INK,
                weight="bold",
            )
    axis.set_ylim(0.0, 1.06)
    axis.set_ylabel(ylabel)


def draw(payload: dict, output_stem: Path) -> None:
    if payload.get("status") != "synthetic_target_pattern":
        raise ValueError("refusing to render an input not marked synthetic")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#9EA7B0",
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    point = payload["point_estimates"]
    intervals = payload["confidence_intervals_95"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    fig.patch.set_facecolor("white")

    _grouped_bars(
        axes[0],
        point,
        intervals,
        u_metric="u_coverage",
        gc_metric="gc_coverage",
        ylabel="Source availability",
    )
    axes[0].set_title("(a) U supplies broad coverage", loc="left", weight="bold")
    axes[0].annotate(
        "U stays broadly available",
        xy=(1 - 0.16, point["high"]["u_coverage"]),
        xytext=(0.50, 0.67),
        fontsize=6.7,
        color="#984511",
        ha="center",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9),
    )

    _grouped_bars(
        axes[1],
        point,
        intervals,
        u_metric="u_conditional_top8_recall",
        gc_metric="gc_conditional_top8_recall",
        ylabel="Conditional top-8 recall",
    )
    axes[1].set_title("(b) The better source switches", loc="left", weight="bold")
    axes[1].text(
        0.0,
        0.16,
        "U > G/C",
        ha="center",
        va="center",
        fontsize=7.0,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#A64B12", ec="none"),
    )
    axes[1].text(
        1.0,
        0.16,
        "G/C > U",
        ha="center",
        va="center",
        fontsize=7.0,
        color="white",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#1749A5", ec="none"),
    )

    x = np.arange(2)
    metric = "gc_minus_u_matched_accept"
    utility_values = [float(point[tail][metric]) for tail in ("low", "high")]
    utility_errors = [
        _errors(utility_values[index], intervals[tail][metric]).ravel()
        for index, tail in enumerate(("low", "high"))
    ]
    axes[2].bar(
        x,
        utility_values,
        0.52,
        yerr=np.asarray(utility_errors).T,
        capsize=3.1,
        color=[ORANGE, BLUE],
        edgecolor="white",
        linewidth=0.7,
        error_kw={"elinewidth": 1.05, "capthick": 1.05, "ecolor": INK},
        zorder=3,
    )
    axes[2].axhline(0.0, color="#5D6670", linestyle=(0, (3, 2)), lw=0.9, zorder=2)
    axes[2].set_ylim(-0.72, 0.78)
    axes[2].set_ylabel("G/C − U accepted tokens\n(equal node budget)")
    axes[2].set_title("(c) Utility reverses", loc="left", weight="bold")
    axes[2].text(
        0,
        -0.20,
        f"{utility_values[0]:+.2f}\nU favored",
        ha="center",
        va="center",
        fontsize=7.0,
        color="white",
        weight="bold",
    )
    axes[2].text(
        1,
        0.24,
        f"{utility_values[1]:+.2f}\nG/C favored",
        ha="center",
        va="center",
        fontsize=7.0,
        color="white",
        weight="bold",
    )
    tick_labels = ["Low visual\n(bottom 20%)", "High visual\n(top 20%)"]
    for axis in axes:
        axis.set_xticks(x, tick_labels)
        axis.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.tick_params(axis="y", length=2.5)

    legend = [
        Line2D([0], [0], color=ORANGE, lw=5.5, label="Persistent U"),
        Line2D([0], [0], color=BLUE, lw=5.5, label="Request-local G/C"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.815, 0.925),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
        fontsize=6.9,
    )
    fig.suptitle(
        "Ideal evidence pattern: multimodal reuse requires state-dependent selection",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=10.0,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.900,
        "HYPOTHETICAL TARGET · SYNTHETIC VALUES · NOT EXPERIMENTAL RESULTS",
        ha="left",
        va="center",
        fontsize=6.5,
        weight="bold",
        color="#7B4B00",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFF0C8", ec="#E5B44A", lw=0.7),
    )
    fig.text(
        0.992,
        0.015,
        "Desired future held-out pattern; error bars illustrate cluster-bootstrap 95% CIs. MME excluded.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.75, bottom=0.23, wspace=0.45)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    draw(payload, args.output_stem)


if __name__ == "__main__":
    main()
