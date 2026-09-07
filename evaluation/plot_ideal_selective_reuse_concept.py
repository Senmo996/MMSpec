"""Draw the *conceptual* target figure for multimodal selective reuse.

This figure is intentionally schematic.  It must not be presented as an
empirical result: the curves are hand-designed to communicate the hypothesis
that the best reuse source can change with visual dependence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ORANGE = "#E8792E"
BLUE = "#2864DC"
GREEN = "#138A4B"
INK = "#17212B"
MUTED = "#66717D"


def _smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def draw(output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#AAB2BA",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Hand-designed curves: these values encode the intended hypothesis, not
    # measurements.  Avoid numeric ticks so they cannot be mistaken for data.
    x = np.linspace(0.0, 1.0, 300)
    s = _smoothstep(x)
    unigram = 0.78 - 0.43 * s
    grounded = 0.32 + 0.50 * s
    adaptive = np.minimum(0.92, np.maximum(unigram, grounded) + 0.085)
    crossover = float(x[np.argmin(np.abs(unigram - grounded))])

    fig, ax = plt.subplots(figsize=(3.48, 3.12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Regime backgrounds make the multimodal change of state immediately
    # visible without adding another panel.
    ax.axvspan(0.0, crossover, color="#FFF4EA", alpha=0.95, zorder=0)
    ax.axvspan(crossover, 1.0, color="#EDF4FF", alpha=0.95, zorder=0)
    ax.axvline(
        crossover,
        color="#8A949E",
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        zorder=1,
    )

    ax.plot(x, unigram, color=ORANGE, linewidth=2.4, zorder=3)
    ax.plot(x, grounded, color=BLUE, linewidth=2.4, zorder=3)
    ax.plot(x, adaptive, color=GREEN, linewidth=2.8, zorder=4)

    # Sparse anchors clarify that the curves represent two qualitatively
    # different token regimes, while remaining legible at one-column width.
    ax.scatter([0.12], [np.interp(0.12, x, unigram)], s=27, color=ORANGE,
               edgecolor="white", linewidth=0.7, zorder=5)
    ax.scatter([0.88], [np.interp(0.88, x, grounded)], s=27, color=BLUE,
               edgecolor="white", linewidth=0.7, zorder=5)

    ax.annotate(
        "broad coverage\nhelps generic tokens",
        xy=(0.12, np.interp(0.12, x, unigram)),
        xytext=(0.06, 0.55),
        textcoords="data",
        color="#9E4B14",
        fontsize=7.2,
        linespacing=1.18,
        arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8),
        ha="left",
        va="top",
    )
    ax.annotate(
        "visual context\nresolves ambiguity",
        xy=(0.88, np.interp(0.88, x, grounded)),
        xytext=(0.68, 0.55),
        textcoords="data",
        color="#1749A5",
        fontsize=7.2,
        linespacing=1.18,
        arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.8),
        ha="left",
        va="top",
    )

    ax.text(
        crossover,
        0.245,
        "source preference changes",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#C7CDD3", lw=0.7),
        zorder=6,
    )

    ax.text(
        0.04,
        0.965,
        "CONCEPTUAL TARGET · NOT MEASURED DATA",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#7B4B00",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFF0C8", ec="#E5B44A", lw=0.65),
        zorder=8,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.20, 1.00)
    ax.set_xticks([0.08, 0.92], ["Language-dominant", "Vision-dominant"])
    ax.set_yticks([])
    ax.set_xlabel("Visual dependence of the next-token decision", labelpad=5)
    ax.set_ylabel("Expected reuse utility\n(equal verifier budget)", labelpad=7)
    ax.set_title(
        "Multimodal reuse should be selective",
        loc="left",
        fontsize=10.2,
        weight="bold",
        color=INK,
        pad=8,
    )

    legend_handles = [
        Line2D([0], [0], color=ORANGE, lw=2.4, label="Persistent U"),
        Line2D([0], [0], color=BLUE, lw=2.4, label="Request-local G/C"),
        Line2D([0], [0], color=GREEN, lw=2.8, label="Selective GWTR"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.90),
        frameon=False,
        ncol=1,
        handlelength=2.1,
        handletextpad=0.6,
        borderaxespad=0,
        labelspacing=0.32,
        fontsize=7.2,
    )

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9AA4AE")
    ax.tick_params(axis="x", length=0, pad=4)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.20)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    draw(args.output_stem)


if __name__ == "__main__":
    main()
