#!/usr/bin/env python3
"""Redraw both Intro figures with one non-overlapping pastel palette."""

import json
from pathlib import Path

from matplotlib.colors import to_hex, to_rgb

import analyze_accept_length_visuality_problem as visuality
import analyze_candidate_alignment_quantiles as alignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    PROJECT_ROOT
    / "outputs/experiments/selective_reuse/"
    "20260903-043841-candidate-alignment-validation100-gpu2"
)
OUTPUT_DIR = RUN_ROOT / "intro_shared_pastel_palette"
PALETTE = {
    "thistle": "#cdb4db",
    "pastel_petal": "#ffc8dd",
    "baby_pink": "#ffafcc",
    "icy_blue": "#bde0fe",
    "sky_blue": "#a2d2ff",
    "soft_periwinkle": "#9381ff",
    "periwinkle": "#b8b8ff",
    "ghost_white": "#f8f7ff",
}


def darken(color: str, factor: float = 0.62) -> str:
    return to_hex(tuple(channel * factor for channel in to_rgb(color)))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: three colors used only by the candidate-alignment figure.
    alignment.U_FILL = "#DCEAF3"
    alignment.U_EDGE = "#176FA6"
    alignment.GC_FILL = "#E5E2F2"
    alignment.GC_EDGE = "#6558B1"
    alignment.GAIN_COLOR = "#159A9C"
    alignment.LEFT_AXIS_COLOR = alignment.U_EDGE
    alignment.RIGHT_AXIS_COLOR = "#117A7D"
    alignment_summary = RUN_ROOT / (
        "validated_candidate_only_visual_alignment/"
        "candidate_alignment_quintile_trend/summary.json"
    )
    alignment.render(
        json.loads(alignment_summary.read_text(encoding="utf-8")),
        OUTPUT_DIR / "candidate_alignment_quantile_trend",
    )

    # Figure 2: requested three-color periwinkle variant.
    visuality.FLIP_FILL = "#b7d5cf"
    visuality.FLIP_EDGE = "#2c7375"
    visuality.FLIP_EDGE_WIDTH = 1.2
    visuality.VISUAL_COLOR = "#e64113"
    visuality.VISUAL_INTERIOR_COLOR = PALETTE["ghost_white"]
    visuality.LEFT_AXIS_COLOR = visuality.FLIP_EDGE
    visuality.RIGHT_AXIS_COLOR = visuality.VISUAL_COLOR
    visuality_summary = RUN_ROOT / "accepted_length_visuality_problem/summary.json"
    visuality.plot(
        json.loads(visuality_summary.read_text(encoding="utf-8")),
        OUTPUT_DIR / "accepted_length_visuality_problem_5x3",
        figsize=(5.0, 3.0),
        preserve_canvas=True,
    )

    for path in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(path)


if __name__ == "__main__":
    main()
