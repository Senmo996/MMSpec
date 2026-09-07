#!/usr/bin/env python3
"""Redraw the 5:3 figure from the cached analysis summary."""

import json
from pathlib import Path

from analyze_accept_length_visuality_problem import plot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/experiments/selective_reuse/"
    "20260903-043841-candidate-alignment-validation100-gpu2/"
    "accepted_length_visuality_problem"
)


def main() -> None:
    payload = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    output_stem = OUTPUT_DIR / "accepted_length_visuality_problem_5x3"
    plot(payload, output_stem, figsize=(5.0, 3.0), preserve_canvas=True)
    print(output_stem.with_suffix(".png"))
    print(output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
