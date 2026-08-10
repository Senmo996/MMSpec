"""Analyze how visual-grounding scores relate to tree routing and acceptance."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_training_free import load_turns


def safe_mean(values):
    return float(statistics.fmean(values)) if values else 0.0


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0 or y_scale == 0:
        return None
    return float(numerator / (x_scale * y_scale))


def summarize_rows(rows):
    accepts = [float(row.get("accept_len", 0)) for row in rows]
    ratios = [
        float(row.get("accept_ratio", 0.0))
        for row in rows
        if int(row.get("used_draft_len", 0)) > 0
    ]
    return {
        "num_iterations": len(rows),
        "avg_grounding_score": safe_mean(
            [float(row.get("grounding_score", 0.0)) for row in rows]
        ),
        "avg_accept_length": safe_mean(accepts),
        "accept_ge_1_ratio": (
            sum(value >= 1 for value in accepts) / len(accepts) if accepts else 0.0
        ),
        "avg_draft_accept_ratio": safe_mean(ratios),
        "avg_used_draft_len": safe_mean(
            [float(row.get("used_draft_len", 0)) for row in rows]
        ),
        "avg_verified_tree_nodes": safe_mean(
            [float(row.get("verified_tree_nodes", 0)) for row in rows]
        ),
    }


def analyze(jsonl_path, threshold):
    rows = [row for turn in load_turns(jsonl_path) for row in turn["trace"]]
    drafted = [row for row in rows if int(row.get("used_draft_len", 0)) > 0]
    scores = [float(row.get("grounding_score", 0.0)) for row in drafted]
    accept_lengths = [float(row.get("accept_len", 0)) for row in drafted]
    accept_ratios = [float(row.get("accept_ratio", 0.0)) for row in drafted]

    regimes = {
        "low_visual": [
            row for row in drafted if float(row.get("grounding_score", 0.0)) < threshold
        ],
        "high_visual": [
            row for row in drafted if float(row.get("grounding_score", 0.0)) >= threshold
        ],
    }
    shapes = defaultdict(list)
    for row in drafted:
        shape = f'{int(row.get("tree_width", 0))}x{int(row.get("tree_depth", 0))}'
        shapes[shape].append(row)

    score_bins = {}
    boundaries = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.000001)]
    for lower, upper in boundaries:
        label = f"[{lower:.2f},{min(upper, 1.0):.2f}{']' if upper > 1 else ')'}"
        score_bins[label] = summarize_rows(
            [
                row
                for row in drafted
                if lower <= float(row.get("grounding_score", 0.0)) < upper
            ]
        )

    return {
        "num_turns": len(load_turns(jsonl_path)),
        "num_iterations": len(rows),
        "num_drafted_iterations": len(drafted),
        "grounding_threshold": threshold,
        "grounding_accept_length_pearson": pearson(scores, accept_lengths),
        "grounding_accept_ratio_pearson": pearson(scores, accept_ratios),
        "by_visual_regime": {
            name: summarize_rows(values) for name, values in regimes.items()
        },
        "by_tree_shape": {
            name: summarize_rows(values) for name, values in sorted(shapes.items())
        },
        "by_grounding_score_bin": score_bins,
        "jsonl_path": str(Path(jsonl_path).resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--policies", required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.output_root)
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    payload = {
        "grounding_threshold": args.threshold,
        "policies": {
            policy: analyze(root / policy / "results.jsonl", args.threshold)
            for policy in policies
        },
    }
    output_path = Path(args.output) if args.output else root / "mechanism_analysis.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
