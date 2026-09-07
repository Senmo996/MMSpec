"""Explore alignment choices for selective-reuse diagnostics.

This utility is intentionally development-only. It compares pre-existing
visual covariates and fixed-horizon variants against root-token and trajectory
utility, but it does not declare confirmatory evidence or select examples.
Any rule motivated by this output must be frozen and evaluated on disjoint
images in a new run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_METRICS = (
    "visual_probe_jsd",
    "visual_probe_max_target_logprob_drop",
    "visual_probe_top1_disagreement_rate",
    "visual_probe_topk_union_size",
)
OUTCOMES = (
    "matched_gc_minus_u_accept",
    "gc_minus_u_root_top8_hit",
)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            row["gc_minus_u_root_top8_hit"] = int(row["gc_top8_hit"]) - int(
                row["u_top8_hit"]
            )
            records.append(row)
    return records


def add_fixed_horizon_metrics(records: list[dict], horizon: int = 4) -> None:
    """Add covariate-only future-window summaries on each fixed text path."""

    groups = defaultdict(list)
    for row in records:
        key = (
            row["benchmark"],
            row["question_id"],
            row.get("choice_index", 0),
            row.get("turn_index", 0),
        )
        groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["output_position"], row["iteration"]))
        for row in rows:
            start = int(row["output_position"])
            stop = start + int(horizon)
            window = [
                future
                for future in rows
                if start <= int(future["output_position"]) < stop
            ]
            for metric in (
                "visual_probe_jsd",
                "visual_probe_max_target_logprob_drop",
            ):
                values = [float(future[metric]) for future in window]
                row[f"future{horizon}_max_{metric}"] = max(values)
                row[f"future{horizon}_mean_{metric}"] = float(np.mean(values))


def average_tie_percentiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return (ranks - 0.5) / len(values)


def add_metric_percentiles(records: list[dict], metric: str) -> None:
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[(row["benchmark"], row["analysis_split"])].append(index)
    key = f"_percentile_{metric}"
    for indices in groups.values():
        values = np.asarray(
            [float(records[index][metric]) for index in indices],
            dtype=np.float64,
        )
        percentiles = average_tie_percentiles(values)
        for index, percentile in zip(indices, percentiles.tolist()):
            records[index][key] = float(percentile)


def summarize_one(
    records: list[dict], metric: str, outcome: str, split: str
) -> dict:
    selected = [
        row
        for row in records
        if (split == "all" or row["analysis_split"] == split)
        and row["u_available"]
        and row["gc_available"]
    ]
    percentile_key = f"_percentile_{metric}"
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in selected}):
        rows = [row for row in selected if row["benchmark"] == benchmark]
        low = [row for row in rows if row[percentile_key] < 0.2]
        high = [row for row in rows if row[percentile_key] >= 0.8]
        low_mean = float(np.mean([row[outcome] for row in low])) if low else None
        high_mean = (
            float(np.mean([row[outcome] for row in high])) if high else None
        )
        interaction = (
            high_mean - low_mean
            if low_mean is not None and high_mean is not None
            else None
        )
        by_benchmark[benchmark] = {
            "eligible_states": len(rows),
            "low_states": len(low),
            "high_states": len(high),
            "low_mean": low_mean,
            "high_mean": high_mean,
            "high_minus_low": interaction,
            "sign_crossover": bool(
                low_mean is not None
                and high_mean is not None
                and low_mean < 0.0 < high_mean
            ),
        }
    valid = [
        row for row in by_benchmark.values() if row["high_minus_low"] is not None
    ]
    return {
        "split": split,
        "metric": metric,
        "outcome": outcome,
        "eligible_states": len(selected),
        "valid_benchmarks": len(valid),
        "macro_low_mean": (
            float(np.mean([row["low_mean"] for row in valid])) if valid else None
        ),
        "macro_high_mean": (
            float(np.mean([row["high_mean"] for row in valid])) if valid else None
        ),
        "macro_high_minus_low": (
            float(np.mean([row["high_minus_low"] for row in valid]))
            if valid
            else None
        ),
        "positive_interaction_benchmarks": sum(
            row["high_minus_low"] > 0.0 for row in valid
        ),
        "sign_crossover_benchmarks": sum(row["sign_crossover"] for row in valid),
        "by_benchmark": by_benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")

    records = read_jsonl(args.records)
    add_fixed_horizon_metrics(records, args.horizon)
    metrics = [
        *BASE_METRICS,
        f"future{args.horizon}_max_visual_probe_jsd",
        f"future{args.horizon}_mean_visual_probe_jsd",
        f"future{args.horizon}_max_visual_probe_max_target_logprob_drop",
        f"future{args.horizon}_mean_visual_probe_max_target_logprob_drop",
    ]
    for metric in metrics:
        add_metric_percentiles(records, metric)

    analyses = [
        summarize_one(records, metric, outcome, split)
        for split in ("discovery", "heldout", "all")
        for outcome in OUTCOMES
        for metric in metrics
    ]
    payload = {
        "analysis_role": "development_only_not_confirmatory",
        "input": str(args.records.resolve()),
        "num_records": len(records),
        "fixed_future_horizon": args.horizon,
        "metric_family": metrics,
        "outcomes": OUTCOMES,
        "tail_definition": (
            "bottom and top 20% average-tie percentiles within benchmark and "
            "original split"
        ),
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ranked = sorted(
        (
            row
            for row in analyses
            if row["split"] == "discovery"
            and row["outcome"] == "matched_gc_minus_u_accept"
            and row["macro_high_minus_low"] is not None
        ),
        key=lambda row: row["macro_high_minus_low"],
        reverse=True,
    )
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "metric",
                        "macro_low_mean",
                        "macro_high_mean",
                        "macro_high_minus_low",
                        "positive_interaction_benchmarks",
                        "sign_crossover_benchmarks",
                    )
                }
                for row in ranked
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
