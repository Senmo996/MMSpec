"""Paired target-only summary for the wrong-image positive control."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .analyze_selective_reuse import (
        _benchmark_name,
        _read_jsonl,
        discover_result_paths,
    )
except ImportError:
    from analyze_selective_reuse import (
        _benchmark_name,
        _read_jsonl,
        discover_result_paths,
    )


METRICS = ("output_changed",)


def _sample_hashes(paths):
    hashes = {}
    metadata = {}
    for path in paths:
        for result in _read_jsonl(path):
            benchmark = _benchmark_name(result, path)
            question_id = str(result["question_id"])
            choices = result.get("choices", [])
            first_hash = None
            if choices and choices[0].get("output_hashes"):
                first_hash = str(choices[0]["output_hashes"][0])
            key = (benchmark, question_id)
            hashes[key] = first_hash
            metadata[key] = result.get("wrong_image_pair", {})
    return hashes, metadata


def build_pairs(original_paths, wrong_paths):
    original_hashes, _ = _sample_hashes(original_paths)
    wrong_hashes, wrong_metadata = _sample_hashes(wrong_paths)
    keys = sorted(set(original_hashes) & set(wrong_hashes))
    pairs = []
    for benchmark, question_id in keys:
        original_hash = original_hashes.get((benchmark, question_id))
        wrong_hash = wrong_hashes.get((benchmark, question_id))
        pairs.append(
            {
                "benchmark": benchmark,
                "question_id": question_id,
                "output_changed": bool(
                    original_hash is not None
                    and wrong_hash is not None
                    and original_hash != wrong_hash
                ),
                "same_category_pair": not bool(
                    wrong_metadata.get((benchmark, question_id), {}).get(
                        "used_category_fallback", False
                    )
                ),
            }
        )
    return pairs


def _macro_means(pairs):
    by_benchmark = defaultdict(list)
    for row in pairs:
        by_benchmark[row["benchmark"]].append(row)
    benchmark_values = {}
    for benchmark, rows in sorted(by_benchmark.items()):
        values = np.asarray(
            [
                [float(row[metric]) for metric in METRICS]
                for row in rows
            ],
            dtype=np.float64,
        )
        benchmark_values[benchmark] = values
    macro = np.mean(
        [values.mean(axis=0) for values in benchmark_values.values()], axis=0
    )
    return benchmark_values, macro


def _clustered_bootstrap(benchmark_values, resamples, seed):
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(METRICS)), dtype=np.float32)
    for draw_index in range(resamples):
        benchmark_means = []
        for values in benchmark_values.values():
            indices = rng.integers(0, len(values), size=len(values))
            benchmark_means.append(values[indices].mean(axis=0))
        draws[draw_index] = np.mean(benchmark_means, axis=0)
    return draws


def summarize_pairs(pairs, bootstrap_resamples=50000, seed=42):
    if not pairs:
        raise ValueError("no matched original/wrong-image samples")
    benchmark_values, macro = _macro_means(pairs)
    draws = _clustered_bootstrap(
        benchmark_values, bootstrap_resamples, seed
    )
    metrics = {}
    for index, metric in enumerate(METRICS):
        metrics[metric] = {
            "estimate": float(macro[index]),
            "clustered_bootstrap_95_ci": [
                float(np.percentile(draws[:, index], 2.5)),
                float(np.percentile(draws[:, index], 97.5)),
            ],
        }
    by_benchmark = {}
    for benchmark, values in benchmark_values.items():
        by_benchmark[benchmark] = {
            "num_pairs": int(len(values)),
            **{
                metric: float(values[:, index].mean())
                for index, metric in enumerate(METRICS)
            },
        }
    return {
        "num_pairs": len(pairs),
        "num_benchmarks": len(benchmark_values),
        "benchmarks": sorted(benchmark_values),
        "pairing": "distinct wrong image; same category unless unavailable",
        "interpretation": (
            "Target-only output-change rate is a positive control for visual "
            "responsiveness. The original and wrong-image runs differ only "
            "in image pixels; no recycling state or shadow trajectory enters."
        ),
        "bootstrap": {
            "unit": "sample/image pair",
            "stratification": "within benchmark",
            "macro_averaging": "equal benchmark weight",
            "resamples": bootstrap_resamples,
            "seed": seed,
        },
        "metrics": metrics,
        "same_category_pair_ratio": float(
            np.mean([row["same_category_pair"] for row in pairs])
        ),
        "by_benchmark": by_benchmark,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-results-roots", type=Path, nargs="+", required=True
    )
    parser.add_argument("--wrong-results-root", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    original_paths = discover_result_paths(args.original_results_roots, "target")
    wrong_paths = discover_result_paths(
        [args.wrong_results_root], "target"
    )
    if not original_paths or not wrong_paths:
        parser.error("missing original or wrong-image policy result files")
    pairs = build_pairs(original_paths, wrong_paths)
    summary = summarize_pairs(
        pairs, args.bootstrap_resamples, args.seed
    )
    summary["generation_policy"] = "target-only"
    summary["comparison_method_policy"] = args.policy
    summary["original_paths"] = [str(path) for path in original_paths]
    summary["wrong_paths"] = [str(path) for path in wrong_paths]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "paired_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
