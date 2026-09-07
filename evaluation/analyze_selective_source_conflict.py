"""Root-aligned analysis for states where U and G/C proposals conflict.

Eligibility is outcome independent: both sources are truncated to the same
root-candidate budget, that budget must be at least four, and the truncated
sets may overlap by at most two tokens.  The response is the G/C-minus-U
target-token hit difference for the same root whose true pixel-occlusion JSD
defines visual sensitivity.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_selective_reuse import (
    discover_result_paths,
    load_selective_records,
)


MAX_CANDIDATE_BUDGET = 8
MIN_MATCHED_CANDIDATE_BUDGET = 4
MAX_CANDIDATE_OVERLAP = 2
LOW_DECILES = (1, 2)
HIGH_DECILES = (9, 10)


def _average_tie_percentiles(values: np.ndarray) -> np.ndarray:
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


def assign_deciles(records: Sequence[dict]) -> None:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[row["benchmark"]].append(index)
    for indices in grouped.values():
        values = np.asarray(
            [float(records[index]["visual_probe_jsd"]) for index in indices],
            dtype=np.float64,
        )
        percentiles = _average_tie_percentiles(values)
        for index, percentile in zip(indices, percentiles.tolist()):
            records[index]["visual_sensitivity_percentile"] = percentile
            records[index]["visual_sensitivity_decile"] = min(
                int(percentile * 10.0) + 1, 10
            )


def validate_and_filter(records: Sequence[dict]) -> tuple[list[dict], dict]:
    audit = defaultdict(int)
    eligible = []
    for row in records:
        audit["input_states"] += 1
        if not row.get("u_available") or not row.get("gc_available"):
            audit["one_or_both_sources_unavailable"] += 1
            continue
        u = [int(token) for token in row.get("u_root_candidate_token_ids", [])]
        gc = [int(token) for token in row.get("gc_root_candidate_token_ids", [])]
        matched_budget = min(len(u), len(gc), MAX_CANDIDATE_BUDGET)
        if matched_budget < MIN_MATCHED_CANDIDATE_BUDGET:
            audit["matched_candidate_budget_lt_4"] += 1
            continue
        u = u[:matched_budget]
        gc = gc[:matched_budget]
        if len(set(u)) != matched_budget or len(set(gc)) != matched_budget:
            audit["duplicate_candidate_row"] += 1
            continue
        overlap = len(set(u) & set(gc))
        if overlap > MAX_CANDIDATE_OVERLAP:
            audit["non_conflict_overlap_gt_2"] += 1
            continue
        copied = dict(row)
        copied["matched_root_candidate_budget"] = matched_budget
        copied["candidate_set_overlap"] = overlap
        target = int(row["target_token_id"])
        copied["u_root_hit"] = int(target in u)
        copied["gc_root_hit"] = int(target in gc)
        copied["gc_minus_u_root_hit"] = (
            copied["gc_root_hit"] - copied["u_root_hit"]
        )
        eligible.append(copied)
    audit["eligible_states"] = len(eligible)
    audit["eligible_ratio"] = len(eligible) / len(records) if records else 0.0
    return eligible, dict(audit)


def _tail_stats(rows: Sequence[dict]) -> tuple[float, float, float] | None:
    low = [
        row
        for row in rows
        if int(row["visual_sensitivity_decile"]) in LOW_DECILES
    ]
    high = [
        row
        for row in rows
        if int(row["visual_sensitivity_decile"]) in HIGH_DECILES
    ]
    if not low or not high:
        return None
    low_delta = float(np.mean([row["gc_minus_u_root_hit"] for row in low]))
    high_delta = float(np.mean([row["gc_minus_u_root_hit"] for row in high]))
    return low_delta, high_delta, high_delta - low_delta


def point_estimates(rows: Sequence[dict]) -> tuple[np.ndarray, dict]:
    values = []
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        stats = _tail_stats(selected)
        if stats is None:
            continue
        low = [
            row
            for row in selected
            if int(row["visual_sensitivity_decile"]) in LOW_DECILES
        ]
        high = [
            row
            for row in selected
            if int(row["visual_sensitivity_decile"]) in HIGH_DECILES
        ]
        by_benchmark[benchmark] = {
            "eligible_states": len(selected),
            "low_states": len(low),
            "high_states": len(high),
            "low_u_recall": float(np.mean([row["u_root_hit"] for row in low])),
            "low_gc_recall": float(np.mean([row["gc_root_hit"] for row in low])),
            "low_gc_minus_u": stats[0],
            "high_u_recall": float(np.mean([row["u_root_hit"] for row in high])),
            "high_gc_recall": float(np.mean([row["gc_root_hit"] for row in high])),
            "high_gc_minus_u": stats[1],
            "high_minus_low": stats[2],
        }
        values.append(stats)
    if not values:
        raise ValueError("no benchmark has eligible states in both visual tails")
    return np.mean(np.asarray(values, dtype=np.float64), axis=0), by_benchmark


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int
) -> np.ndarray:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    draws = np.full((int(resamples), 3), np.nan, dtype=np.float64)
    for draw_index in range(int(resamples)):
        values = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            stats = _tail_stats(sampled)
            if stats is not None:
                values.append(stats)
        if values:
            draws[draw_index] = np.mean(
                np.asarray(values, dtype=np.float64), axis=0
            )
    return draws


def _ci(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def decile_curves(rows: Sequence[dict]) -> list[dict]:
    benchmarks = sorted({row["benchmark"] for row in rows})
    output = []
    for decile in range(1, 11):
        values = []
        states = 0
        for benchmark in benchmarks:
            selected = [
                row
                for row in rows
                if row["benchmark"] == benchmark
                and int(row["visual_sensitivity_decile"]) == decile
            ]
            if not selected:
                continue
            states += len(selected)
            values.append(
                (
                    float(np.mean([row["u_root_hit"] for row in selected])),
                    float(np.mean([row["gc_root_hit"] for row in selected])),
                    float(
                        np.mean([row["gc_minus_u_root_hit"] for row in selected])
                    ),
                )
            )
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "decile": decile,
                "eligible_states": states,
                "u_recall": float(array[:, 0].mean()) if len(array) else None,
                "gc_recall": float(array[:, 1].mean()) if len(array) else None,
                "gc_minus_u": float(array[:, 2].mean()) if len(array) else None,
            }
        )
    return output


def plot(payload: dict, output_stem: Path) -> None:
    rows = payload["macro_deciles"]
    x = np.arange(1, 11)
    u = np.asarray([row["u_recall"] for row in rows], dtype=float)
    gc = np.asarray([row["gc_recall"] for row in rows], dtype=float)
    delta = np.asarray([row["gc_minus_u"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45), constrained_layout=True)
    axes[0].plot(x, u, marker="o", color="#E67E22", label="U")
    axes[0].plot(x, gc, marker="o", color="#2468B4", label="G/C")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Root target-token top-8 recall")
    axes[0].set_title("(a) Conflicting source candidates")
    axes[0].legend(frameon=False)
    axes[1].plot(x, delta, marker="o", color="#7A3E9D")
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("G/C minus U recall")
    axes[1].set_title("(b) Preferred source changes")
    for axis in axes:
        axis.set_xlabel("True-occlusion visual-sensitivity decile")
        axis.set_xticks(x)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    role = payload["analysis_role"].upper()
    fig.suptitle(
        f"{role} · matched budget ≥ 4, overlap ≤ 2 · {payload['num_benchmarks']} benchmarks",
        fontsize=10.5,
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-role",
        choices=("development", "validation", "confirmatory"),
        required=True,
    )
    parser.add_argument(
        "--analysis-split",
        choices=("all", "discovery", "heldout"),
        default="all",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=161803)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        raise FileNotFoundError("no candidate result JSONL files found")
    records = load_selective_records(paths)
    if args.analysis_split != "all":
        records = [
            row for row in records if row["analysis_split"] == args.analysis_split
        ]
    assign_deciles(records)
    eligible, audit = validate_and_filter(records)
    if not eligible:
        raise ValueError("no candidate-conflict states passed the frozen rule")
    point, by_benchmark = point_estimates(eligible)
    draws = clustered_bootstrap(
        eligible, resamples=args.bootstrap_resamples, seed=args.seed
    )
    low_ci, high_ci, interaction_ci = (
        _ci(draws[:, 0]),
        _ci(draws[:, 1]),
        _ci(draws[:, 2]),
    )
    positive_benchmarks = sum(
        row["high_minus_low"] > 0.0 for row in by_benchmark.values()
    )
    crossover = bool(point[0] < 0.0 < point[1])
    if args.analysis_role == "development":
        passed = bool(crossover and point[2] > 0.0 and positive_benchmarks >= 5)
        verdict = "development_rule_selected" if passed else "development_no_go"
    elif args.analysis_role == "validation":
        passed = bool(
            len(by_benchmark) >= 6
            and crossover
            and point[2] > 0.0
            and positive_benchmarks >= 5
        )
        verdict = "validation_go" if passed else "validation_no_go"
    else:
        passed = bool(
            len(by_benchmark) == 8
            and low_ci[1] is not None
            and low_ci[1] < 0.0
            and high_ci[0] is not None
            and high_ci[0] > 0.0
            and interaction_ci[0] is not None
            and interaction_ci[0] > 0.0
            and positive_benchmarks >= 6
        )
        verdict = "selective_source_conflict_supported" if passed else "not_supported"

    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": args.analysis_split,
        "input_paths": [str(path) for path in paths],
        "maximum_candidate_budget_per_source": MAX_CANDIDATE_BUDGET,
        "minimum_matched_candidate_budget": MIN_MATCHED_CANDIDATE_BUDGET,
        "maximum_candidate_set_overlap": MAX_CANDIDATE_OVERLAP,
        "eligibility_is_outcome_independent": True,
        "num_input_states": len(records),
        "num_eligible_states": len(eligible),
        "num_benchmarks": len(by_benchmark),
        "benchmarks": sorted(by_benchmark),
        "excluded_benchmarks": ["MME"],
        "audit": audit,
        "visual_sensitivity": "same-root JSD over full image and four true pixel-occluded views",
        "outcome": "G/C-minus-U target-token top-8 hit at the same root",
        "tail_definition": "within-benchmark bottom/top two visual-sensitivity deciles",
        "evidence_gate": {
            "verdict": verdict,
            "pass": passed,
            "sign_crossover": crossover,
            "low_visual_gc_minus_u": float(point[0]),
            "low_visual_gc_minus_u_95_ci": low_ci,
            "high_visual_gc_minus_u": float(point[1]),
            "high_visual_gc_minus_u_95_ci": high_ci,
            "high_minus_low": float(point[2]),
            "high_minus_low_95_ci": interaction_ci,
            "positive_interaction_benchmarks": positive_benchmarks,
            "confirmatory_rule": "all 8 benchmarks; low CI < 0; high CI > 0; interaction CI > 0; at least 6/8 benchmark interactions positive",
        },
        "bootstrap": {
            "unit": "image cluster",
            "stratification": "benchmark",
            "macro_averaging": "equal benchmark weight",
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "macro_deciles": decile_curves(eligible),
        "by_benchmark": by_benchmark,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "eligible_records.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in eligible:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot(payload, args.output_dir / "selective_source_conflict")
    print(json.dumps(payload["evidence_gate"], indent=2))


if __name__ == "__main__":
    main()
