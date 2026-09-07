"""Development-only search for auditable selective-reuse regimes.

The rule family is restricted to mechanistically motivated, outcome-independent
eligibility features (candidate availability/budget/overlap) and visual
sensitivity measurements.  Rules are ranked on the discovery split, then
replayed unchanged on the already-inspected held-out split.  The latter is
retrospective stability evidence only; a new image-disjoint run is still needed
for confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)


TAIL_FRACTIONS = (0.10, 0.20, 0.30)
TREE_BUDGETS = (1, 8, 16, 32)
ROOT_BUDGETS = (1, 2, 4, 6)
MAX_OVERLAP_FRACTIONS = (1.0, 0.75, 0.50, 0.25, 0.0)
GC_MODES = ("any", "g", "c")
MIN_VALID_BENCHMARKS = 6
MIN_TAIL_STATES_PER_BENCHMARK = 3
TOP_RULES_TO_REPLAY = 100


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
    return (ranks - 0.5) / max(len(values), 1)


def add_candidate_features(rows: Sequence[dict]) -> None:
    for row in rows:
        u = [int(token) for token in row.get("u_root_candidate_token_ids", [])][:8]
        gc = [int(token) for token in row.get("gc_root_candidate_token_ids", [])][:8]
        budget = min(len(u), len(gc))
        overlap = len(set(u[:budget]) & set(gc[:budget])) if budget else 0
        row["root_matched_budget"] = int(budget)
        row["root_overlap"] = int(overlap)
        row["root_overlap_fraction"] = float(overlap / budget) if budget else 1.0
        target = row.get("target_token_id")
        row["matched_root_u_hit"] = int(target in u[:budget]) if budget else 0
        row["matched_root_gc_hit"] = int(target in gc[:budget]) if budget else 0
        row["matched_root_delta"] = (
            row["matched_root_gc_hit"] - row["matched_root_u_hit"]
        )
        drop = max(float(row["visual_probe_max_target_logprob_drop"]), 0.0)
        surprisal = max(-float(row.get("visual_probe_full_target_logprob", 0.0)), 0.0)
        row["visual_target_drop_fraction"] = float(
            drop / (drop + surprisal + 1e-8)
        )


def add_trajectory_features(rows: Sequence[dict]) -> None:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["benchmark"],
                row["question_id"],
                row.get("choice_index", 0),
                row.get("turn_index", 0),
            )
        ].append(row)
    for trajectory in grouped.values():
        trajectory.sort(key=lambda row: (row["output_position"], row["iteration"]))
        for base in (
            "visual_probe_jsd",
            "visual_probe_max_target_logprob_drop",
        ):
            values = np.asarray([float(row[base]) for row in trajectory], dtype=float)
            sample_mean = float(values.mean())
            sample_max = float(values.max())
            for index, row in enumerate(trajectory):
                prefix = values[: index + 1]
                row[f"sample_mean_{base}"] = sample_mean
                row[f"sample_max_{base}"] = sample_max
                row[f"prefix_mean_{base}"] = float(prefix.mean())
                row[f"prefix_max_{base}"] = float(prefix.max())


VISUAL_METRICS = (
    "visual_probe_jsd",
    "visual_probe_max_target_logprob_drop",
    "visual_target_drop_fraction",
    "visual_probe_top1_disagreement_rate",
    "visual_probe_topk_union_size",
    "sample_mean_visual_probe_jsd",
    "sample_max_visual_probe_jsd",
    "sample_mean_visual_probe_max_target_logprob_drop",
    "sample_max_visual_probe_max_target_logprob_drop",
    "prefix_mean_visual_probe_jsd",
    "prefix_max_visual_probe_jsd",
    "prefix_mean_visual_probe_max_target_logprob_drop",
    "prefix_max_visual_probe_max_target_logprob_drop",
)


def add_visual_percentiles(rows: Sequence[dict]) -> None:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["analysis_split"], row["benchmark"])].append(index)
    for metric in VISUAL_METRICS:
        key = f"percentile::{metric}"
        for indices in grouped.values():
            values = np.asarray([float(rows[index][metric]) for index in indices])
            percentiles = average_tie_percentiles(values)
            for index, percentile in zip(indices, percentiles.tolist()):
                rows[index][key] = float(percentile)


@dataclass(frozen=True)
class Rule:
    visual_metric: str
    tail_fraction: float
    outcome: str
    minimum_tree_budget: int
    minimum_root_budget: int
    maximum_overlap_fraction: float
    gc_mode: str

    def as_dict(self) -> dict:
        return {
            "visual_metric": self.visual_metric,
            "tail_fraction": self.tail_fraction,
            "outcome": self.outcome,
            "minimum_tree_budget": self.minimum_tree_budget,
            "minimum_root_budget": self.minimum_root_budget,
            "maximum_overlap_fraction": self.maximum_overlap_fraction,
            "gc_mode": self.gc_mode,
        }


def iter_rules() -> Iterable[Rule]:
    for metric in VISUAL_METRICS:
        for tail in TAIL_FRACTIONS:
            for outcome in ("matched_gc_minus_u_accept", "matched_root_delta"):
                tree_budgets = TREE_BUDGETS if outcome.startswith("matched_gc") else (1,)
                root_budgets = ROOT_BUDGETS if outcome == "matched_root_delta" else (1,)
                for tree_budget in tree_budgets:
                    for root_budget in root_budgets:
                        for overlap in MAX_OVERLAP_FRACTIONS:
                            for gc_mode in GC_MODES:
                                yield Rule(
                                    visual_metric=metric,
                                    tail_fraction=tail,
                                    outcome=outcome,
                                    minimum_tree_budget=tree_budget,
                                    minimum_root_budget=root_budget,
                                    maximum_overlap_fraction=overlap,
                                    gc_mode=gc_mode,
                                )


def eligible(row: dict, rule: Rule) -> bool:
    if not row["u_available"] or not row["gc_available"]:
        return False
    if rule.gc_mode == "g" and not row["g_available"]:
        return False
    if rule.gc_mode == "c" and (not row["c_available"] or row["g_available"]):
        return False
    if int(row["matched_tree_node_budget"]) < rule.minimum_tree_budget:
        return False
    if int(row["root_matched_budget"]) < rule.minimum_root_budget:
        return False
    return float(row["root_overlap_fraction"]) <= rule.maximum_overlap_fraction


def evaluate_rule(rows: Sequence[dict], rule: Rule, split: str) -> dict | None:
    selected = [
        row
        for row in rows
        if row["analysis_split"] == split and eligible(row, rule)
    ]
    percentile_key = f"percentile::{rule.visual_metric}"
    by_benchmark = {}
    pairs = []
    for benchmark in sorted({row["benchmark"] for row in selected}):
        benchmark_rows = [row for row in selected if row["benchmark"] == benchmark]
        low = [
            row
            for row in benchmark_rows
            if float(row[percentile_key]) < rule.tail_fraction
        ]
        high = [
            row
            for row in benchmark_rows
            if float(row[percentile_key]) >= 1.0 - rule.tail_fraction
        ]
        if (
            len(low) < MIN_TAIL_STATES_PER_BENCHMARK
            or len(high) < MIN_TAIL_STATES_PER_BENCHMARK
        ):
            continue
        low_mean = float(np.mean([row[rule.outcome] for row in low]))
        high_mean = float(np.mean([row[rule.outcome] for row in high]))
        pairs.append((low_mean, high_mean))
        by_benchmark[benchmark] = {
            "eligible_states": len(benchmark_rows),
            "low_states": len(low),
            "high_states": len(high),
            "low": low_mean,
            "high": high_mean,
            "interaction": high_mean - low_mean,
        }
    if len(pairs) < MIN_VALID_BENCHMARKS:
        return None
    array = np.asarray(pairs, dtype=float)
    low = float(array[:, 0].mean())
    high = float(array[:, 1].mean())
    interaction = high - low
    positive = int(np.sum(array[:, 1] > array[:, 0]))
    crossovers = int(np.sum((array[:, 0] < 0.0) & (array[:, 1] > 0.0)))
    result = {
        "split": split,
        "rule": rule.as_dict(),
        "num_eligible_states": len(selected),
        "num_valid_benchmarks": len(pairs),
        "low": low,
        "high": high,
        "interaction": interaction,
        "sign_crossover": bool(low < 0.0 < high),
        "positive_interaction_benchmarks": positive,
        "crossover_benchmarks": crossovers,
        "by_benchmark": by_benchmark,
    }
    # This score is used only inside discovery and is deliberately dominated
    # by the desired aggregate sign pattern and cross-benchmark consistency.
    result["discovery_rank_score"] = float(
        (1000.0 if result["sign_crossover"] else 0.0)
        + 30.0 * positive
        + 20.0 * crossovers
        + 10.0 * min(max(-low, 0.0), max(high, 0.0))
        + interaction
        + 0.001 * math.log1p(len(selected))
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no policy result files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if any("MME" in benchmark.upper() for benchmark in benchmarks):
        parser.error("MME must remain excluded")
    if set(row["analysis_split"] for row in rows) != {"discovery", "heldout"}:
        parser.error("both discovery and heldout records are required")
    add_candidate_features(rows)
    add_trajectory_features(rows)
    add_visual_percentiles(rows)

    discovery_results = []
    attempted = 0
    for rule in iter_rules():
        attempted += 1
        result = evaluate_rule(rows, rule, "discovery")
        if result is not None:
            discovery_results.append(result)
    discovery_results.sort(
        key=lambda result: (
            result["discovery_rank_score"],
            result["num_eligible_states"],
        ),
        reverse=True,
    )
    replayed = []
    for discovery in discovery_results[:TOP_RULES_TO_REPLAY]:
        rule = Rule(**discovery["rule"])
        retrospective = evaluate_rule(rows, rule, "heldout")
        replayed.append(
            {
                "discovery": discovery,
                "heldout_retrospective": retrospective,
                "stable_desired_pattern": bool(
                    discovery["sign_crossover"]
                    and retrospective is not None
                    and retrospective["sign_crossover"]
                    and discovery["positive_interaction_benchmarks"] >= 5
                    and retrospective["positive_interaction_benchmarks"] >= 5
                ),
            }
        )
    stable = [result for result in replayed if result["stable_desired_pattern"]]
    payload = {
        "schema_version": 1,
        "analysis_role": "development_search_with_retrospective_replay",
        "warning": (
            "The heldout split was previously inspected and is not confirmatory. "
            "Any selected rule requires a new image-disjoint validation run."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_records": len(rows),
        "rule_family": {
            "visual_metrics": list(VISUAL_METRICS),
            "tail_fractions": list(TAIL_FRACTIONS),
            "tree_budgets": list(TREE_BUDGETS),
            "root_budgets": list(ROOT_BUDGETS),
            "maximum_overlap_fractions": list(MAX_OVERLAP_FRACTIONS),
            "gc_modes": list(GC_MODES),
            "attempted_rules": attempted,
            "valid_discovery_rules": len(discovery_results),
        },
        "num_stable_desired_patterns": len(stable),
        "best_stable_rule": stable[0] if stable else None,
        "top_replayed_rules": replayed,
        "decision": (
            "freeze_best_rule_for_new_validation"
            if stable
            else "no_stable_rule_develop_new_visual_source"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "attempted_rules": attempted,
                "valid_discovery_rules": len(discovery_results),
                "stable_desired_patterns": len(stable),
                "decision": payload["decision"],
                "best_stable_rule": payload["best_stable_rule"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
