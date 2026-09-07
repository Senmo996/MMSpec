"""Development-only search for causal, lag-aware visual-regime scores.

A visual observation can determine an anchor token and then influence several
language-like continuation tokens.  Instantaneous occlusion scores label only
the anchor as visual, even though request-local transitions may become useful
immediately after it.  This script tests a small, predeclared family of causal
rolling/decayed scores on development data and replays the selected rule on an
already-inspected held-out split.  The replay is retrospective, never
confirmatory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)


TAIL_FRACTION = 0.20
MIN_ROOT_BUDGET = 2
MAX_ROOT_BUDGET = 8
MAX_OVERLAP_FRACTION = 0.25
MIN_TAIL_STATES_PER_BENCHMARK = 3
MIN_VALID_BENCHMARKS = 6
WINDOWS = (2, 4, 8)
DECAYS = (0.50, 0.75, 0.90)


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


def prepare(rows: list[dict]) -> list[str]:
    grouped = defaultdict(list)
    for row in rows:
        drop = max(float(row["visual_probe_max_target_logprob_drop"]), 0.0)
        surprisal = max(-float(row["visual_probe_full_target_logprob"]), 0.0)
        row["visual_target_drop_fraction"] = drop / (drop + surprisal + 1e-8)
        u = [int(token) for token in row.get("u_root_candidate_token_ids", [])][
            :MAX_ROOT_BUDGET
        ]
        gc = [int(token) for token in row.get("gc_root_candidate_token_ids", [])][
            :MAX_ROOT_BUDGET
        ]
        budget = min(len(u), len(gc))
        row["lag_rule_eligible"] = False
        if row["u_available"] and row["gc_available"] and budget >= MIN_ROOT_BUDGET:
            u, gc = u[:budget], gc[:budget]
            if len(set(u)) == budget and len(set(gc)) == budget:
                overlap_fraction = len(set(u) & set(gc)) / budget
                if overlap_fraction <= MAX_OVERLAP_FRACTION:
                    target = int(row["target_token_id"])
                    row["lag_rule_eligible"] = True
                    row["lag_root_delta"] = int(target in gc) - int(target in u)
        grouped[
            (
                row["analysis_split"],
                row["benchmark"],
                row["question_id"],
                row.get("choice_index", 0),
                row.get("turn_index", 0),
            )
        ].append(row)

    metric_names = []
    for window in WINDOWS:
        metric_names.extend((f"recent_max_{window}", f"past_max_{window}"))
    metric_names.extend(f"causal_decay_{decay:.2f}" for decay in DECAYS)
    for trajectory in grouped.values():
        trajectory.sort(key=lambda row: (row["output_position"], row["iteration"]))
        values = np.asarray(
            [row["visual_target_drop_fraction"] for row in trajectory], dtype=float
        )
        for index, row in enumerate(trajectory):
            for window in WINDOWS:
                row[f"recent_max_{window}"] = float(
                    values[max(0, index - window + 1) : index + 1].max()
                )
                past = values[max(0, index - window) : index]
                row[f"past_max_{window}"] = float(
                    past.max() if len(past) else values[index]
                )
        for decay in DECAYS:
            carried = 0.0
            name = f"causal_decay_{decay:.2f}"
            for row, value in zip(trajectory, values.tolist()):
                carried = max(float(value), float(decay) * carried)
                row[name] = carried

    percentile_groups = defaultdict(list)
    for index, row in enumerate(rows):
        percentile_groups[(row["analysis_split"], row["benchmark"])].append(index)
    for metric in metric_names:
        for indices in percentile_groups.values():
            percentiles = average_tie_percentiles(
                np.asarray([rows[index][metric] for index in indices], dtype=float)
            )
            for index, percentile in zip(indices, percentiles.tolist()):
                rows[index][f"percentile::{metric}"] = float(percentile)
    return metric_names


def evaluate(rows: list[dict], metric: str, split: str) -> dict | None:
    split_rows = [row for row in rows if row["analysis_split"] == split]
    pairs = []
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in split_rows}):
        benchmark_rows = [row for row in split_rows if row["benchmark"] == benchmark]
        low = [
            row
            for row in benchmark_rows
            if row["lag_rule_eligible"]
            and row[f"percentile::{metric}"] < TAIL_FRACTION
        ]
        high = [
            row
            for row in benchmark_rows
            if row["lag_rule_eligible"]
            and row[f"percentile::{metric}"] >= 1.0 - TAIL_FRACTION
        ]
        if (
            len(low) < MIN_TAIL_STATES_PER_BENCHMARK
            or len(high) < MIN_TAIL_STATES_PER_BENCHMARK
        ):
            continue
        low_delta = float(np.mean([row["lag_root_delta"] for row in low]))
        high_delta = float(np.mean([row["lag_root_delta"] for row in high]))
        interaction = high_delta - low_delta
        pairs.append((low_delta, high_delta, interaction))
        by_benchmark[benchmark] = {
            "low_states": len(low),
            "high_states": len(high),
            "low_delta": low_delta,
            "high_delta": high_delta,
            "interaction": interaction,
        }
    if len(pairs) < MIN_VALID_BENCHMARKS:
        return None
    array = np.asarray(pairs, dtype=float)
    low, high, interaction = array.mean(axis=0).tolist()
    positive_interactions = int(np.sum(array[:, 2] > 0.0))
    crossover_benchmarks = int(np.sum((array[:, 0] < 0.0) & (array[:, 1] > 0.0)))
    return {
        "split": split,
        "metric": metric,
        "num_valid_benchmarks": len(pairs),
        "low_delta": low,
        "high_delta": high,
        "interaction": interaction,
        "sign_crossover": bool(low < 0.0 < high),
        "positive_interaction_benchmarks": positive_interactions,
        "crossover_benchmarks": crossover_benchmarks,
        "by_benchmark": by_benchmark,
        "development_rank_score": float(
            (1000.0 if low < 0.0 < high else 0.0)
            + 30.0 * positive_interactions
            + 20.0 * crossover_benchmarks
            + 10.0 * min(max(-low, 0.0), max(high, 0.0))
            + interaction
        ),
    }


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
    metrics = prepare(rows)
    candidates = [result for metric in metrics if (result := evaluate(rows, metric, "discovery"))]
    if not candidates:
        parser.error("no lag-aware metric has enough development states")
    candidates.sort(key=lambda row: row["development_rank_score"], reverse=True)
    selected = candidates[0]
    replay = evaluate(rows, selected["metric"], "heldout")
    stable = bool(
        selected["sign_crossover"]
        and replay is not None
        and replay["sign_crossover"]
        and selected["positive_interaction_benchmarks"] >= 5
        and replay["positive_interaction_benchmarks"] >= 5
    )
    frozen_rule = {
        "rule_id": f"lagged-{selected['metric']}-tail20-root2-overlap25",
        "visual_metric": selected["metric"],
        "causal": True,
        "tail_fraction": TAIL_FRACTION,
        "minimum_matched_root_budget": MIN_ROOT_BUDGET,
        "maximum_root_budget": MAX_ROOT_BUDGET,
        "maximum_candidate_overlap_fraction": MAX_OVERLAP_FRACTION,
        "outcome": "equal-root-budget G/C-minus-U target-token hit",
    }
    payload = {
        "schema_version": 1,
        "analysis_role": "development_search_with_retrospective_replay",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "The heldout split was previously inspected. This replay is only "
            "retrospective stability evidence and requires a new disjoint run."
        ),
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "candidate_metrics": metrics,
        "fixed_eligibility": frozen_rule,
        "selected_development_result": selected,
        "heldout_retrospective": replay,
        "stable_desired_pattern": stable,
        "decision": "freeze_for_new_validation" if stable else "no_go",
        "frozen_rule": frozen_rule if stable else None,
        "all_development_results": candidates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if stable:
        (args.output_dir / "frozen_rule.json").write_text(
            json.dumps(frozen_rule, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "selected_metric": selected["metric"],
                "development": {
                    key: selected[key]
                    for key in ("low_delta", "high_delta", "interaction")
                },
                "heldout_retrospective": (
                    {
                        key: replay[key]
                        for key in ("low_delta", "high_delta", "interaction")
                    }
                    if replay
                    else None
                ),
                "decision": payload["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
