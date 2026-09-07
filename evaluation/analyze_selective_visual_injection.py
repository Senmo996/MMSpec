"""Analyze a frozen, equal-budget visual injection into the U proposal row.

This diagnostic is deliberately root aligned: the visual covariate and the
candidate-hit outcome refer to the exact same next token.  The compared rows
have eight candidates each.  ``U+V`` preserves U's first seven candidates and
uses one prompt-native ``visual_max`` candidate for the final slot.

Development runs are hypothesis-generating only.  Confirmatory use requires a
new image-cluster-disjoint sample after the one-slot rule has been frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_selective_reuse import (
    discover_result_paths,
    load_selective_records,
)


SOURCE = "prompt_visual_max"
CANDIDATE_BUDGET = 8
VISUAL_SLOTS = 1
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
    eligible = []
    audit = defaultdict(int)
    for row in records:
        audit["input_states"] += 1
        if not row.get("u_available"):
            audit["u_unavailable"] += 1
            continue
        if not row.get("v_available") or row.get("v_source") != SOURCE:
            audit["v_unavailable_or_wrong_source"] += 1
            continue
        if int(row.get("uv_visual_slots", 0)) != VISUAL_SLOTS:
            audit["wrong_visual_slot_count"] += 1
            continue
        if int(row.get("uv_candidate_budget", 0)) != CANDIDATE_BUDGET:
            audit["non_eight_candidate_budget"] += 1
            continue
        u = [int(token) for token in row.get("u_root_candidate_token_ids", [])]
        uv = [int(token) for token in row.get("uv_root_candidate_token_ids", [])]
        if len(u) != CANDIDATE_BUDGET or len(uv) != CANDIDATE_BUDGET:
            audit["candidate_length_mismatch"] += 1
            continue
        if len(set(u)) != len(u) or len(set(uv)) != len(uv):
            audit["duplicate_candidates"] += 1
            continue
        if uv[: CANDIDATE_BUDGET - VISUAL_SLOTS] != u[
            : CANDIDATE_BUDGET - VISUAL_SLOTS
        ]:
            audit["anchor_prefix_changed"] += 1
            continue
        copied = dict(row)
        copied["u_hit"] = int(bool(row["u_top8_hit"]))
        copied["uv_hit"] = int(bool(row["uv_top8_hit"]))
        copied["visual_injection_gain"] = copied["uv_hit"] - copied["u_hit"]
        eligible.append(copied)
    audit["eligible_states"] = len(eligible)
    audit["eligible_ratio"] = len(eligible) / len(records) if records else 0.0
    audit["matched_candidate_count_ratio"] = (
        1.0 if eligible else 0.0
    )
    audit["changed_row_ratio"] = (
        float(
            np.mean(
                [
                    row["u_root_candidate_token_ids"]
                    != row["uv_root_candidate_token_ids"]
                    for row in eligible
                ]
            )
        )
        if eligible
        else 0.0
    )
    return eligible, dict(audit)


def _mean(rows: Iterable[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else None


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
    low_gain = float(np.mean([row["visual_injection_gain"] for row in low]))
    high_gain = float(np.mean([row["visual_injection_gain"] for row in high]))
    return low_gain, high_gain, high_gain - low_gain


def _macro_tail_stats(rows: Sequence[dict]) -> tuple[np.ndarray, dict]:
    by_benchmark = {}
    values = []
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
            "low_u_recall": _mean(low, "u_hit"),
            "low_uv_recall": _mean(low, "uv_hit"),
            "low_visual_injection_gain": stats[0],
            "high_u_recall": _mean(high, "u_hit"),
            "high_uv_recall": _mean(high, "uv_hit"),
            "high_visual_injection_gain": stats[1],
            "high_minus_low_gain": stats[2],
        }
        values.append(stats)
    if not values:
        raise ValueError("no benchmark has both low- and high-visual eligible states")
    return np.mean(np.asarray(values, dtype=np.float64), axis=0), by_benchmark


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int
) -> np.ndarray:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(resamples), 3), dtype=np.float64)
    benchmarks = sorted(grouped)
    for draw_index in range(int(resamples)):
        benchmark_values = []
        for benchmark in benchmarks:
            clusters = sorted(grouped[benchmark])
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            sample_rows = [
                row
                for cluster_index in sampled.tolist()
                for row in grouped[benchmark][clusters[cluster_index]]
            ]
            stats = _tail_stats(sample_rows)
            if stats is not None:
                benchmark_values.append(stats)
        draws[draw_index] = (
            np.mean(np.asarray(benchmark_values, dtype=np.float64), axis=0)
            if benchmark_values
            else np.asarray([np.nan, np.nan, np.nan])
        )
    return draws


def _ci(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    ]


def decile_curves(rows: Sequence[dict]) -> list[dict]:
    output = []
    benchmarks = sorted({row["benchmark"] for row in rows})
    for decile in range(1, 11):
        per_benchmark = []
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
            per_benchmark.append(
                (
                    _mean(selected, "u_hit"),
                    _mean(selected, "uv_hit"),
                    _mean(selected, "visual_injection_gain"),
                )
            )
        array = np.asarray(per_benchmark, dtype=np.float64)
        output.append(
            {
                "decile": decile,
                "eligible_states": states,
                "u_recall": float(array[:, 0].mean()) if len(array) else None,
                "uv_recall": float(array[:, 1].mean()) if len(array) else None,
                "visual_injection_gain": (
                    float(array[:, 2].mean()) if len(array) else None
                ),
            }
        )
    return output


def plot(payload: dict, output_stem: Path) -> None:
    rows = payload["macro_deciles"]
    x = np.arange(1, 11)
    u = np.asarray([row["u_recall"] for row in rows], dtype=float)
    uv = np.asarray([row["uv_recall"] for row in rows], dtype=float)
    gain = np.asarray([row["visual_injection_gain"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45), constrained_layout=True)
    axes[0].plot(x, u, marker="o", color="#E67E22", label="U (8 slots)")
    axes[0].plot(x, uv, marker="o", color="#2468B4", label="U+V (7+1 slots)")
    axes[0].set_ylabel("Root target-token recall")
    axes[0].set_ylim(0.0, min(1.0, max(np.nanmax(u), np.nanmax(uv)) + 0.12))
    axes[0].legend(frameon=False)
    axes[0].set_title("(a) Equal-budget candidate recall")
    axes[1].plot(x, gain, marker="o", color="#7A3E9D")
    axes[1].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("U+V minus U recall")
    axes[1].set_title("(b) Value of one visual slot")
    for axis in axes:
        axis.set_xlabel("True-occlusion visual-sensitivity decile")
        axis.set_xticks(x)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    label = (
        "DEVELOPMENT — mechanism pilot, not confirmatory evidence"
        if payload["analysis_role"] == "development"
        else "CONFIRMATORY — frozen rule on image-disjoint data"
    )
    fig.suptitle(f"{label} · {payload['num_benchmarks']} benchmarks", fontsize=10.5)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(payload: dict, path: Path) -> None:
    gate = payload["evidence_gate"]
    role = payload["analysis_role"]
    report = f"""# Selective Visual-Injection Diagnostic

- Analysis role: **{role}**.
- Included benchmarks: {", ".join(payload["benchmarks"])}.
- MME: **excluded**.
- Frozen comparison: U top-8 versus equal-budget U+V (seven U candidates plus one prompt-native `visual_max` candidate).
- Eligible states: {payload["num_eligible_states"]:,} from {payload["num_input_states"]:,} input states.

## Tail result

- Low visual sensitivity (deciles 1-2), U+V minus U: **{gate["low_visual_gain"]:+.4f}**, 95% cluster-bootstrap CI **[{gate["low_visual_gain_95_ci"][0]:+.4f}, {gate["low_visual_gain_95_ci"][1]:+.4f}]**.
- High visual sensitivity (deciles 9-10), U+V minus U: **{gate["high_visual_gain"]:+.4f}**, 95% CI **[{gate["high_visual_gain_95_ci"][0]:+.4f}, {gate["high_visual_gain_95_ci"][1]:+.4f}]**.
- High-minus-low interaction: **{gate["high_minus_low_gain"]:+.4f}**, 95% CI **[{gate["high_minus_low_gain_95_ci"][0]:+.4f}, {gate["high_minus_low_gain_95_ci"][1]:+.4f}]**.
- Positive interaction direction: {gate["positive_interaction_benchmarks"]}/{payload["num_benchmarks"]} benchmarks.
- Verdict: **{gate["verdict"]}**.

This is a root-token mechanism diagnostic and is not an acceleration measurement. The visual row is a shadow candidate row and does not alter live decoding or its measured speed.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analysis-role", choices=("development", "confirmatory"), required=True
    )
    parser.add_argument("--analysis-split", choices=("all", "discovery", "heldout"), default="all")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=271828)
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
        raise ValueError("no valid U versus U+V states found")

    point, by_benchmark = _macro_tail_stats(eligible)
    draws = clustered_bootstrap(
        eligible, resamples=args.bootstrap_resamples, seed=args.seed
    )
    interaction_ci = _ci(draws[:, 2])
    high_ci = _ci(draws[:, 1])
    positive_benchmarks = sum(
        row["high_minus_low_gain"] > 0.0 for row in by_benchmark.values()
    )
    if args.analysis_role == "development":
        passes = bool(point[1] > 0.0 and point[2] > 0.0 and positive_benchmarks >= 5)
        verdict = "pilot_go_to_disjoint_confirmation" if passes else "pilot_no_go"
    else:
        passes = bool(
            high_ci[0] is not None
            and high_ci[0] > 0.0
            and interaction_ci[0] is not None
            and interaction_ci[0] > 0.0
            and positive_benchmarks >= 6
        )
        verdict = "selective_visual_injection_supported" if passes else "not_supported"

    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": args.analysis_split,
        "source": SOURCE,
        "candidate_budget": CANDIDATE_BUDGET,
        "visual_slots": VISUAL_SLOTS,
        "input_paths": [str(path) for path in paths],
        "num_input_states": len(records),
        "num_eligible_states": len(eligible),
        "num_benchmarks": len(by_benchmark),
        "benchmarks": sorted(by_benchmark),
        "excluded_benchmarks": ["MME"],
        "audit": audit,
        "visual_sensitivity": "root-token JSD over full image and four true pixel-occluded views",
        "outcome": "root target-token hit in an exactly matched eight-candidate row",
        "tail_definition": "within-benchmark bottom/top two visual-sensitivity deciles",
        "bootstrap": {
            "unit": "image cluster",
            "stratification": "benchmark",
            "macro_averaging": "equal benchmark weight",
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "evidence_gate": {
            "verdict": verdict,
            "pass": passes,
            "low_visual_gain": float(point[0]),
            "low_visual_gain_95_ci": _ci(draws[:, 0]),
            "high_visual_gain": float(point[1]),
            "high_visual_gain_95_ci": high_ci,
            "high_minus_low_gain": float(point[2]),
            "high_minus_low_gain_95_ci": interaction_ci,
            "positive_interaction_benchmarks": positive_benchmarks,
            "confirmatory_rule": "high-gain CI > 0, interaction CI > 0, and positive interaction in at least 6/8 benchmarks",
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
    plot(payload, args.output_dir / "selective_visual_injection")
    write_report(payload, args.output_dir / "REPORT.md")
    print(json.dumps(payload["evidence_gate"], indent=2))


if __name__ == "__main__":
    main()
