"""Plot the supported held-out selective-reuse tension as low/high tails."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "u_coverage",
    "gc_coverage",
    "u_conditional_top8_recall",
    "gc_conditional_top8_recall",
    "gc_minus_u_matched_accept",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eligible_for_acceptance(row: Mapping, rule: Mapping) -> bool:
    return bool(
        row["u_available"]
        and row["gc_available"]
        and int(row["matched_tree_node_budget"])
        >= int(rule["minimum_matched_tree_node_budget"])
    )


def summarize_rows(rows: Sequence[dict], rule: Mapping) -> dict[str, float]:
    if not rows:
        raise ValueError("tail contains no states")
    u_available = [row for row in rows if row["u_available"]]
    gc_available = [row for row in rows if row["gc_available"]]
    acceptance = [row for row in rows if eligible_for_acceptance(row, rule)]
    if not u_available or not gc_available or not acceptance:
        raise ValueError("tail lacks a required source or acceptance state")
    return {
        "u_coverage": float(np.mean([row["u_available"] for row in rows])),
        "gc_coverage": float(np.mean([row["gc_available"] for row in rows])),
        "u_conditional_top8_recall": float(
            np.mean([row["u_top8_hit"] for row in u_available])
        ),
        "gc_conditional_top8_recall": float(
            np.mean([row["gc_top8_hit"] for row in gc_available])
        ),
        "gc_minus_u_matched_accept": float(
            np.mean([row["matched_gc_minus_u_accept"] for row in acceptance])
        ),
    }


def macro_summary(rows: Sequence[dict], rule: Mapping) -> dict[str, dict]:
    output = {}
    for tail, predicate in (
        ("low", lambda row: int(row["visual_sensitivity_decile"]) <= 2),
        ("high", lambda row: int(row["visual_sensitivity_decile"]) >= 9),
    ):
        per_benchmark = []
        state_count = 0
        for benchmark in sorted({row["benchmark"] for row in rows}):
            selected = [
                row
                for row in rows
                if row["benchmark"] == benchmark and predicate(row)
            ]
            if not selected:
                continue
            try:
                per_benchmark.append(summarize_rows(selected, rule))
                state_count += len(selected)
            except ValueError:
                continue
        if not per_benchmark:
            raise ValueError(f"no valid benchmarks in {tail} tail")
        output[tail] = {
            "num_states": state_count,
            "num_benchmarks": len(per_benchmark),
            **{
                metric: float(np.mean([row[metric] for row in per_benchmark]))
                for metric in METRICS
            },
        }
    return output


def clustered_bootstrap(
    rows: Sequence[dict], rule: Mapping, *, resamples: int, seed: int
) -> dict[str, dict[str, list[float]]]:
    # Aggregate once to cluster-level sufficient statistics.  Repeatedly
    # rebuilding Python row lists made the original, equivalent bootstrap far
    # slower than the GPU experiment itself.
    grouped: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "low": np.zeros(7, dtype=np.float64),
                "high": np.zeros(7, dtype=np.float64),
            }
        )
    )
    for row in rows:
        decile = int(row["visual_sensitivity_decile"])
        tail = "low" if decile <= 2 else ("high" if decile >= 9 else None)
        if tail is None:
            continue
        vector = grouped[row["benchmark"]][row["cluster_id"]][tail]
        vector[0] += 1.0
        if row["u_available"]:
            vector[1] += 1.0
            vector[3] += float(row["u_top8_hit"])
        if row["gc_available"]:
            vector[2] += 1.0
            vector[4] += float(row["gc_top8_hit"])
        if eligible_for_acceptance(row, rule):
            vector[5] += 1.0
            vector[6] += float(row["matched_gc_minus_u_accept"])

    def metrics_from_vector(vector: np.ndarray) -> dict[str, float] | None:
        if vector[0] <= 0 or vector[1] <= 0 or vector[2] <= 0 or vector[5] <= 0:
            return None
        return {
            "u_coverage": float(vector[1] / vector[0]),
            "gc_coverage": float(vector[2] / vector[0]),
            "u_conditional_top8_recall": float(vector[3] / vector[1]),
            "gc_conditional_top8_recall": float(vector[4] / vector[2]),
            "gc_minus_u_matched_accept": float(vector[6] / vector[5]),
        }

    rng = np.random.default_rng(int(seed))
    draws = {
        tail: {metric: [] for metric in METRICS} for tail in ("low", "high")
    }
    for _draw in range(int(resamples)):
        per_tail = {"low": [], "high": []}
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            indices = rng.integers(0, len(clusters), size=len(clusters))
            for tail in ("low", "high"):
                vector = np.sum(
                    [
                        grouped[benchmark][clusters[index]][tail]
                        for index in indices.tolist()
                    ],
                    axis=0,
                )
                metrics = metrics_from_vector(vector)
                if metrics is not None:
                    per_tail[tail].append(metrics)
        for tail in ("low", "high"):
            for metric in METRICS:
                draws[tail][metric].append(
                    float(
                        np.mean([row[metric] for row in per_tail[tail]])
                    )
                )
    return {
        tail: {
            metric: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for metric, values in metrics.items()
        }
        for tail, metrics in draws.items()
    }


def asymmetric_error(value: float, interval: Sequence[float]) -> list[list[float]]:
    return [[value - float(interval[0])], [float(interval[1]) - value]]


def plot(payload: dict, output_stem: Path) -> None:
    point = payload["point_estimates"]
    ci = payload["cluster_bootstrap_95_ci"]
    colors = {"U": "#E67E22", "GC": "#2468B4"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.65), constrained_layout=True)
    x = np.arange(2)
    width = 0.34

    def grouped_bars(axis, u_metric: str, gc_metric: str, ylabel: str, title: str):
        for offset, source, metric in (
            (-width / 2, "U", u_metric),
            (width / 2, "GC", gc_metric),
        ):
            values = [point[tail][metric] for tail in ("low", "high")]
            lower = [
                values[index] - ci[tail][metric][0]
                for index, tail in enumerate(("low", "high"))
            ]
            upper = [
                ci[tail][metric][1] - values[index]
                for index, tail in enumerate(("low", "high"))
            ]
            axis.bar(
                x + offset,
                values,
                width,
                yerr=np.asarray([lower, upper]),
                capsize=3,
                color=colors[source],
                alpha=0.9,
                label="G/C" if source == "GC" else source,
            )
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(frameon=False, loc="lower left")

    grouped_bars(
        axes[0],
        "u_coverage",
        "gc_coverage",
        "Available-source probability",
        "(a) Availability",
    )
    grouped_bars(
        axes[1],
        "u_conditional_top8_recall",
        "gc_conditional_top8_recall",
        "Target-token top-8 recall",
        "(b) Quality when available",
    )

    metric = "gc_minus_u_matched_accept"
    values = [point[tail][metric] for tail in ("low", "high")]
    lower = [
        values[index] - ci[tail][metric][0]
        for index, tail in enumerate(("low", "high"))
    ]
    upper = [
        ci[tail][metric][1] - values[index]
        for index, tail in enumerate(("low", "high"))
    ]
    axes[2].bar(
        x,
        values,
        0.52,
        yerr=np.asarray([lower, upper]),
        capsize=4,
        color=["#4C78A8", "#9ECAE1"],
    )
    axes[2].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("G/C minus U accepted tokens")
    axes[2].set_title("(c) Equal-node utility advantage")
    reduction = payload["supported_contrast"]["relative_advantage_reduction"]
    axes[2].text(
        0.5,
        max(values) * 0.72,
        f"{reduction * 100:.0f}% smaller",
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    for axis in axes:
        axis.set_xticks(x, ["Low visual\n(bottom 20%)", "High visual\n(top 20%)"])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "HELD-OUT · 8 multimodal benchmarks · true pixel-occlusion sensitivity",
        fontsize=11.5,
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    records = [
        row
        for row in read_jsonl(args.records)
        if row["analysis_split"] == "heldout"
    ]
    benchmarks = sorted({row["benchmark"] for row in records})
    if "MME" in benchmarks or "MME_Benchmark" in benchmarks:
        parser.error("MME must remain excluded")
    if len(benchmarks) != 8:
        parser.error(f"expected 8 included benchmarks, found {benchmarks}")
    rule = json.loads(args.frozen_rule.read_text(encoding="utf-8"))
    point = macro_summary(records, rule)
    intervals = clustered_bootstrap(
        records,
        rule,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    low = point["low"]["gc_minus_u_matched_accept"]
    high = point["high"]["gc_minus_u_matched_accept"]
    payload = {
        "schema_version": 1,
        "status": "heldout_supported_result",
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(records),
        "num_image_clusters": len({row["cluster_id"] for row in records}),
        "tail_definition": "within-benchmark bottom/top two deciles",
        "frozen_rule": rule,
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "supported_contrast": {
            "low_gc_minus_u_matched_accept": low,
            "high_gc_minus_u_matched_accept": high,
            "high_minus_low": high - low,
            "relative_advantage_reduction": (low - high) / low,
            "interpretation": "G/C remains stronger when available, but its equal-budget advantage over U is substantially smaller on visually sensitive states.",
        },
        "claim_boundary": "This supports visual-sensitivity-dependent weakening, not a sign crossover or a visual-injection speedup.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(payload, args.output_dir / "supported_selective_reuse_tension")
    print(json.dumps(payload["supported_contrast"], indent=2))


if __name__ == "__main__":
    main()
