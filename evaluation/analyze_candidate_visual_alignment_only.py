"""Hierarchical v16 validation of candidate-relative visual alignment only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_candidate_visual_alignment import (  # noqa: E402
    VISUAL_ALIGNMENT_FIELD,
    assign_candidate_alignment_strata,
)
from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    BLUE,
    INK,
    MUTED,
    ORANGE,
    average_tie_percentiles,
    prepare,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)
from evaluation.analyze_visual_reliability_arbitration import (  # noqa: E402
    MIN_VALIDATION_STATES_PER_STRATUM,
    TAIL_FRACTION,
    clustered_bootstrap,
    macro_summary,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    COUNTERFACTUAL_BANK_PROTOCOL,
)


DEFAULT_EXPECTED_BENCHMARKS = (
    "MMT-Bench,SEEDBench,ScienceQA,OCRBench,ChartQA,MathVista,TextVQA"
)


def assign_candidate_only_strata(
    rows: Sequence[dict], *, tail_fraction: float = TAIL_FRACTION
) -> dict:
    """Replace the joint labels with frozen candidate-alignment-only tails."""

    # This populates and audits the candidate-alignment rank using the frozen
    # score.  The joint label it initially writes is intentionally discarded.
    base_audit = assign_candidate_alignment_strata(
        rows, tail_fraction=tail_fraction
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["arbitration_stratum"] = "ineligible"
        row["arbitration_score"] = None
        row["arbitration_score_percentile"] = None
        if row.get("arbitration_visual_rank") is not None:
            grouped[row["benchmark"]].append(row)

    by_benchmark = {}
    for benchmark, benchmark_rows in sorted(grouped.items()):
        raw_values = np.asarray(
            [float(row[VISUAL_ALIGNMENT_FIELD]) for row in benchmark_rows]
        )
        # Ranking the already ranked values is equivalent except for explicit
        # tied-score handling; recompute from raw D to make that rule visible.
        percentiles = average_tie_percentiles(raw_values)
        counts: dict[str, int] = defaultdict(int)
        for row, percentile in zip(benchmark_rows, percentiles.tolist()):
            row["arbitration_visual_rank"] = float(percentile)
            row["arbitration_score"] = float(percentile)
            row["arbitration_score_percentile"] = float(percentile)
            if float(percentile) < float(tail_fraction):
                stratum = "low"
            elif float(percentile) >= 1.0 - float(tail_fraction):
                stratum = "high"
            else:
                stratum = "ambiguous"
            row["arbitration_stratum"] = stratum
            counts[stratum] += 1
        counts["reference_states"] = len(benchmark_rows)
        by_benchmark[benchmark] = dict(counts)
    return {
        "uses_source_reliability": False,
        "uses_source_hit_outcomes": False,
        "uses_accepted_length_outcomes": False,
        "uses_target_token_id": False,
        "tail_fraction": float(tail_fraction),
        "score_field": VISUAL_ALIGNMENT_FIELD,
        "base_candidate_alignment_audit": base_audit,
        "by_benchmark": by_benchmark,
    }


def _bar_errors(values, names, intervals):
    return np.asarray(
        [
            [value - intervals[name][0] for value, name in zip(values, names)],
            [intervals[name][1] - value for value, name in zip(values, names)],
        ]
    )


def plot(payload: dict, output_stem: Path) -> None:
    point = payload["point_estimates"]
    intervals = payload["cluster_bootstrap_95_ci"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.48))
    x = np.arange(2)
    labels = ["U-aligned\nregime", "G/C-aligned\nregime"]

    names = ("low_visual_rank", "high_visual_rank")
    values = [point[name] for name in names]
    axes[0].bar(
        x,
        values,
        0.48,
        yerr=_bar_errors(values, names, intervals),
        capsize=2.8,
        color=[ORANGE, BLUE],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("Candidate-alignment rank")
    axes[0].set_title("(a) Visual source evidence", loc="left", weight="bold")

    width = 0.32
    for offset, prefix, color, label in (
        (-width / 2, "u", ORANGE, "Persistent U"),
        (width / 2, "gc", BLUE, "Request-local G/C"),
    ):
        names = (f"low_{prefix}_recall", f"high_{prefix}_recall")
        values = [point[name] for name in names]
        axes[1].bar(
            x + offset,
            values,
            width,
            yerr=_bar_errors(values, names, intervals),
            capsize=2.4,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axes[1].set_ylim(0.0, 1.03)
    axes[1].set_ylabel("Equal-budget root recall")
    axes[1].set_title("(b) Root recall", loc="left", weight="bold")

    names = ("low_accept_delta", "high_accept_delta")
    values = [point[name] for name in names]
    axes[2].bar(
        x,
        values,
        0.48,
        yerr=_bar_errors(values, names, intervals),
        capsize=2.8,
        color=[ORANGE, BLUE],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[2].axhline(0.0, color="#5D6670", linestyle=(0, (3, 2)), lw=0.9)
    bound = max(
        0.25,
        max(
            abs(intervals[name][edge])
            for name in names
            for edge in (0, 1)
        )
        * 1.15,
    )
    axes[2].set_ylim(-bound, bound)
    axes[2].set_ylabel("G/C - U accepted tokens")
    axes[2].set_title("(c) Accepted-token gain", loc="left", weight="bold")
    for index, value in enumerate(values):
        offset = 0.045 * bound
        axes[2].text(
            index,
            value + (offset if value >= 0.0 else -offset),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=6.2,
            weight="bold",
            color=INK,
        )

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.title.set_fontsize(7.5)
        axis.yaxis.label.set_size(6.3)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.955),
        ncol=2,
        frameon=False,
        fontsize=6.1,
        handlelength=1.5,
        columnspacing=1.0,
    )
    fig.suptitle(
        "The image selects which reusable candidate source is trustworthy",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=9.0,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.905,
        f"HIERARCHICAL VALIDATION · {len(payload['benchmarks'])} benchmarks · "
        "MME/MMSpec excluded · no reliability term",
        ha="left",
        va="center",
        fontsize=5.8,
        weight="bold",
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "Outcome-blind strata; equal benchmark weight; matched root/tree "
        "budget; image-cluster bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=5.1,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.77, bottom=0.25, wspace=0.48)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=223607)
    parser.add_argument("--tail-fraction", type=float, default=TAIL_FRACTION)
    parser.add_argument(
        "--expected-benchmarks", default=DEFAULT_EXPECTED_BENCHMARKS
    )
    args = parser.parse_args()
    if not 0.0 < args.tail_fraction < 0.5:
        parser.error("tail fraction must lie strictly between 0 and 0.5")
    is_confirmatory = bool(np.isclose(args.tail_fraction, TAIL_FRACTION))

    expected = {
        item.strip()
        for item in str(args.expected_benchmarks).split(",")
        if item.strip()
    }
    if not expected or "MME" in expected or "MME_Benchmark" in expected:
        parser.error("expected benchmark set must be nonempty and exclude MME")
    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != expected:
        parser.error(
            f"benchmark mismatch: expected {sorted(expected)}, got {benchmarks}"
        )
    protocols = sorted({str(row["visual_probe_protocol"]) for row in rows})
    if protocols != [COUNTERFACTUAL_BANK_PROTOCOL]:
        parser.error(f"unexpected visual probe protocols: {protocols}")
    if not all(row.get("visual_probe_same_text_trajectory") for row in rows):
        parser.error("counterfactual probes changed a text trajectory")
    if not all(row.get("visual_probe_recomputed_vision_encoder") for row in rows):
        parser.error("counterfactual probes did not recompute every vision input")

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    assignment_audit = assign_candidate_only_strata(
        rows, tail_fraction=args.tail_fraction
    )
    try:
        point, by_benchmark, insufficient = macro_summary(
            rows,
            minimum_states_per_stratum=MIN_VALIDATION_STATES_PER_STRATUM,
        )
    except ValueError:
        point, by_benchmark, insufficient = {}, {}, benchmarks
    support_complete = not insufficient and set(by_benchmark) == expected
    intervals, valid_draws = clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    required_draws = max(100, int(0.90 * args.bootstrap_resamples))
    directional = bool(
        support_complete
        and point.get("low_root_delta", 0.0) < 0.0
        and point.get("high_root_delta", 0.0) > 0.0
        and point.get("root_interaction", 0.0) > 0.0
        and point.get("low_accept_delta", 0.0) < 0.0
        and point.get("high_accept_delta", 0.0) > 0.0
        and point.get("accept_interaction", 0.0) > 0.0
    )
    strict_metrics = (
        ("low_root_delta", 1, "negative"),
        ("high_root_delta", 0, "positive"),
        ("root_interaction", 0, "positive"),
        ("low_accept_delta", 1, "negative"),
        ("high_accept_delta", 0, "positive"),
        ("accept_interaction", 0, "positive"),
    )
    strict = bool(
        directional
        and valid_draws >= required_draws
        and all(
            (
                intervals.get(metric, [0.0, 0.0])[edge] < 0.0
                if direction == "negative"
                else intervals.get(metric, [0.0, 0.0])[edge] > 0.0
            )
            for metric, edge, direction in strict_metrics
        )
    )
    if is_confirmatory:
        decision = "validated" if strict else "validation_no_go"
    else:
        decision = "sensitivity_pass" if strict else "sensitivity_no_go"
    payload = {
        "schema_version": 1,
        "analysis_role": (
            "hierarchical_new_validation"
            if is_confirmatory
            else "post_hoc_tail_fraction_sensitivity"
        ),
        "warning": (
            "This route was frozen while v15 inference was active and before "
            "any validation outcome was inspected."
            if is_confirmatory
            else "Post-hoc tail-fraction sensitivity analysis; not an "
            "independent confirmatory result."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME", "MMSpec"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "frozen_rule": {
            "candidate_alignment_field": VISUAL_ALIGNMENT_FIELD,
            "source_reliability_weight": 0.0,
            "tail_fraction": args.tail_fraction,
            "confirmatory": is_confirmatory,
            "uses_source_reliability": False,
            "uses_source_hit_outcomes": False,
            "uses_accepted_length_outcomes": False,
            "uses_target_token_id": False,
            "minimum_states_per_stratum": MIN_VALIDATION_STATES_PER_STRATUM,
        },
        "eligibility_audit": eligibility_audit,
        "assignment_audit": assignment_audit,
        "point_estimates": point,
        "by_benchmark": by_benchmark,
        "insufficient_support_benchmarks": insufficient,
        "support_complete": support_complete,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "bootstrap_required_valid_draws": required_draws,
        "directional_double_crossover": directional,
        "strict_double_crossover": strict,
        "strict_confirmatory_double_crossover": strict and is_confirmatory,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if support_complete and valid_draws:
        plot(payload, args.output_dir / "candidate_only_visual_alignment_triptych")
    print(
        json.dumps(
            {
                "point_estimates": point,
                "insufficient_support_benchmarks": insufficient,
                "strict_double_crossover": strict,
                "strict_confirmatory_double_crossover": (
                    strict and is_confirmatory
                ),
                "tail_fraction": args.tail_fraction,
                "decision": decision,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
