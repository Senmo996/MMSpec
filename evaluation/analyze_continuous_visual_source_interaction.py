"""Continuous visual-demand × candidate-source interaction analysis.

Low/high tail plots can lose power and depend on an arbitrary cut point.  This
analysis keeps the frozen visual score and source-conflict eligibility rule but
fits equal-benchmark-weighted linear probability trends over every eligible
state.  Image-cluster bootstrap intervals quantify the source-preference slope
and the predicted G/C-minus-U differences at visual percentiles 0.10 and 0.90.

The percentile reference population can be either all decoder states (the
historical default) or only outcome-independently eligible source-conflict
states.  The latter avoids defining visual tails on a population and then
discarding most of it before comparing sources.

This is a diagnostic model, not an online router and not an end-to-end speed
measurement.
"""

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

from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    BLUE,
    DEFAULT_VISUAL_METRIC,
    INK,
    MUTED,
    ORANGE,
    VISUAL_METRICS,
    VISUAL_METRIC_DESCRIPTIONS,
    VISUAL_METRIC_FIELDS,
    average_tie_percentiles,
    macro_summary as tail_macro_summary,
    prepare,
    summarize_benchmark,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)


LOW_ENDPOINT = 0.10
HIGH_ENDPOINT = 0.90
VISUAL_REFERENCE_POPULATIONS = ("all_states", "eligible_conflicts")
CONTINUOUS_METRICS = (
    "u_intercept",
    "u_slope",
    "gc_intercept",
    "gc_slope",
    "delta_intercept",
    "delta_slope",
    "delta_p10",
    "delta_p90",
)
COVERAGE_METRICS = (
    "low_u_availability",
    "low_gc_availability",
    "high_u_availability",
    "high_gc_availability",
)
EMPIRICAL_TAIL_METRICS = (
    "low_u_recall",
    "low_gc_recall",
    "high_u_recall",
    "high_gc_recall",
    "low_delta",
    "high_delta",
    "interaction",
)


def fit_linear_probability(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = intercept + slope * (visual_percentile - 0.5)."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be equal-length one-dimensional arrays")
    if len(x) < 3 or float(np.ptp(x)) <= 0.0:
        raise ValueError("at least three states with varying visual scores are required")
    design = np.column_stack((np.ones(len(x)), x - 0.5))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def assign_analysis_visual_percentiles(
    rows: Sequence[dict], *, reference_population: str
) -> dict:
    """Assign visual ranks without using U/G/C hit outcomes.

    ``prepare`` always records the historical all-state percentile. This
    helper stores the percentile used by the continuous model separately so
    source-coverage summaries retain their original all-state definition.
    """

    if reference_population not in VISUAL_REFERENCE_POPULATIONS:
        raise ValueError(
            "unsupported visual reference population: "
            f"{reference_population}"
        )
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["all_state_visual_percentile"] = float(
            row["frozen_visual_percentile"]
        )
        row["analysis_visual_percentile"] = None
        if reference_population == "all_states" or bool(
            row.get("frozen_eligible", False)
        ):
            by_benchmark[row["benchmark"]].append(row)

    counts = {}
    for benchmark, benchmark_rows in by_benchmark.items():
        percentiles = average_tie_percentiles(
            np.asarray(
                [float(row["frozen_visual_score"]) for row in benchmark_rows],
                dtype=float,
            )
        )
        for row, percentile in zip(benchmark_rows, percentiles.tolist()):
            row["analysis_visual_percentile"] = float(percentile)
        counts[benchmark] = len(benchmark_rows)
    return {
        "reference_population": reference_population,
        "uses_source_hit_outcomes": False,
        "states_by_benchmark": counts,
        "num_reference_states": int(sum(counts.values())),
    }


def summarize_continuous_benchmark(
    rows: Sequence[dict], *, percentile_field: str = "frozen_visual_percentile"
) -> dict | None:
    eligible = [row for row in rows if row.get("frozen_eligible", False)]
    if len(eligible) < 3:
        return None
    x = np.asarray(
        [float(row[percentile_field]) for row in eligible], dtype=float
    )
    if float(np.ptp(x)) <= 0.0:
        return None
    output = {"eligible_states": len(eligible)}
    for source, field in (
        ("u", "frozen_u_hit"),
        ("gc", "frozen_gc_hit"),
        ("delta", "frozen_delta"),
    ):
        intercept, slope = fit_linear_probability(
            x, np.asarray([float(row[field]) for row in eligible], dtype=float)
        )
        output[f"{source}_intercept"] = intercept
        output[f"{source}_slope"] = slope
    output["delta_p10"] = output["delta_intercept"] + (
        LOW_ENDPOINT - 0.5
    ) * output["delta_slope"]
    output["delta_p90"] = output["delta_intercept"] + (
        HIGH_ENDPOINT - 0.5
    ) * output["delta_slope"]
    return output


def continuous_macro_summary(
    rows: Sequence[dict], *, percentile_field: str = "frozen_visual_percentile"
) -> tuple[dict, dict]:
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_continuous_benchmark(
            [row for row in rows if row["benchmark"] == benchmark],
            percentile_field=percentile_field,
        )
        if summary is not None:
            by_benchmark[benchmark] = summary
    if not by_benchmark:
        raise ValueError("no benchmark has a valid continuous source comparison")
    macro = {
        metric: float(np.mean([row[metric] for row in by_benchmark.values()]))
        for metric in CONTINUOUS_METRICS
    }
    return macro, by_benchmark


def summarize_empirical_conflict_tails_benchmark(
    rows: Sequence[dict], *, percentile_field: str
) -> dict | None:
    """Summarize observed, rather than fitted, outcomes in the visual tails."""

    eligible = [
        row
        for row in rows
        if row.get("frozen_eligible", False)
        and row.get(percentile_field) is not None
    ]
    low = [row for row in eligible if float(row[percentile_field]) < 0.20]
    high = [row for row in eligible if float(row[percentile_field]) >= 0.80]
    if not low or not high:
        return None
    output = {
        "eligible_states": len(eligible),
        "low_eligible_states": len(low),
        "high_eligible_states": len(high),
        "low_u_recall": float(np.mean([row["frozen_u_hit"] for row in low])),
        "low_gc_recall": float(np.mean([row["frozen_gc_hit"] for row in low])),
        "high_u_recall": float(np.mean([row["frozen_u_hit"] for row in high])),
        "high_gc_recall": float(np.mean([row["frozen_gc_hit"] for row in high])),
        "low_delta": float(np.mean([row["frozen_delta"] for row in low])),
        "high_delta": float(np.mean([row["frozen_delta"] for row in high])),
    }
    output["interaction"] = output["high_delta"] - output["low_delta"]
    return output


def empirical_conflict_tail_macro_summary(
    rows: Sequence[dict], *, percentile_field: str
) -> tuple[dict, dict]:
    """Equal-benchmark mean of direct bottom/top-quintile observations."""

    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_empirical_conflict_tails_benchmark(
            [row for row in rows if row["benchmark"] == benchmark],
            percentile_field=percentile_field,
        )
        if summary is not None:
            by_benchmark[benchmark] = summary
    if not by_benchmark:
        raise ValueError("no benchmark has observed eligible states in both tails")
    macro = {
        metric: float(np.mean([row[metric] for row in by_benchmark.values()]))
        for metric in EMPIRICAL_TAIL_METRICS
    }
    return macro, by_benchmark


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int, percentile_field: str
) -> tuple[dict, dict, dict, dict, int]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    metric_names = CONTINUOUS_METRICS + COVERAGE_METRICS
    draws = {metric: [] for metric in metric_names}
    empirical_tail_draws = {metric: [] for metric in EMPIRICAL_TAIL_METRICS}
    empirical_tail_valid_all_benchmark_draws = 0
    coefficient_draws = []
    for _ in range(int(resamples)):
        continuous_summaries = []
        tail_summaries = []
        empirical_tail_summaries = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            continuous = summarize_continuous_benchmark(
                sampled, percentile_field=percentile_field
            )
            if continuous is not None:
                continuous_summaries.append(continuous)
            tail = summarize_benchmark(sampled)
            if tail is not None:
                tail_summaries.append(tail)
            empirical_tail = summarize_empirical_conflict_tails_benchmark(
                sampled, percentile_field=percentile_field
            )
            if empirical_tail is not None:
                empirical_tail_summaries.append(empirical_tail)
        if continuous_summaries:
            current = {
                metric: float(
                    np.mean([row[metric] for row in continuous_summaries])
                )
                for metric in CONTINUOUS_METRICS
            }
            for metric, value in current.items():
                draws[metric].append(value)
            coefficient_draws.append(
                [
                    current["u_intercept"],
                    current["u_slope"],
                    current["gc_intercept"],
                    current["gc_slope"],
                    current["delta_intercept"],
                    current["delta_slope"],
                ]
            )
        if tail_summaries:
            for metric in COVERAGE_METRICS:
                draws[metric].append(
                    float(np.mean([row[metric] for row in tail_summaries]))
                )
        if len(empirical_tail_summaries) == len(grouped):
            empirical_tail_valid_all_benchmark_draws += 1
            for metric in EMPIRICAL_TAIL_METRICS:
                empirical_tail_draws[metric].append(
                    float(
                        np.mean(
                            [row[metric] for row in empirical_tail_summaries]
                        )
                    )
                )
    intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
        if values
    }
    grid = np.linspace(0.0, 1.0, 101)
    coefficients = np.asarray(coefficient_draws, dtype=float)
    bands = {"visual_percentile": grid.tolist()}
    for source, offset in (("u", 0), ("gc", 2), ("delta", 4)):
        values = coefficients[:, offset, None] + coefficients[:, offset + 1, None] * (
            grid[None, :] - 0.5
        )
        bands[source] = {
            "lower": np.percentile(values, 2.5, axis=0).tolist(),
            "upper": np.percentile(values, 97.5, axis=0).tolist(),
        }
    empirical_tail_intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in empirical_tail_draws.items()
        if values
    }
    return (
        intervals,
        bands,
        {metric: len(values) for metric, values in draws.items()},
        empirical_tail_intervals,
        empirical_tail_valid_all_benchmark_draws,
    )


def plot(payload: dict, output_stem: Path) -> None:
    tail = payload["tail_point_estimates"]
    continuous = payload["continuous_point_estimates"]
    ci = payload["cluster_bootstrap_95_ci"]
    bands = payload["continuous_prediction_bands_95"]
    empirical = payload["empirical_conflict_tail_point_estimates"]
    empirical_ci = payload["empirical_conflict_tail_cluster_bootstrap_95_ci"]
    grid = np.asarray(bands["visual_percentile"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    x = np.arange(2)
    width = 0.32
    labels = ["Low visual\n(bottom 20%)", "High visual\n(top 20%)"]

    for offset, metrics, color, label in (
        (
            -width / 2,
            ("low_u_availability", "high_u_availability"),
            ORANGE,
            "Persistent U",
        ),
        (
            width / 2,
            ("low_gc_availability", "high_gc_availability"),
            BLUE,
            "Request-local G/C",
        ),
    ):
        values = [tail[metric] for metric in metrics]
        errors = np.asarray(
            [
                [values[index] - ci[metric][0] for index, metric in enumerate(metrics)],
                [ci[metric][1] - values[index] for index, metric in enumerate(metrics)],
            ]
        )
        axes[0].bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=2.5,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("Availability")
    axes[0].set_title("(a) Source coverage", loc="left", weight="bold")

    for source, color, label in (
        ("u", ORANGE, "Persistent U"),
        ("gc", BLUE, "Request-local G/C"),
    ):
        mean = continuous[f"{source}_intercept"] + continuous[f"{source}_slope"] * (
            grid - 0.5
        )
        lower = np.asarray(bands[source]["lower"], dtype=float)
        upper = np.asarray(bands[source]["upper"], dtype=float)
        axes[1].fill_between(grid, lower, upper, color=color, alpha=0.16, lw=0)
        axes[1].plot(grid, mean, color=color, lw=1.8, label=label)
        empirical_metrics = (
            (f"low_{source}_recall", LOW_ENDPOINT),
            (f"high_{source}_recall", HIGH_ENDPOINT),
        )
        for metric, endpoint in empirical_metrics:
            value = empirical[metric]
            interval = empirical_ci[metric]
            axes[1].errorbar(
                [endpoint],
                [value],
                yerr=[[value - interval[0]], [interval[1] - value]],
                fmt="o",
                ms=3.4,
                color=color,
                mec="white",
                mew=0.45,
                capsize=2.0,
                lw=0.8,
                zorder=5,
            )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Visual-score percentile")
    axes[1].set_ylabel("Predicted root recall")
    axes[1].set_title("(b) Continuous recall", loc="left", weight="bold")

    delta = continuous["delta_intercept"] + continuous["delta_slope"] * (grid - 0.5)
    axes[2].fill_between(
        grid,
        np.asarray(bands["delta"]["lower"], dtype=float),
        np.asarray(bands["delta"]["upper"], dtype=float),
        color=BLUE,
        alpha=0.16,
        lw=0,
    )
    axes[2].plot(grid, delta, color=INK, lw=1.8)
    axes[2].axhline(0.0, color="#5D6670", linestyle=(0, (3, 2)), lw=0.9)
    axes[2].axvline(LOW_ENDPOINT, color=ORANGE, alpha=0.45, lw=0.8)
    axes[2].axvline(HIGH_ENDPOINT, color=BLUE, alpha=0.45, lw=0.8)
    for metric, endpoint, color in (
        ("low_delta", LOW_ENDPOINT, ORANGE),
        ("high_delta", HIGH_ENDPOINT, BLUE),
    ):
        value = empirical[metric]
        interval = empirical_ci[metric]
        axes[2].errorbar(
            [endpoint],
            [value],
            yerr=[[value - interval[0]], [interval[1] - value]],
            fmt="o",
            ms=4.0,
            color=color,
            mec="white",
            mew=0.45,
            capsize=2.2,
            lw=0.85,
            zorder=5,
        )
    axes[2].set_xlim(0.0, 1.0)
    bound = max(
        0.15,
        max(
            abs(float(np.min(bands["delta"]["lower"]))),
            abs(float(np.max(bands["delta"]["upper"]))),
            abs(float(empirical_ci["low_delta"][0])),
            abs(float(empirical_ci["low_delta"][1])),
            abs(float(empirical_ci["high_delta"][0])),
            abs(float(empirical_ci["high_delta"][1])),
        )
        * 1.1,
    )
    axes[2].set_ylim(-bound, bound)
    axes[2].set_xlabel("Visual-score percentile")
    axes[2].set_ylabel("Δ root recall (G/C − U)")
    axes[2].set_title("(c) Source preference", loc="left", weight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.title.set_fontsize(8.0)
    role = payload["analysis_role"].upper().replace("_", " ")
    reference_population = payload["frozen_rule"][
        "visual_percentile_reference_population"
    ]
    title = (
        "Conflict-conditioned visual-demand source test"
        if reference_population == "eligible_conflicts"
        else "Continuous visual-demand source-conflict test"
    )
    fig.suptitle(
        title,
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=9.2,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.905,
        f"{role} · {len(payload['benchmarks'])} benchmarks · MME excluded",
        ha="left",
        va="center",
        fontsize=5.9,
        weight="bold",
        color=MUTED,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.953),
        ncol=2,
        frameon=False,
        fontsize=6.1,
        handlelength=1.5,
        columnspacing=1.0,
    )
    protocol = payload["visual_probe_protocols"][0]
    intervention = (
        "strongest regional occlusion"
        if protocol == "teacher_forced_mean_patch_occlusion_v1"
        else "whole-image content ablation"
    )
    visual_description = VISUAL_METRIC_DESCRIPTIONS[
        payload["frozen_rule"]["visual_metric"]
    ].format(intervention=intervention)
    fig.text(
        0.99,
        0.012,
        f"Visual score: {visual_description}; recall rank reference: "
        f"{reference_population.replace('_', ' ')}; linear probability trend; "
        "image-cluster bootstrap 95% CI. Circles are observed conflict-tail "
        "means; lines are fitted trends. Coverage bars retain all-state tails.",
        ha="right",
        va="bottom",
        fontsize=5.3,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.77, bottom=0.25, wspace=0.56)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--analysis-split", choices=("discovery", "heldout", "all"), required=True
    )
    parser.add_argument(
        "--analysis-role",
        choices=(
            "development",
            "retrospective_heldout",
            "retrospective_independent_pilot",
            "outcome_blind_secondary_validation",
            "new_validation",
        ),
        required=True,
    )
    parser.add_argument(
        "--visual-metric", choices=VISUAL_METRICS, default=DEFAULT_VISUAL_METRIC
    )
    parser.add_argument(
        "--visual-reference-population",
        choices=VISUAL_REFERENCE_POPULATIONS,
        default="all_states",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=161803)
    args = parser.parse_args()
    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    if args.analysis_split != "all":
        rows = [row for row in rows if row["analysis_split"] == args.analysis_split]
    benchmarks = sorted({row["benchmark"] for row in rows})
    if any("MME" in benchmark.upper() for benchmark in benchmarks):
        parser.error("MME must remain excluded")
    protocols = sorted({str(row["visual_probe_protocol"]) for row in rows})
    if len(protocols) != 1:
        parser.error(f"expected exactly one visual probe protocol, got {protocols}")
    same_text_ratio = float(
        np.mean([bool(row["visual_probe_same_text_trajectory"]) for row in rows])
    )
    recomputed_vision_ratio = float(
        np.mean([bool(row["visual_probe_recomputed_vision_encoder"]) for row in rows])
    )
    if same_text_ratio != 1.0 or recomputed_vision_ratio != 1.0:
        parser.error("visual probes failed the same-path/recomputed-vision audit")

    eligibility_audit = prepare(rows, visual_metric=args.visual_metric)
    percentile_audit = assign_analysis_visual_percentiles(
        rows, reference_population=args.visual_reference_population
    )
    continuous, continuous_by_benchmark = continuous_macro_summary(
        rows, percentile_field="analysis_visual_percentile"
    )
    empirical_tail, empirical_tail_by_benchmark = (
        empirical_conflict_tail_macro_summary(
            rows, percentile_field="analysis_visual_percentile"
        )
    )
    tail, tail_by_benchmark = tail_macro_summary(rows)
    (
        intervals,
        bands,
        bootstrap_draw_counts,
        empirical_tail_intervals,
        empirical_tail_valid_draws,
    ) = clustered_bootstrap(
        rows,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
        percentile_field="analysis_visual_percentile",
    )
    evidence = {
        "positive_continuous_interaction": bool(
            intervals["delta_slope"][0] > 0.0
        ),
        "p10_u_favored": bool(intervals["delta_p10"][1] < 0.0),
        "p90_gc_favored": bool(intervals["delta_p90"][0] > 0.0),
    }
    evidence["strict_continuous_crossover"] = all(evidence.values())
    empirical_evidence = {
        "observed_low_u_favored": bool(
            empirical_tail_intervals.get("low_delta", [0.0, 0.0])[1] < 0.0
        ),
        "observed_high_gc_favored": bool(
            empirical_tail_intervals.get("high_delta", [0.0, 0.0])[0] > 0.0
        ),
        "observed_positive_interaction": bool(
            empirical_tail_intervals.get("interaction", [0.0, 0.0])[0] > 0.0
        ),
    }
    empirical_evidence["strict_observed_tail_crossover"] = all(
        empirical_evidence.values()
    )
    evidence.update(empirical_evidence)
    evidence["strict_joint_crossover"] = bool(
        evidence["strict_continuous_crossover"]
        and evidence["strict_observed_tail_crossover"]
    )
    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": args.analysis_split,
        "warning": (
            "This is confirmatory only when the rule and continuous estimand "
            "were frozen before inspecting an image-disjoint validation run."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "probe_audit": {
            "same_text_trajectory_ratio": same_text_ratio,
            "recomputed_vision_encoder_ratio": recomputed_vision_ratio,
        },
        "frozen_rule": {
            "visual_metric": args.visual_metric,
            "visual_metric_field": VISUAL_METRIC_FIELDS[args.visual_metric],
            "visual_tail_assignment": (
                "within-benchmark percentiles over "
                f"{args.visual_reference_population}"
            ),
            "visual_percentile_reference_population": (
                args.visual_reference_population
            ),
            "source_conflict_eligibility": (
                "both available; unique matched root budget >=2 and <=8; "
                "candidate overlap <=25%"
            ),
            "continuous_model": (
                "per-benchmark linear probability model on centered visual "
                "percentile; equal-benchmark coefficient average"
            ),
            "display_endpoints": [LOW_ENDPOINT, HIGH_ENDPOINT],
            "outcome": "equal-root-budget G/C-minus-U target-token hit",
        },
        "eligibility_audit": eligibility_audit,
        "visual_percentile_audit": percentile_audit,
        "tail_point_estimates": tail,
        "tail_by_benchmark": tail_by_benchmark,
        "continuous_point_estimates": continuous,
        "continuous_by_benchmark": continuous_by_benchmark,
        "empirical_conflict_tail_point_estimates": empirical_tail,
        "empirical_conflict_tail_by_benchmark": empirical_tail_by_benchmark,
        "empirical_conflict_tail_cluster_bootstrap_95_ci": (
            empirical_tail_intervals
        ),
        "empirical_conflict_tail_valid_all_benchmark_bootstrap_draws": (
            empirical_tail_valid_draws
        ),
        "cluster_bootstrap_95_ci": intervals,
        "continuous_prediction_bands_95": bands,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_draw_counts": bootstrap_draw_counts,
        "evidence_gate": evidence,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(payload, args.output_dir / "continuous_visual_interaction_triptych")
    print(
        json.dumps(
            {
                "continuous_point_estimates": continuous,
                "evidence_gate": evidence,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
