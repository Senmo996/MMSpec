"""Evaluate a frozen visual-reliance × source-conflict reuse rule.

The rule was selected in development analysis and is intentionally hard-coded:

* visual reliance is the fraction of target-token surprisal attributable to
  the strongest true pixel-region occlusion;
* low/high tails are the within-benchmark bottom/top 20%;
* both U and G/C must expose at least two root candidates; and
* their matched candidate sets must overlap by at most 25%.

The response is the equal-root-budget target-token hit difference.  This is a
candidate-selection diagnostic, not an end-to-end speed measurement.

The default score is the original instantaneous target-token score.  A second,
explicitly offline score can average the current and next recorded proposal
states.  The latter is useful for diagnosing short trajectory segments, but it
must not be described as an online routing feature because it observes the next
proposal state.
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

from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)


TAIL_FRACTION = 0.20
MIN_ROOT_BUDGET = 2
MAX_ROOT_BUDGET = 8
MAX_OVERLAP_FRACTION = 0.25
DEFAULT_VISUAL_METRIC = "visual_target_drop_fraction"
VISUAL_METRICS = (
    DEFAULT_VISUAL_METRIC,
    "target_logprob_drop",
    "local2_mean_target_drop_fraction",
    "span2_mean_target_drop_fraction",
    "span2_mean_target_logprob_drop",
    "jsd",
    "local2_mean_jsd",
    "span2_mean_jsd",
)
VISUAL_METRIC_FIELDS = {
    "visual_target_drop_fraction": "visual_target_drop_fraction",
    "target_logprob_drop": "visual_probe_max_target_logprob_drop",
    "local2_mean_target_drop_fraction": (
        "visual_local2_mean_target_drop_fraction"
    ),
    "span2_mean_target_drop_fraction": (
        "visual_probe_span2_mean_target_drop_fraction"
    ),
    "span2_mean_target_logprob_drop": (
        "visual_probe_span2_mean_target_logprob_drop"
    ),
    "jsd": "visual_probe_jsd",
    "local2_mean_jsd": "visual_local2_mean_jsd",
    "span2_mean_jsd": "visual_probe_span2_mean_jsd",
}
VISUAL_METRIC_DESCRIPTIONS = {
    "visual_target_drop_fraction": (
        "target-logprob drop under {intervention} / "
        "(drop + full-view surprisal)"
    ),
    "target_logprob_drop": (
        "target-token log-probability loss under {intervention}"
    ),
    "local2_mean_target_drop_fraction": (
        "mean target-drop fraction over the current and next proposal states "
        "under {intervention} (offline trajectory diagnostic)"
    ),
    "span2_mean_target_drop_fraction": (
        "mean target-drop fraction over the next two exact generated tokens "
        "under {intervention} (offline trajectory diagnostic)"
    ),
    "span2_mean_target_logprob_drop": (
        "mean target-token log-probability loss over the next two exact "
        "generated tokens under {intervention} (offline trajectory diagnostic)"
    ),
    "jsd": "full/counterfactual prediction JSD under {intervention}",
    "local2_mean_jsd": (
        "mean full/counterfactual prediction JSD over the current and next "
        "proposal states under {intervention} (offline trajectory diagnostic)"
    ),
    "span2_mean_jsd": (
        "mean full/counterfactual prediction JSD over the next two exact "
        "generated tokens under {intervention} (offline trajectory diagnostic)"
    ),
    "whole_image_top1_change": (
        "binary next-token top-1 change under {intervention} "
        "(same top-1 = low; changed top-1 = high)"
    ),
}
ORANGE = "#E8792E"
BLUE = "#2864DC"
INK = "#17212B"
MUTED = "#66717D"


METRICS = (
    "low_u_availability",
    "low_gc_availability",
    "high_u_availability",
    "high_gc_availability",
    "low_u_recall",
    "low_gc_recall",
    "high_u_recall",
    "high_gc_recall",
    "low_delta",
    "high_delta",
    "interaction",
)


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


def add_visual_features(rows: Sequence[dict]) -> None:
    """Add instantaneous and two-proposal-state visual features in-place."""

    trajectories: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        drop = max(float(row["visual_probe_max_target_logprob_drop"]), 0.0)
        surprisal = max(
            -float(row.get("visual_probe_full_target_logprob", 0.0)), 0.0
        )
        row["visual_target_drop_fraction"] = float(
            drop / (drop + surprisal + 1e-8)
        )
        trajectories[
            (
                row.get("analysis_split", "all"),
                row["benchmark"],
                row["question_id"],
                row.get("choice_index", 0),
                row.get("turn_index", 0),
            )
        ].append(row)

    for trajectory in trajectories.values():
        trajectory.sort(key=lambda row: (row["output_position"], row["iteration"]))
        for index, row in enumerate(trajectory):
            local = trajectory[index : index + 2]
            row["visual_local2_mean_target_drop_fraction"] = float(
                np.mean(
                    [item["visual_target_drop_fraction"] for item in local]
                )
            )
            row["visual_local2_mean_jsd"] = float(
                np.mean([float(item["visual_probe_jsd"]) for item in local])
            )


def prepare(
    rows: Sequence[dict], *, visual_metric: str = DEFAULT_VISUAL_METRIC
) -> dict:
    if visual_metric not in VISUAL_METRIC_FIELDS:
        raise ValueError(f"unsupported visual metric: {visual_metric}")
    add_visual_features(rows)
    visual_field = VISUAL_METRIC_FIELDS[visual_metric]
    missing_visual_scores = sum(
        row.get(visual_field) is None for row in rows
    )
    if missing_visual_scores:
        raise ValueError(
            f"visual metric {visual_metric} is absent from "
            f"{missing_visual_scores}/{len(rows)} states"
        )
    audit = defaultdict(int)
    by_benchmark: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row["frozen_visual_score"] = float(row[visual_field])
        by_benchmark[row["benchmark"]].append(index)
    for indices in by_benchmark.values():
        percentiles = average_tie_percentiles(
            np.asarray(
                [rows[index]["frozen_visual_score"] for index in indices],
                dtype=float,
            )
        )
        for index, percentile in zip(indices, percentiles.tolist()):
            rows[index]["frozen_visual_percentile"] = float(percentile)

    for row in rows:
        audit["input_states"] += 1
        u = [int(token) for token in row.get("u_root_candidate_token_ids", [])][
            :MAX_ROOT_BUDGET
        ]
        gc = [int(token) for token in row.get("gc_root_candidate_token_ids", [])][
            :MAX_ROOT_BUDGET
        ]
        budget = min(len(u), len(gc))
        row["frozen_eligible"] = False
        if not row["u_available"] or not row["gc_available"]:
            audit["one_or_both_sources_unavailable"] += 1
            continue
        if budget < MIN_ROOT_BUDGET:
            audit["matched_root_budget_lt_2"] += 1
            continue
        u = u[:budget]
        gc = gc[:budget]
        if len(set(u)) != budget or len(set(gc)) != budget:
            audit["duplicate_candidate_row"] += 1
            continue
        overlap = len(set(u) & set(gc))
        overlap_fraction = overlap / budget
        if overlap_fraction > MAX_OVERLAP_FRACTION:
            audit["candidate_overlap_gt_25pct"] += 1
            continue
        target = int(row["target_token_id"])
        row["frozen_eligible"] = True
        row["frozen_root_budget"] = budget
        row["frozen_overlap_fraction"] = overlap_fraction
        row["frozen_u_hit"] = int(target in u)
        row["frozen_gc_hit"] = int(target in gc)
        row["frozen_delta"] = row["frozen_gc_hit"] - row["frozen_u_hit"]
        audit["eligible_states"] += 1
    audit["eligible_ratio"] = (
        audit["eligible_states"] / audit["input_states"]
        if audit["input_states"]
        else 0.0
    )
    return dict(audit)


def is_low(row: dict) -> bool:
    return float(row["frozen_visual_percentile"]) < TAIL_FRACTION


def is_high(row: dict) -> bool:
    return float(row["frozen_visual_percentile"]) >= 1.0 - TAIL_FRACTION


def summarize_benchmark(rows: Sequence[dict]) -> dict | None:
    low_all = [row for row in rows if is_low(row)]
    high_all = [row for row in rows if is_high(row)]
    low = [row for row in low_all if row["frozen_eligible"]]
    high = [row for row in high_all if row["frozen_eligible"]]
    if not low_all or not high_all or not low or not high:
        return None
    output = {
        "low_states": len(low_all),
        "high_states": len(high_all),
        "low_eligible_states": len(low),
        "high_eligible_states": len(high),
        "low_u_availability": float(np.mean([row["u_available"] for row in low_all])),
        "low_gc_availability": float(np.mean([row["gc_available"] for row in low_all])),
        "high_u_availability": float(
            np.mean([row["u_available"] for row in high_all])
        ),
        "high_gc_availability": float(
            np.mean([row["gc_available"] for row in high_all])
        ),
        "low_u_recall": float(np.mean([row["frozen_u_hit"] for row in low])),
        "low_gc_recall": float(np.mean([row["frozen_gc_hit"] for row in low])),
        "high_u_recall": float(np.mean([row["frozen_u_hit"] for row in high])),
        "high_gc_recall": float(np.mean([row["frozen_gc_hit"] for row in high])),
        "low_delta": float(np.mean([row["frozen_delta"] for row in low])),
        "high_delta": float(np.mean([row["frozen_delta"] for row in high])),
    }
    output["interaction"] = output["high_delta"] - output["low_delta"]
    return output


def macro_summary(rows: Sequence[dict]) -> tuple[dict, dict]:
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_benchmark(
            [row for row in rows if row["benchmark"] == benchmark]
        )
        if summary is not None:
            by_benchmark[benchmark] = summary
    if not by_benchmark:
        raise ValueError("no benchmark has eligible states in both visual tails")
    macro = {
        metric: float(np.mean([row[metric] for row in by_benchmark.values()]))
        for metric in METRICS
    }
    return macro, by_benchmark


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int
) -> dict[str, list[float]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    draws = {metric: [] for metric in METRICS}
    for _ in range(int(resamples)):
        summaries = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            summary = summarize_benchmark(sampled)
            if summary is not None:
                summaries.append(summary)
        if summaries:
            for metric in METRICS:
                draws[metric].append(float(np.mean([row[metric] for row in summaries])))
    return {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
    }


def _error(value: float, interval: Sequence[float]) -> list[list[float]]:
    return [[value - float(interval[0])], [float(interval[1]) - value]]


def plot(payload: dict, output_stem: Path) -> None:
    point = payload["point_estimates"]
    ci = payload["cluster_bootstrap_95_ci"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))
    x = np.arange(2)
    width = 0.32
    labels = payload.get(
        "strata_labels",
        ["Low visual\n(bottom 20%)", "High visual\n(top 20%)"],
    )

    def grouped(axis, u_metric_low, u_metric_high, gc_metric_low, gc_metric_high, ylabel):
        for offset, metrics, color, label in (
            (-width / 2, (u_metric_low, u_metric_high), ORANGE, "Persistent U"),
            (width / 2, (gc_metric_low, gc_metric_high), BLUE, "Request-local G/C"),
        ):
            values = [point[metric] for metric in metrics]
            errors = np.asarray(
                [
                    [values[i] - ci[metric][0] for i, metric in enumerate(metrics)],
                    [ci[metric][1] - values[i] for i, metric in enumerate(metrics)],
                ]
            )
            axis.bar(
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
        axis.set_ylim(0.0, 1.03)
        axis.set_ylabel(ylabel)

    grouped(
        axes[0],
        "low_u_availability",
        "high_u_availability",
        "low_gc_availability",
        "high_gc_availability",
        "Availability",
    )
    axes[0].set_title("(a) Source coverage", loc="left", weight="bold")
    grouped(
        axes[1],
        "low_u_recall",
        "high_u_recall",
        "low_gc_recall",
        "high_gc_recall",
        "Matched root recall",
    )
    axes[1].set_title("(b) Recall under conflict", loc="left", weight="bold")

    delta_values = [point["low_delta"], point["high_delta"]]
    delta_metrics = ("low_delta", "high_delta")
    delta_errors = np.asarray(
        [
            [delta_values[i] - ci[metric][0] for i, metric in enumerate(delta_metrics)],
            [ci[metric][1] - delta_values[i] for i, metric in enumerate(delta_metrics)],
        ]
    )
    axes[2].bar(
        x,
        delta_values,
        0.52,
        yerr=delta_errors,
        capsize=2.8,
        color=[ORANGE, BLUE],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    axes[2].axhline(0.0, color="#5D6670", linestyle=(0, (3, 2)), lw=0.9)
    bound = max(0.15, max(abs(ci["low_delta"][0]), abs(ci["high_delta"][1])) * 1.25)
    axes[2].set_ylim(-bound, bound)
    axes[2].set_ylabel("Δ root recall (G/C − U)")
    axes[2].set_title("(c) Source preference", loc="left", weight="bold")
    for index, value in enumerate(delta_values):
        axes[2].text(
            index,
            value + (0.025 * bound if value >= 0 else -0.025 * bound),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.7,
            weight="bold",
            color=INK,
        )

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3)
        axis.title.set_fontsize(8.0)
    role = payload["analysis_role"].upper().replace("_", " ")
    fig.suptitle(
        payload.get(
            "figure_title", "Frozen visual-regime source-conflict test"
        ),
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
        f"Visual score: {visual_description}; "
        "error bars: image-cluster bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.77, bottom=0.24, wspace=0.56)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--analysis-split", choices=("discovery", "heldout", "all"), required=True)
    parser.add_argument(
        "--analysis-role",
        choices=(
            "development",
            "retrospective_heldout",
            "retrospective_independent_pilot",
            "new_validation",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--visual-metric",
        choices=VISUAL_METRICS,
        default=DEFAULT_VISUAL_METRIC,
        help=(
            "Frozen visual-regime score. local2 variants are offline "
            "trajectory diagnostics and are not valid online gates."
        ),
    )
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
        parser.error(
            "all records must share the teacher-forced text path and rerun "
            "the vision encoder"
        )
    audit = prepare(rows, visual_metric=args.visual_metric)
    point, by_benchmark = macro_summary(rows)
    intervals = clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    evidence = {
        "low_u_favored": bool(intervals["low_delta"][1] < 0.0),
        "high_gc_favored": bool(intervals["high_delta"][0] > 0.0),
        "positive_interaction": bool(intervals["interaction"][0] > 0.0),
    }
    evidence["strict_crossover"] = all(evidence.values())
    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": args.analysis_split,
        "warning": (
            "This is confirmatory only when analysis_role is new_validation "
            "and the images were not used during rule selection."
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
            "full_view_top1_matches_target_ratio": float(
                np.mean(
                    [bool(row["visual_probe_full_top1_matches_target"]) for row in rows]
                )
            ),
        },
        "frozen_rule": {
            "visual_metric": args.visual_metric,
            "visual_metric_field": VISUAL_METRIC_FIELDS[args.visual_metric],
            "uses_future_proposal_state": args.visual_metric.startswith("local2_"),
            "uses_future_generated_token": args.visual_metric.startswith("span2_"),
            "online_routing_feature": not args.visual_metric.startswith(
                ("local2_", "span2_")
            ),
            "visual_tail_assignment": "within-benchmark percentiles",
            "tail_fraction": TAIL_FRACTION,
            "minimum_matched_root_budget": MIN_ROOT_BUDGET,
            "maximum_root_budget": MAX_ROOT_BUDGET,
            "maximum_candidate_overlap_fraction": MAX_OVERLAP_FRACTION,
            "outcome": "equal-root-budget G/C-minus-U target-token hit",
            "eligibility_is_outcome_independent": True,
        },
        "audit": audit,
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "evidence_gate": evidence,
        "by_benchmark": by_benchmark,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(payload, args.output_dir / "selective_conflict_triptych")
    print(json.dumps({"point_estimates": point, "evidence_gate": evidence}, indent=2))


if __name__ == "__main__":
    main()
