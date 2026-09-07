"""Diagnose source preference under a frozen visual-provenance regime.

Unlike an instantaneous counterfactual label, visual provenance carries a
consensus image-effect signal forward for a short token window.  The rule is
absolute and outcome-blind; U/G/C hits and accepted lengths are outcomes only.
This script is a development diagnostic, not a runtime router.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    BLUE,
    INK,
    MUTED,
    ORANGE,
    prepare,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    COUNTERFACTUAL_BANK_PROTOCOL,
)


EXPECTED_BENCHMARKS = {
    "MMSpec",
    "MMT-Bench",
    "SEEDBench",
    "ScienceQA",
    "OCRBench",
    "ChartQA",
    "MathVista",
    "TextVQA",
}
MEAN_JSD_FIELD = "visual_probe_mean_jsd"
WRONG_JSD_FIELD = "visual_probe_wrong_jsd"
LOOKBACK_TOKENS = 8
HALF_LIFE_TOKENS = 4.0
LOW_EQUIVALENCE_JSD = 0.005
HIGH_EFFECT_JSD = 0.020
MIN_DEVELOPMENT_STATES_PER_STRATUM = 5
METRICS = (
    "low_current_consensus_jsd",
    "high_current_consensus_jsd",
    "low_provenance_jsd",
    "high_provenance_jsd",
    "low_u_recall",
    "low_gc_recall",
    "high_u_recall",
    "high_gc_recall",
    "low_root_delta",
    "high_root_delta",
    "root_interaction",
    "low_u_matched_accept",
    "low_gc_matched_accept",
    "high_u_matched_accept",
    "high_gc_matched_accept",
    "low_accept_delta",
    "high_accept_delta",
    "accept_interaction",
    "low_fraction",
    "high_fraction",
    "ambiguous_fraction",
)


def _trajectory_key(row: dict) -> tuple:
    return (
        row.get("analysis_split", "all"),
        row["benchmark"],
        row["question_id"],
        row.get("choice_index", 0),
        row.get("turn_index", 0),
    )


def assign_visual_provenance_strata(rows: Sequence[dict]) -> dict:
    """Assign absolute low/high labels from current and recent visual JSD."""

    trajectories: dict[tuple, list[dict]] = defaultdict(list)
    invalid_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        row["visual_provenance_stratum"] = "ineligible"
        row["visual_current_consensus_jsd"] = None
        row["visual_current_union_jsd"] = None
        row["visual_provenance_consensus_jsd"] = None
        row["visual_provenance_union_jsd"] = None
        if row.get(MEAN_JSD_FIELD) is None or row.get(WRONG_JSD_FIELD) is None:
            invalid_reasons["missing_counterfactual_jsd"] += 1
            continue
        if int(row.get("output_position", -1)) < 0:
            invalid_reasons["missing_output_position"] += 1
            continue
        mean_jsd = max(0.0, float(row[MEAN_JSD_FIELD]))
        wrong_jsd = max(0.0, float(row[WRONG_JSD_FIELD]))
        row["visual_current_consensus_jsd"] = min(mean_jsd, wrong_jsd)
        row["visual_current_union_jsd"] = max(mean_jsd, wrong_jsd)
        trajectories[_trajectory_key(row)].append(row)

    for trajectory in trajectories.values():
        trajectory.sort(key=lambda row: (row["output_position"], row["iteration"]))
        for index, row in enumerate(trajectory):
            position = int(row["output_position"])
            consensus_candidates = []
            union_candidates = []
            for prior in reversed(trajectory[: index + 1]):
                distance = position - int(prior["output_position"])
                if distance < 0:
                    continue
                if distance > LOOKBACK_TOKENS:
                    break
                decay = 2.0 ** (-float(distance) / HALF_LIFE_TOKENS)
                consensus_candidates.append(
                    decay * float(prior["visual_current_consensus_jsd"])
                )
                union_candidates.append(
                    decay * float(prior["visual_current_union_jsd"])
                )
            row["visual_provenance_consensus_jsd"] = max(
                consensus_candidates, default=0.0
            )
            row["visual_provenance_union_jsd"] = max(
                union_candidates, default=0.0
            )

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    relabeled = 0
    instantaneous_low = 0
    eligible = 0
    for row in rows:
        if not row.get("frozen_eligible", False):
            continue
        if not row.get("visual_probe_full_top1_matches_target", False):
            row["visual_provenance_stratum"] = "invalid_full_anchor"
            invalid_reasons["invalid_full_anchor"] += 1
            continue
        if row.get("visual_provenance_consensus_jsd") is None:
            row["visual_provenance_stratum"] = "missing_signal"
            invalid_reasons["eligible_missing_signal"] += 1
            continue
        eligible += 1
        current_union = float(row["visual_current_union_jsd"])
        provenance_consensus = float(row["visual_provenance_consensus_jsd"])
        provenance_union = float(row["visual_provenance_union_jsd"])
        if provenance_union <= LOW_EQUIVALENCE_JSD:
            stratum = "low"
        elif provenance_consensus >= HIGH_EFFECT_JSD:
            stratum = "high"
        else:
            stratum = "ambiguous"
        row["visual_provenance_stratum"] = stratum
        counts[row["benchmark"]][stratum] += 1
        counts[row["benchmark"]]["eligible"] += 1
        if current_union <= LOW_EQUIVALENCE_JSD:
            instantaneous_low += 1
            if stratum == "high":
                relabeled += 1

    return {
        "uses_source_hit_outcomes": False,
        "uses_accepted_length_outcomes": False,
        "mean_jsd_field": MEAN_JSD_FIELD,
        "wrong_jsd_field": WRONG_JSD_FIELD,
        "lookback_tokens": LOOKBACK_TOKENS,
        "half_life_tokens": HALF_LIFE_TOKENS,
        "low_equivalence_jsd": LOW_EQUIVALENCE_JSD,
        "high_effect_jsd": HIGH_EFFECT_JSD,
        "eligible_states": eligible,
        "instantaneous_low_states": instantaneous_low,
        "instantaneous_low_relabeled_high": relabeled,
        "instantaneous_low_relabeled_high_fraction": (
            relabeled / instantaneous_low if instantaneous_low else 0.0
        ),
        "invalid_reasons": dict(invalid_reasons),
        "by_benchmark": {
            benchmark: dict(benchmark_counts)
            for benchmark, benchmark_counts in sorted(counts.items())
        },
    }


def summarize_benchmark(rows: Sequence[dict]) -> dict | None:
    low = [row for row in rows if row.get("visual_provenance_stratum") == "low"]
    high = [row for row in rows if row.get("visual_provenance_stratum") == "high"]
    ambiguous = [
        row for row in rows if row.get("visual_provenance_stratum") == "ambiguous"
    ]
    valid = low + high + ambiguous
    if not low or not high or not valid:
        return None

    def mean(group: Sequence[dict], field: str) -> float:
        return float(np.mean([float(row[field]) for row in group]))

    output = {
        "valid_states": len(valid),
        "low_states": len(low),
        "high_states": len(high),
        "ambiguous_states": len(ambiguous),
        "low_current_consensus_jsd": mean(low, "visual_current_consensus_jsd"),
        "high_current_consensus_jsd": mean(high, "visual_current_consensus_jsd"),
        "low_provenance_jsd": mean(low, "visual_provenance_consensus_jsd"),
        "high_provenance_jsd": mean(high, "visual_provenance_consensus_jsd"),
        "low_u_recall": mean(low, "frozen_u_hit"),
        "low_gc_recall": mean(low, "frozen_gc_hit"),
        "high_u_recall": mean(high, "frozen_u_hit"),
        "high_gc_recall": mean(high, "frozen_gc_hit"),
        "low_u_matched_accept": mean(low, "u_matched_accept"),
        "low_gc_matched_accept": mean(low, "gc_matched_accept"),
        "high_u_matched_accept": mean(high, "u_matched_accept"),
        "high_gc_matched_accept": mean(high, "gc_matched_accept"),
        "low_fraction": len(low) / len(valid),
        "high_fraction": len(high) / len(valid),
        "ambiguous_fraction": len(ambiguous) / len(valid),
    }
    output["low_root_delta"] = output["low_gc_recall"] - output["low_u_recall"]
    output["high_root_delta"] = output["high_gc_recall"] - output["high_u_recall"]
    output["root_interaction"] = output["high_root_delta"] - output["low_root_delta"]
    output["low_accept_delta"] = (
        output["low_gc_matched_accept"] - output["low_u_matched_accept"]
    )
    output["high_accept_delta"] = (
        output["high_gc_matched_accept"] - output["high_u_matched_accept"]
    )
    output["accept_interaction"] = (
        output["high_accept_delta"] - output["low_accept_delta"]
    )
    return output


def macro_summary(
    rows: Sequence[dict], *, minimum_states_per_stratum: int
) -> tuple[dict, dict, list[str]]:
    by_benchmark = {}
    insufficient = []
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_benchmark(
            [row for row in rows if row["benchmark"] == benchmark]
        )
        if (
            summary is None
            or summary["low_states"] < minimum_states_per_stratum
            or summary["high_states"] < minimum_states_per_stratum
        ):
            insufficient.append(benchmark)
        else:
            by_benchmark[benchmark] = summary
    if not by_benchmark:
        return {}, {}, insufficient
    point = {
        metric: float(np.mean([summary[metric] for summary in by_benchmark.values()]))
        for metric in METRICS
    }
    return point, by_benchmark, insufficient


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int
) -> tuple[dict[str, list[float]], int]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    draws = {metric: [] for metric in METRICS}
    valid_draws = 0
    for _ in range(int(resamples)):
        summaries = []
        for benchmark in sorted(grouped):
            clusters = sorted(grouped[benchmark])
            sampled_indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in sampled_indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            summary = summarize_benchmark(sampled)
            if summary is None:
                break
            summaries.append(summary)
        if len(summaries) != len(grouped):
            continue
        valid_draws += 1
        for metric in METRICS:
            draws[metric].append(
                float(np.mean([summary[metric] for summary in summaries]))
            )
    intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
        if values
    }
    return intervals, valid_draws


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
    width = 0.32
    labels = ["Provenance-low", "Provenance-high"]

    for offset, names, color, label in (
        (
            -width / 2,
            ("low_current_consensus_jsd", "high_current_consensus_jsd"),
            "#9AA4AF",
            "Current token",
        ),
        (
            width / 2,
            ("low_provenance_jsd", "high_provenance_jsd"),
            "#4C9FBE",
            "Recent provenance",
        ),
    ):
        values = [point[name] for name in names]
        axes[0].bar(
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
    axes[0].axhline(HIGH_EFFECT_JSD, color="#6A737D", lw=0.8, ls=(0, (3, 2)))
    axes[0].set_ylabel("Consensus image-effect JSD")
    axes[0].set_title("(a) Visual provenance", loc="left", weight="bold")
    axes[0].legend(loc="upper left", frameon=False, fontsize=5.2)

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
        max(abs(intervals[name][edge]) for name in names for edge in (0, 1))
        * 1.15,
    )
    axes[2].set_ylim(-bound, bound)
    axes[2].set_ylabel("G/C − U accepted tokens")
    axes[2].set_title("(c) Accepted-token gain", loc="left", weight="bold")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9DEE4", linewidth=0.55, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=3, labelsize=6.0)
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
    )
    fig.suptitle(
        "Visual evidence can persist after the current token becomes text-driven",
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
        f"DEVELOPMENT DIAGNOSTIC · {len(payload['benchmarks'])} benchmarks · "
        "MME excluded · absolute thresholds",
        ha="left",
        va="center",
        fontsize=5.9,
        weight="bold",
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "Eight-token lookback; four-token half-life; matched root/tree budget; "
        "image-cluster bootstrap 95% CI.",
        ha="right",
        va="bottom",
        fontsize=5.3,
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
    parser.add_argument("--seed", type=int, default=81173)
    args = parser.parse_args()

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != EXPECTED_BENCHMARKS:
        parser.error(
            "expected exactly the eight non-MME benchmarks, got " + repr(benchmarks)
        )
    protocols = sorted({str(row["visual_probe_protocol"]) for row in rows})
    if protocols != [COUNTERFACTUAL_BANK_PROTOCOL]:
        parser.error(f"unexpected visual probe protocols: {protocols}")
    if not all(row.get("visual_probe_same_text_trajectory") for row in rows):
        parser.error("counterfactual probes did not preserve every trajectory")
    if not all(row.get("visual_probe_recomputed_vision_encoder") for row in rows):
        parser.error("counterfactual probes did not recompute every vision input")

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    stratum_audit = assign_visual_provenance_strata(rows)
    point, by_benchmark, insufficient = macro_summary(
        rows,
        minimum_states_per_stratum=MIN_DEVELOPMENT_STATES_PER_STRATUM,
    )
    support_complete = not insufficient and set(by_benchmark) == EXPECTED_BENCHMARKS
    intervals, valid_draws = clustered_bootstrap(
        rows, resamples=args.bootstrap_resamples, seed=args.seed
    )
    directional = bool(
        support_complete
        and point.get("low_root_delta", 0.0) < 0.0
        and point.get("high_root_delta", 0.0) > 0.0
        and point.get("root_interaction", 0.0) > 0.0
        and point.get("low_accept_delta", 0.0) < 0.0
        and point.get("high_accept_delta", 0.0) > 0.0
        and point.get("accept_interaction", 0.0) > 0.0
    )
    payload = {
        "schema_version": 1,
        "analysis_role": "development_diagnostic",
        "warning": (
            "This rule is frozen only for retrospective development diagnosis; "
            "it is not independent confirmatory evidence."
        ),
        "input_paths": [str(path) for path in paths],
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME"],
        "num_states": len(rows),
        "num_image_clusters": len({row["cluster_id"] for row in rows}),
        "visual_probe_protocols": protocols,
        "frozen_rule": {
            "mean_jsd_field": MEAN_JSD_FIELD,
            "wrong_jsd_field": WRONG_JSD_FIELD,
            "lookback_tokens": LOOKBACK_TOKENS,
            "half_life_tokens": HALF_LIFE_TOKENS,
            "low_equivalence_jsd": LOW_EQUIVALENCE_JSD,
            "high_effect_jsd": HIGH_EFFECT_JSD,
            "minimum_states_per_stratum": MIN_DEVELOPMENT_STATES_PER_STRATUM,
            "uses_future_states": False,
            "uses_source_hit_outcomes": False,
            "uses_accepted_length_outcomes": False,
        },
        "eligibility_audit": eligibility_audit,
        "stratum_audit": stratum_audit,
        "point_estimates": point,
        "by_benchmark": by_benchmark,
        "insufficient_support_benchmarks": insufficient,
        "support_complete": support_complete,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "development_directional_double_crossover": directional,
        "decision": (
            "development_directional_go" if directional else "development_no_go"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if support_complete and valid_draws:
        plot(payload, args.output_dir / "visual_provenance_triptych")
    print(
        json.dumps(
            {
                "point_estimates": point,
                "stratum_audit": stratum_audit,
                "insufficient_support_benchmarks": insufficient,
                "development_directional_double_crossover": directional,
                "decision": payload["decision"],
                "output": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
