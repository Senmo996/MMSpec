"""Show that long reuse paths concentrate on visually stable attempted roots.

This is a descriptive problem diagnostic over actual speculative-decoding
draft attempts, including attempts whose root is rejected.  Root visuality is
measured with teacher-forced mean-image and matched-wrong-image
counterfactuals; acceptance is read only after visual features and labels have
been constructed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_selective_reuse import (  # noqa: E402
    _benchmark_name,
    _read_jsonl,
    discover_result_paths,
)
from evaluation.analyze_candidate_alignment_quantiles import (  # noqa: E402
    GAIN_COLOR as VISUAL_COLOR,
    GC_EDGE as FLIP_EDGE,
    GC_FILL as FLIP_FILL,
    LEFT_AXIS_COLOR,
    RIGHT_AXIS_COLOR,
    _rounded_bars,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    COUNTERFACTUAL_BANK_PROTOCOL,
)


EXPECTED_BENCHMARKS = (
    "MMT-Bench",
    "SEEDBench",
    "ScienceQA",
    "OCRBench",
    "ChartQA",
    "MathVista",
    "TextVQA",
)
LENGTH_BINS = ("0", "1", "2", "3", "4", "5+")
LENGTH_LABELS = ("0\n(rejected)", "1", "2", "3", "4", "5+")
STABLE = "image_stable"
ONE_FLIP = "one_control_flip"
DUAL_FLIP = "dual_control_flip"

INK = "#25323B"
FONT_FAMILY = "Liberation Sans"
VISUAL_INTERIOR_COLOR = "white"
FLIP_EDGE_WIDTH = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_bin(accept_length: int) -> str:
    if int(accept_length) < 0:
        raise ValueError("draft-attempt analysis requires accept_length >= 0")
    if int(accept_length) == 0:
        return "0"
    if int(accept_length) <= 4:
        return str(int(accept_length))
    return "5+"


def _flip_class(disagreement_rate: float) -> str:
    value = float(disagreement_rate)
    if np.isclose(value, 0.0, atol=1e-8):
        return STABLE
    if np.isclose(value, 0.5, atol=1e-8):
        return ONE_FLIP
    if np.isclose(value, 1.0, atol=1e-8):
        return DUAL_FLIP
    raise ValueError(f"unexpected two-control disagreement rate: {value}")


def load_root_events(paths: Sequence[Path]) -> tuple[list[dict], dict]:
    rows = []
    seen = set()
    audit: dict[str, object] = defaultdict(int)
    all_clusters = set()
    protocols = set()
    for path in paths:
        for result in _read_jsonl(path):
            audit["result_records"] += 1
            benchmark = _benchmark_name(result, path)
            question_id = str(result.get("question_id"))
            image_cluster = str(result.get("image_cluster_id") or question_id)
            cluster_id = f"{benchmark}:{image_cluster}"
            all_clusters.add(cluster_id)
            analysis_split = str(result.get("analysis_split", "unsplit"))
            for choice in result.get("choices", []):
                choice_index = int(choice.get("index", 0))
                for turn_index, trace in enumerate(choice.get("policy_trace", [])):
                    for record in trace:
                        if not record.get("selective_reuse_diagnostics"):
                            continue
                        audit["diagnostic_states"] += 1
                        protocol = str(record.get("visual_probe_protocol", "missing"))
                        protocols.add(protocol)
                        if protocol != COUNTERFACTUAL_BANK_PROTOCOL:
                            audit["unexpected_protocol"] += 1
                            continue
                        if not record.get("visual_probe_same_text_trajectory", False):
                            audit["changed_text_trajectory"] += 1
                            continue
                        if not record.get("visual_probe_recomputed_vision_encoder", False):
                            audit["vision_not_recomputed"] += 1
                            continue
                        used_draft = int(record.get("used_draft_len", 0))
                        accepted = int(record.get("accept_len", 0))
                        if used_draft <= 0:
                            audit["no_draft"] += 1
                            continue
                        audit["drafted_states"] += 1
                        if accepted < 0:
                            audit["negative_accept_length"] += 1
                            continue
                        if accepted == 0:
                            audit["root_rejected"] += 1
                        if accepted > used_draft:
                            audit["accept_exceeds_used_draft"] += 1
                            continue
                        if not record.get("visual_probe_full_top1_matches_target", False):
                            audit["full_anchor_mismatch"] += 1
                            continue
                        required = (
                            record.get("visual_probe_mean_target_logprob_drop"),
                            record.get("visual_probe_wrong_target_logprob_drop"),
                            record.get("visual_probe_top1_disagreement_rate"),
                        )
                        if any(value is None for value in required):
                            audit["missing_visual_measure"] += 1
                            continue
                        mean_drop, wrong_drop, disagreement = map(float, required)
                        if not all(np.isfinite(value) for value in required):
                            audit["nonfinite_visual_measure"] += 1
                            continue
                        try:
                            flip_class = _flip_class(disagreement)
                        except ValueError:
                            audit["invalid_disagreement_rate"] += 1
                            continue
                        iteration = int(record.get("iteration", 0))
                        key = (
                            benchmark,
                            question_id,
                            choice_index,
                            turn_index,
                            iteration,
                        )
                        if key in seen:
                            raise ValueError(f"duplicate root event: {key!r}")
                        seen.add(key)
                        rows.append(
                            {
                                "benchmark": benchmark,
                                "cluster_id": cluster_id,
                                "question_id": question_id,
                                "analysis_split": analysis_split,
                                "choice_index": choice_index,
                                "turn_index": turn_index,
                                "iteration": iteration,
                                "output_position": int(
                                    record.get("selective_output_position", -1)
                                ),
                                "target_token_id": int(
                                    record.get("selective_target_token_id", -1)
                                ),
                                "used_draft_len": used_draft,
                                "accept_length": accepted,
                                "length_bin": length_bin(accepted),
                                "mean_image_target_logprob_drop": mean_drop,
                                "wrong_image_target_logprob_drop": wrong_drop,
                                "conservative_visual_dependence": min(
                                    mean_drop, wrong_drop
                                ),
                                "top1_disagreement_rate": disagreement,
                                "flip_class": flip_class,
                                "image_sensitive": flip_class != STABLE,
                            }
                        )
                        audit["included_root_events"] += 1
                        if accepted == 0:
                            audit["included_root_rejections"] += 1
    audit["all_image_clusters"] = len(all_clusters)
    audit["included_image_clusters"] = len({row["cluster_id"] for row in rows})
    audit["protocols"] = sorted(protocols)
    return rows, dict(audit)


def _mean(rows: Sequence[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize_benchmark(rows: Sequence[dict]) -> dict | None:
    output: dict[str, float | int] = {"num_events": len(rows)}
    for bin_name in LENGTH_BINS:
        group = [row for row in rows if row["length_bin"] == bin_name]
        if not group:
            return None
        prefix = "accept_" + bin_name.replace("+", "plus")
        output[f"{prefix}_count"] = len(group)
        output[f"{prefix}_visual_dependence"] = _mean(
            group, "conservative_visual_dependence"
        )
        stable_rate = float(np.mean([row["flip_class"] == STABLE for row in group]))
        one_rate = float(
            np.mean([row["flip_class"] == ONE_FLIP for row in group])
        )
        dual_rate = float(
            np.mean([row["flip_class"] == DUAL_FLIP for row in group])
        )
        output[f"{prefix}_stable_rate"] = stable_rate
        output[f"{prefix}_one_flip_rate"] = one_rate
        output[f"{prefix}_dual_flip_rate"] = dual_rate
        output[f"{prefix}_sensitive_rate"] = one_rate + dual_rate

    stable = [row for row in rows if not row["image_sensitive"]]
    sensitive = [row for row in rows if row["image_sensitive"]]
    if not stable or not sensitive:
        return None
    output["stable_count"] = len(stable)
    output["sensitive_count"] = len(sensitive)
    output["stable_mean_accept_length"] = _mean(stable, "accept_length")
    output["sensitive_mean_accept_length"] = _mean(sensitive, "accept_length")
    output["stable_short_rate"] = float(
        np.mean([row["accept_length"] == 1 for row in stable])
    )
    output["sensitive_short_rate"] = float(
        np.mean([row["accept_length"] == 1 for row in sensitive])
    )
    output["stable_reject_rate"] = float(
        np.mean([row["accept_length"] == 0 for row in stable])
    )
    output["sensitive_reject_rate"] = float(
        np.mean([row["accept_length"] == 0 for row in sensitive])
    )
    output["stable_long_rate"] = float(
        np.mean([row["accept_length"] >= 5 for row in stable])
    )
    output["sensitive_long_rate"] = float(
        np.mean([row["accept_length"] >= 5 for row in sensitive])
    )
    output["long_rate_relative_reduction"] = float(
        1.0 - output["sensitive_long_rate"] / output["stable_long_rate"]
    )
    output["short_long_visual_gap"] = float(
        output["accept_1_visual_dependence"]
        - output["accept_5plus_visual_dependence"]
    )
    output["short_long_sensitive_rate_gap"] = float(
        output["accept_1_sensitive_rate"] - output["accept_5plus_sensitive_rate"]
    )
    output["rejected_long_visual_gap"] = float(
        output["accept_0_visual_dependence"]
        - output["accept_5plus_visual_dependence"]
    )
    output["rejected_long_sensitive_rate_gap"] = float(
        output["accept_0_sensitive_rate"]
        - output["accept_5plus_sensitive_rate"]
    )
    return output


def summarize_macro(
    rows: Sequence[dict], expected_benchmarks: Iterable[str]
) -> tuple[dict, dict, list[str]]:
    expected = set(expected_benchmarks)
    by_benchmark = {}
    missing = []
    for benchmark in sorted(expected):
        summary = summarize_benchmark(
            [row for row in rows if row["benchmark"] == benchmark]
        )
        if summary is None:
            missing.append(benchmark)
        else:
            by_benchmark[benchmark] = summary
    if not by_benchmark:
        raise ValueError("no benchmark has complete problem-figure support")
    numeric_metrics = [
        key
        for key, value in next(iter(by_benchmark.values())).items()
        if isinstance(value, float)
    ]
    macro = {
        metric: float(
            np.mean([summary[metric] for summary in by_benchmark.values()])
        )
        for metric in numeric_metrics
    }
    macro["rejected_vs_long_flip_risk_ratio"] = float(
        macro["accept_0_sensitive_rate"]
        / macro["accept_5plus_sensitive_rate"]
    )
    macro["rejected_vs_long_visual_relative_decline"] = float(
        1.0
        - macro["accept_5plus_visual_dependence"]
        / macro["accept_0_visual_dependence"]
    )
    return macro, by_benchmark, missing


def clustered_bootstrap(
    rows: Sequence[dict],
    *,
    expected_benchmarks: Sequence[str],
    resamples: int,
    seed: int,
) -> tuple[dict, int]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["benchmark"]][row["cluster_id"]].append(row)
    rng = np.random.default_rng(int(seed))
    metric_names = [
        key
        for key in summarize_macro(rows, expected_benchmarks)[0]
        if key
        in {
            *(f"accept_{name.replace('+', 'plus')}_visual_dependence" for name in LENGTH_BINS),
            *(f"accept_{name.replace('+', 'plus')}_stable_rate" for name in LENGTH_BINS),
            *(f"accept_{name.replace('+', 'plus')}_one_flip_rate" for name in LENGTH_BINS),
            *(f"accept_{name.replace('+', 'plus')}_dual_flip_rate" for name in LENGTH_BINS),
            *(f"accept_{name.replace('+', 'plus')}_sensitive_rate" for name in LENGTH_BINS),
            "stable_mean_accept_length",
            "sensitive_mean_accept_length",
            "stable_short_rate",
            "sensitive_short_rate",
            "stable_reject_rate",
            "sensitive_reject_rate",
            "stable_long_rate",
            "sensitive_long_rate",
            "long_rate_relative_reduction",
            "short_long_visual_gap",
            "short_long_sensitive_rate_gap",
            "rejected_long_visual_gap",
            "rejected_long_sensitive_rate_gap",
            "rejected_vs_long_flip_risk_ratio",
            "rejected_vs_long_visual_relative_decline",
        }
    ]
    draws = {metric: [] for metric in metric_names}
    valid_draws = 0
    for _ in range(int(resamples)):
        sampled = []
        for benchmark in expected_benchmarks:
            clusters = sorted(grouped[benchmark])
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled.extend(
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            )
        point, _, missing = summarize_macro(sampled, expected_benchmarks)
        if missing:
            continue
        valid_draws += 1
        for metric in metric_names:
            draws[metric].append(point[metric])
    intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
    }
    return intervals, valid_draws


def _errors(values, names, intervals):
    return np.asarray(
        [
            [value - intervals[name][0] for value, name in zip(values, names)],
            [intervals[name][1] - value for value, name in zip(values, names)],
        ]
    )


def plot(
    payload: dict,
    output_stem: Path,
    figsize: tuple[float, float] = (7.16, 2.85),
    preserve_canvas: bool = False,
) -> None:
    point = payload["point_estimates"]
    intervals = payload["cluster_bootstrap_95_ci"]
    fig, flip_axis = plt.subplots(1, 1, figsize=figsize)
    visual_axis = flip_axis.twinx()
    x = np.arange(len(LENGTH_BINS))

    prefixes = [f"accept_{name.replace('+', 'plus')}" for name in LENGTH_BINS]
    flip_names = tuple(f"{prefix}_sensitive_rate" for prefix in prefixes)
    flip_values = [point[name] for name in flip_names]
    _rounded_bars(
        flip_axis,
        x,
        flip_values,
        0.58,
        FLIP_FILL,
        FLIP_EDGE,
        "Flip rate",
        rounding_size=0.0014,
        linewidth=FLIP_EDGE_WIDTH,
    )
    for position, value in zip(x, flip_values):
        flip_axis.text(
            position,
            0.006,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            weight="bold",
            color=FLIP_EDGE,
            zorder=6,
        )

    visual_names = tuple(
        f"accept_{name.replace('+', 'plus')}_visual_dependence"
        for name in LENGTH_BINS
    )
    visual_values = [point[name] for name in visual_names]
    visual_axis.errorbar(
        x,
        visual_values,
        yerr=_errors(visual_values, visual_names, intervals),
        color=VISUAL_COLOR,
        marker="o",
        markerfacecolor=VISUAL_INTERIOR_COLOR,
        markeredgecolor=VISUAL_COLOR,
        markeredgewidth=1.2,
        markersize=5.6,
        linewidth=2.35,
        elinewidth=1.0,
        capsize=2.8,
        solid_capstyle="round",
        zorder=4,
    )
    for position, name, value in zip(x, visual_names, visual_values):
        visual_axis.annotate(
            f"{value:.2f}",
            xy=(position, intervals[name][1]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
            weight="bold",
            color=VISUAL_COLOR,
            zorder=7,
            annotation_clip=False,
        )

    flip_axis.set_ylim(0.0, 0.42)
    flip_axis.set_yticks(np.arange(0.0, 0.41, 0.10))
    flip_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    flip_axis.set_ylabel(
        "Prediction change rate",
        color=LEFT_AXIS_COLOR,
        fontsize=9.0,
        fontweight="bold",
    )
    flip_axis.tick_params(axis="y", colors=LEFT_AXIS_COLOR)
    flip_axis.spines["left"].set_color(LEFT_AXIS_COLOR)
    flip_axis.spines["left"].set_linewidth(1.15)

    visual_axis.set_ylim(0.0, 0.70)
    visual_axis.set_yticks(np.arange(0.0, 0.71, 0.10))
    visual_axis.set_ylabel(
        "Visual dependence",
        color=RIGHT_AXIS_COLOR,
        fontsize=9.0,
        fontweight="bold",
    )
    visual_axis.tick_params(axis="y", colors=RIGHT_AXIS_COLOR)
    visual_axis.spines["right"].set_color(RIGHT_AXIS_COLOR)
    visual_axis.spines["right"].set_linewidth(1.15)

    flip_axis.set_xticks(x, LENGTH_LABELS)
    flip_axis.set_xlabel(
        "Accepted tokens", labelpad=3, fontsize=9.5, fontweight="bold"
    )
    flip_axis.set_xlim(-0.58, len(LENGTH_BINS) - 0.42)
    flip_axis.grid(
        axis="y",
        color="#DDE4F0",
        linewidth=0.75,
        linestyle=(0, (1.5, 2.5)),
        zorder=0,
    )
    flip_axis.set_axisbelow(True)
    flip_axis.spines["top"].set_visible(False)
    visual_axis.spines[["top", "left", "bottom"]].set_visible(False)
    flip_axis.spines["bottom"].set_color(INK)
    flip_axis.spines["bottom"].set_linewidth(1.1)
    flip_axis.tick_params(
        axis="x", length=3.0, width=0.9, color=INK, pad=3, labelsize=8.0, colors=INK
    )
    flip_axis.tick_params(axis="y", length=3.5, width=0.9, labelsize=7.5)
    visual_axis.tick_params(axis="y", labelsize=7.5)
    flip_axis.xaxis.set_label_coords(0.5, -0.1)
    tick_labels = flip_axis.get_xticklabels()
    for tick in (*tick_labels, *flip_axis.get_yticklabels(), *visual_axis.get_yticklabels()):
        tick.set_weight("bold")
    legend_handles = [
        Patch(facecolor=FLIP_FILL, edgecolor=FLIP_EDGE, linewidth=0.9),
        Line2D(
            [0],
            [0],
            color=VISUAL_COLOR,
            marker="o",
            markerfacecolor=VISUAL_INTERIOR_COLOR,
            markeredgecolor=VISUAL_COLOR,
            markeredgewidth=1.2,
            linewidth=2.35,
        ),
    ]
    fig.legend(
        legend_handles,
        ["Prediction change rate", "Visual dependence"],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.9),
        ncol=2,
        frameon=False,
        prop={"family": FONT_FAMILY, "size": 8.5, "weight": "bold"},
        labelcolor=INK,
        handlelength=1.45,
        columnspacing=1.5,
    )
    fig.subplots_adjust(
        left=0.105 if preserve_canvas else 0.082,
        right=0.918,
        top=0.82,
        bottom=0.17 if preserve_canvas else 0.25,
    )
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT_FAMILY)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    bbox_inches = None if preserve_canvas else "tight"
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=bbox_inches, transparent=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=320, bbox_inches=bbox_inches, transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=141421)
    parser.add_argument(
        "--expected-benchmarks", default=",".join(EXPECTED_BENCHMARKS)
    )
    args = parser.parse_args()

    expected = tuple(
        item.strip()
        for item in str(args.expected_benchmarks).split(",")
        if item.strip()
    )
    if set(expected) != set(EXPECTED_BENCHMARKS):
        parser.error(
            f"expected benchmarks must be exactly {sorted(EXPECTED_BENCHMARKS)}"
        )
    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows, audit = load_root_events(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != set(expected):
        parser.error(f"benchmark mismatch: got {benchmarks}")
    if audit.get("protocols") != [COUNTERFACTUAL_BANK_PROTOCOL]:
        parser.error(f"unexpected protocols: {audit.get('protocols')}")
    technical_failures = {
        key: int(audit.get(key, 0))
        for key in (
            "unexpected_protocol",
            "changed_text_trajectory",
            "vision_not_recomputed",
            "accept_exceeds_used_draft",
            "negative_accept_length",
            "missing_visual_measure",
            "nonfinite_visual_measure",
            "invalid_disagreement_rate",
        )
        if int(audit.get(key, 0)) > 0
    }
    if technical_failures:
        parser.error(f"technical audit failed: {technical_failures}")

    point, by_benchmark, missing = summarize_macro(rows, expected)
    if missing:
        parser.error(f"insufficient benchmark support: {missing}")
    intervals, valid_draws = clustered_bootstrap(
        rows,
        expected_benchmarks=expected,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    required_draws = max(100, int(0.90 * args.bootstrap_resamples))
    split_summaries = {}
    for split in ("discovery", "heldout"):
        split_point, _, split_missing = summarize_macro(
            [row for row in rows if row["analysis_split"] == split], expected
        )
        split_summaries[split] = {
            "point_estimates": split_point,
            "missing_benchmarks": split_missing,
        }

    visual_names = [
        f"accept_{name.replace('+', 'plus')}_visual_dependence"
        for name in LENGTH_BINS
    ]
    sensitive_names = [
        f"accept_{name.replace('+', 'plus')}_sensitive_rate"
        for name in LENGTH_BINS
    ]
    payload = {
        "schema_version": 1,
        "analysis_role": "post_hoc_descriptive_problem_figure",
        "warning": (
            "Association on attempted draft roots; not an independent confirmatory, "
            "semantic image-irrelevance, universal, or causal claim."
        ),
        "input_paths": [str(path) for path in paths],
        "policy": args.policy,
        "benchmarks": benchmarks,
        "excluded_benchmarks": ["MME", "MMSpec"],
        "num_input_image_clusters": int(audit["all_image_clusters"]),
        "num_included_image_clusters": int(audit["included_image_clusters"]),
        "num_root_events": len(rows),
        "eligibility_audit": audit,
        "frozen_descriptive_rule": {
            "actual_acceptance_bins": list(LENGTH_BINS),
            "requires_used_draft": True,
            "includes_rejected_root_attempts": True,
            "requires_full_image_top1_target_match": True,
            "visual_dependence": (
                "min(true-minus-mean, true-minus-matched-wrong target "
                "log-probability drop)"
            ),
            "image_sensitive": "either image counterfactual changes root top-1",
            "uses_acceptance_to_define_visuality": False,
            "equal_benchmark_weight": True,
        },
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "bootstrap_required_valid_draws": required_draws,
        "support_complete": valid_draws >= required_draws,
        "by_benchmark": by_benchmark,
        "split_robustness": split_summaries,
        "diagnostic_checks": {
            "visual_dependence_monotone_decreasing": all(
                point[left] > point[right]
                for left, right in zip(visual_names, visual_names[1:])
            ),
            "sensitive_rate_monotone_decreasing": all(
                point[left] > point[right]
                for left, right in zip(sensitive_names, sensitive_names[1:])
            ),
            "short_long_visual_gap_positive_benchmarks": sum(
                summary["short_long_visual_gap"] > 0.0
                for summary in by_benchmark.values()
            ),
            "short_long_sensitive_gap_positive_benchmarks": sum(
                summary["short_long_sensitive_rate_gap"] > 0.0
                for summary in by_benchmark.values()
            ),
            "rejected_long_visual_gap_positive_benchmarks": sum(
                summary["rejected_long_visual_gap"] > 0.0
                for summary in by_benchmark.values()
            ),
            "rejected_long_sensitive_gap_positive_benchmarks": sum(
                summary["rejected_long_sensitive_rate_gap"] > 0.0
                for summary in by_benchmark.values()
            ),
            "discovery_visual_dependence_monotone_decreasing": all(
                split_summaries["discovery"]["point_estimates"][left]
                > split_summaries["discovery"]["point_estimates"][right]
                for left, right in zip(visual_names, visual_names[1:])
            ),
            "heldout_visual_dependence_monotone_decreasing": all(
                split_summaries["heldout"]["point_estimates"][left]
                > split_summaries["heldout"]["point_estimates"][right]
                for left, right in zip(visual_names, visual_names[1:])
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_stem = args.output_dir / "accepted_length_visuality_problem"
    plot(payload, output_stem)
    manifest = {
        "schema_version": 1,
        "summary": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "figure_png": str(output_stem.with_suffix(".png").resolve()),
        "figure_png_sha256": _sha256(output_stem.with_suffix(".png")),
        "figure_pdf": str(output_stem.with_suffix(".pdf").resolve()),
        "figure_pdf_sha256": _sha256(output_stem.with_suffix(".pdf")),
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
