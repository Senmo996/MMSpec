"""Direct low/high tails after outcome-independent source-conflict filtering."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.analyze_frozen_selective_conflict_regime import (  # noqa: E402
    MAX_OVERLAP_FRACTION,
    MAX_ROOT_BUDGET,
    METRICS,
    MIN_ROOT_BUDGET,
    VISUAL_METRIC_DESCRIPTIONS,
    VISUAL_METRIC_FIELDS,
    plot,
    prepare,
    summarize_benchmark,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
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
EXPECTED_PROTOCOL = "teacher_forced_whole_image_mean_ablation_v1"
VISUAL_METRIC = "span2_mean_target_drop_fraction"
TAIL_FRACTION = 0.20
BOOTSTRAP_SEED = 424243


def assign_conflict_reference_percentiles(rows: Sequence[dict]) -> dict:
    """Map every state through the eligible-conflict empirical score CDF."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("frozen_eligible", False):
            grouped[row["benchmark"]].append(row)
    counts = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        reference = np.sort(
            np.asarray(
                [
                    float(row["frozen_visual_score"])
                    for row in grouped.get(benchmark, [])
                ],
                dtype=np.float64,
            )
        )
        if len(reference) == 0:
            raise ValueError(f"no eligible conflict states for {benchmark}")
        counts[benchmark] = int(len(reference))
        for row in rows:
            if row["benchmark"] != benchmark:
                continue
            value = float(row["frozen_visual_score"])
            left = int(np.searchsorted(reference, value, side="left"))
            right = int(np.searchsorted(reference, value, side="right"))
            row["frozen_visual_percentile"] = float(
                (left + right) / (2.0 * len(reference))
            )
    return {
        "reference_population": "eligible_conflicts",
        "uses_source_hit_outcomes": False,
        "states_by_benchmark": counts,
        "num_reference_states": int(sum(counts.values())),
    }


def macro_summary(
    rows: Sequence[dict], *, minimum_tail_states: int
) -> tuple[dict, dict]:
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_benchmark(
            [row for row in rows if row["benchmark"] == benchmark]
        )
        if (
            summary is not None
            and int(summary["low_eligible_states"]) >= minimum_tail_states
            and int(summary["high_eligible_states"]) >= minimum_tail_states
        ):
            by_benchmark[benchmark] = summary
    if set(by_benchmark) != EXPECTED_BENCHMARKS:
        missing = sorted(EXPECTED_BENCHMARKS - set(by_benchmark))
        raise ValueError(
            "conflict-conditioned tails lack support for benchmarks: "
            + ", ".join(missing)
        )
    return (
        {
            metric: float(
                np.mean([summary[metric] for summary in by_benchmark.values()])
            )
            for metric in METRICS
        },
        by_benchmark,
    )


def clustered_bootstrap(
    rows: Sequence[dict], *, resamples: int, seed: int, minimum_valid_fraction: float
) -> tuple[dict, int]:
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
            indices = rng.integers(0, len(clusters), size=len(clusters))
            sampled = [
                row
                for index in indices.tolist()
                for row in grouped[benchmark][clusters[index]]
            ]
            summary = summarize_benchmark(sampled)
            if summary is not None:
                summaries.append(summary)
        if len(summaries) != len(EXPECTED_BENCHMARKS):
            continue
        valid_draws += 1
        for metric in METRICS:
            draws[metric].append(
                float(np.mean([summary[metric] for summary in summaries]))
            )
    required = max(100, int(float(minimum_valid_fraction) * int(resamples)))
    if valid_draws < required:
        raise ValueError(
            f"only {valid_draws}/{resamples} bootstrap draws retained all benchmarks; "
            f"require {required}"
        )
    return (
        {
            metric: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for metric, values in draws.items()
        },
        valid_draws,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--analysis-role", choices=("development", "secondary_screening", "new_validation"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
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
    if protocols != [EXPECTED_PROTOCOL]:
        parser.error(f"unexpected visual probe protocols: {protocols}")
    same_text_ratio = float(
        np.mean([bool(row["visual_probe_same_text_trajectory"]) for row in rows])
    )
    recomputed_vision_ratio = float(
        np.mean([bool(row["visual_probe_recomputed_vision_encoder"]) for row in rows])
    )
    if same_text_ratio != 1.0 or recomputed_vision_ratio != 1.0:
        parser.error("visual probes failed same-path/recomputed-vision audit")

    eligibility_audit = prepare(rows, visual_metric=VISUAL_METRIC)
    percentile_audit = assign_conflict_reference_percentiles(rows)
    minimum = 3 if args.analysis_role == "development" else 20
    try:
        point, by_benchmark = macro_summary(
            rows, minimum_tail_states=minimum
        )
        intervals, valid_draws = clustered_bootstrap(
            rows,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
            minimum_valid_fraction=(
                0.0 if args.analysis_role == "development" else 0.90
            ),
        )
    except ValueError as error:
        payload = {
            "schema_version": 1,
            "analysis_role": args.analysis_role,
            "analysis_split": "all",
            "decision": "no_go_insufficient_support",
            "support_failure": str(error),
            "benchmarks": benchmarks,
            "excluded_benchmarks": ["MME"],
            "visual_probe_protocols": protocols,
            "eligibility_audit": eligibility_audit,
            "visual_percentile_audit": percentile_audit,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    evidence = {
        "low_u_favored": bool(intervals["low_delta"][1] < 0.0),
        "high_gc_favored": bool(intervals["high_delta"][0] > 0.0),
        "positive_interaction": bool(intervals["interaction"][0] > 0.0),
    }
    evidence["strict_crossover"] = all(evidence.values())
    directional = bool(
        point["low_delta"] < 0.0
        and point["high_delta"] > 0.0
        and point["interaction"] > 0.0
    )
    if args.analysis_role == "development":
        decision = "go_to_independent_confirmation" if directional else "development_no_go"
    elif args.analysis_role == "new_validation":
        decision = "validated" if evidence["strict_crossover"] else "validation_no_go"
    else:
        decision = "screen_pass" if evidence["strict_crossover"] else "screen_no_go"
    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": "all",
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
        "eligibility_audit": eligibility_audit,
        "visual_percentile_audit": percentile_audit,
        "frozen_rule": {
            "visual_metric": VISUAL_METRIC,
            "visual_metric_field": VISUAL_METRIC_FIELDS[VISUAL_METRIC],
            "visual_metric_description": VISUAL_METRIC_DESCRIPTIONS[VISUAL_METRIC],
            "visual_tail_assignment": "bottom/top 20% over eligible conflicts",
            "tail_fraction": TAIL_FRACTION,
            "minimum_tail_states_per_benchmark": minimum,
            "minimum_matched_root_budget": MIN_ROOT_BUDGET,
            "maximum_root_budget": MAX_ROOT_BUDGET,
            "maximum_candidate_overlap_fraction": MAX_OVERLAP_FRACTION,
            "uses_source_hit_outcomes": False,
            "outcome": "equal-root-budget G/C-minus-U target-token hit",
        },
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "development_directional_crossover": directional,
        "evidence_gate": evidence,
        "decision": decision,
        "by_benchmark": by_benchmark,
        "strata_labels": [
            "Low visual\n(conflict bottom 20%)",
            "High visual\n(conflict top 20%)",
        ],
        "figure_title": "Conflict-conditioned visual-tail source test",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(payload, args.output_dir / "conflict_conditioned_tail_triptych")
    print(
        json.dumps(
            {
                "point_estimates": point,
                "development_directional_crossover": directional,
                "evidence_gate": evidence,
                "decision": decision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
