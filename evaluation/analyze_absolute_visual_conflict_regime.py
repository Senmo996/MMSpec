"""Screen an absolute visual-reliance source-conflict regime.

The historical relative-tail analysis can call a state "low visual" even when
its absolute visual-reliance score is large. This outcome-blind secondary
screen uses interpretable absolute bands selected on old development data.
"""

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
    DEFAULT_VISUAL_METRIC,
    MAX_OVERLAP_FRACTION,
    MAX_ROOT_BUDGET,
    METRICS,
    MIN_ROOT_BUDGET,
    VISUAL_METRIC_FIELDS,
    plot,
    prepare,
)
from evaluation.analyze_selective_reuse import (  # noqa: E402
    discover_result_paths,
    load_selective_records,
)


LOW_MAX = 0.30
HIGH_MIN = 0.70
MIN_TAIL_STATES_PER_BENCHMARK = 8
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


def summarize_absolute_benchmark(
    rows: Sequence[dict], *, minimum_tail_states: int
) -> dict | None:
    low_all = [
        row for row in rows if float(row["frozen_visual_score"]) <= LOW_MAX
    ]
    high_all = [
        row for row in rows if float(row["frozen_visual_score"]) >= HIGH_MIN
    ]
    low = [row for row in low_all if row["frozen_eligible"]]
    high = [row for row in high_all if row["frozen_eligible"]]
    if len(low) < minimum_tail_states or len(high) < minimum_tail_states:
        return None
    output = {
        "low_states": len(low_all),
        "high_states": len(high_all),
        "low_eligible_states": len(low),
        "high_eligible_states": len(high),
        "low_u_availability": float(np.mean([row["u_available"] for row in low_all])),
        "low_gc_availability": float(np.mean([row["gc_available"] for row in low_all])),
        "high_u_availability": float(np.mean([row["u_available"] for row in high_all])),
        "high_gc_availability": float(np.mean([row["gc_available"] for row in high_all])),
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
        summary = summarize_absolute_benchmark(
            [row for row in rows if row["benchmark"] == benchmark],
            minimum_tail_states=MIN_TAIL_STATES_PER_BENCHMARK,
        )
        if summary is not None:
            by_benchmark[benchmark] = summary
    if set(by_benchmark) != EXPECTED_BENCHMARKS:
        missing = sorted(EXPECTED_BENCHMARKS - set(by_benchmark))
        raise ValueError(
            "absolute visual bands lack frozen support for benchmarks: "
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
    rows: Sequence[dict], *, resamples: int, seed: int
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
            summary = summarize_absolute_benchmark(
                sampled, minimum_tail_states=1
            )
            if summary is not None:
                summaries.append(summary)
        if len(summaries) != len(EXPECTED_BENCHMARKS):
            continue
        valid_draws += 1
        for metric in METRICS:
            draws[metric].append(
                float(np.mean([summary[metric] for summary in summaries]))
            )
    if valid_draws < max(100, int(0.90 * resamples)):
        raise ValueError(
            f"only {valid_draws}/{resamples} bootstrap draws retained all benchmarks"
        )
    intervals = {
        metric: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for metric, values in draws.items()
    }
    return intervals, valid_draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--analysis-split", choices=("all",), default="all")
    parser.add_argument(
        "--analysis-role",
        choices=("outcome_blind_secondary_screening", "new_confirmation"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=57721)
    args = parser.parse_args()

    paths = discover_result_paths(args.results_roots, args.policy)
    if not paths:
        parser.error("no result JSONL files found")
    rows = load_selective_records(paths)
    benchmarks = sorted({row["benchmark"] for row in rows})
    if set(benchmarks) != EXPECTED_BENCHMARKS:
        parser.error(
            "expected exactly the eight non-MME benchmarks, got "
            + repr(benchmarks)
        )
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
        parser.error("visual probes failed same-path/recomputed-vision audit")

    audit = prepare(rows, visual_metric=DEFAULT_VISUAL_METRIC)
    try:
        point, by_benchmark = macro_summary(rows)
    except ValueError as error:
        payload = {
            "schema_version": 1,
            "analysis_role": args.analysis_role,
            "analysis_split": "all",
            "decision": "no_go_insufficient_support",
            "support_failure": str(error),
            "benchmarks": benchmarks,
            "excluded_benchmarks": ["MME"],
            "num_states": len(rows),
            "num_image_clusters": len({row["cluster_id"] for row in rows}),
            "visual_probe_protocols": protocols,
            "audit": audit,
            "frozen_rule": {
                "visual_metric": DEFAULT_VISUAL_METRIC,
                "visual_tail_assignment": "absolute score bands",
                "low_max": LOW_MAX,
                "high_min": HIGH_MIN,
                "minimum_tail_states_per_benchmark": (
                    MIN_TAIL_STATES_PER_BENCHMARK
                ),
            },
            "evidence_gate": {
                "low_u_favored": False,
                "high_gc_favored": False,
                "positive_interaction": False,
                "strict_crossover": False,
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"decision": payload["decision"], "error": str(error)}))
        return
    intervals, valid_draws = clustered_bootstrap(
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
        "analysis_split": "all",
        "warning": (
            "This is a secondary screen selected on previously inspected data. "
            "A strict result requires a later image-disjoint confirmation."
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
            "visual_metric": DEFAULT_VISUAL_METRIC,
            "visual_metric_field": VISUAL_METRIC_FIELDS[DEFAULT_VISUAL_METRIC],
            "visual_tail_assignment": "absolute score bands",
            "low_max": LOW_MAX,
            "high_min": HIGH_MIN,
            "minimum_tail_states_per_benchmark": MIN_TAIL_STATES_PER_BENCHMARK,
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
        "bootstrap_valid_all_benchmark_draws": valid_draws,
        "evidence_gate": evidence,
        "decision": (
            "screen_pass_requires_new_confirmation"
            if evidence["strict_crossover"]
            else "screen_no_go"
        ),
        "by_benchmark": by_benchmark,
        "strata_labels": [
            f"Low visual\n(V <= {LOW_MAX:.2f})",
            f"High visual\n(V >= {HIGH_MIN:.2f})",
        ],
        "figure_title": "Absolute visual-demand source-conflict test",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(payload, args.output_dir / "absolute_visual_conflict_triptych")
    print(json.dumps({"point_estimates": point, "evidence_gate": evidence}, indent=2))


if __name__ == "__main__":
    main()
