"""Analyze a threshold-free binary whole-image visual-necessity label.

Low visual means the next-token top-1 prediction is unchanged by whole-image
mean-content ablation; high visual means it changes.  U/G/C source hits are
never used to assign the label.  The analysis is diagnostic and outside decode
timing.
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
    MAX_OVERLAP_FRACTION,
    MAX_ROOT_BUDGET,
    METRICS,
    MIN_ROOT_BUDGET,
    plot,
    prepare,
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
BOOTSTRAP_SEED = 8675309


def assign_binary_label(rows: Sequence[dict]) -> dict:
    audit = defaultdict(int)
    for row in rows:
        audit["input_states"] += 1
        row["binary_visual_label"] = None
        if not bool(row.get("visual_probe_full_top1_matches_target", False)):
            audit["full_view_top1_mismatch"] += 1
            continue
        value = float(row["visual_probe_top1_disagreement_rate"])
        if np.isclose(value, 0.0, rtol=0.0, atol=1e-8):
            row["binary_visual_label"] = "low"
            audit["low_visual_states"] += 1
        elif np.isclose(value, 1.0, rtol=0.0, atol=1e-8):
            row["binary_visual_label"] = "high"
            audit["high_visual_states"] += 1
        else:
            audit["nonbinary_disagreement"] += 1
    if audit["nonbinary_disagreement"]:
        raise ValueError(
            "whole-image probe produced nonbinary top-1 disagreement for "
            f"{audit['nonbinary_disagreement']} states"
        )
    return dict(audit)


def summarize_binary_benchmark(
    rows: Sequence[dict], *, minimum_stratum_states: int
) -> dict | None:
    low_all = [row for row in rows if row["binary_visual_label"] == "low"]
    high_all = [row for row in rows if row["binary_visual_label"] == "high"]
    low = [row for row in low_all if row["frozen_eligible"]]
    high = [row for row in high_all if row["frozen_eligible"]]
    if (
        len(low) < int(minimum_stratum_states)
        or len(high) < int(minimum_stratum_states)
    ):
        return None
    output = {
        "low_states": len(low_all),
        "high_states": len(high_all),
        "low_eligible_states": len(low),
        "high_eligible_states": len(high),
        "low_u_availability": float(np.mean([row["u_available"] for row in low_all])),
        "low_gc_availability": float(
            np.mean([row["gc_available"] for row in low_all])
        ),
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


def macro_summary(
    rows: Sequence[dict], *, minimum_stratum_states: int
) -> tuple[dict, dict]:
    by_benchmark = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        summary = summarize_binary_benchmark(
            [row for row in rows if row["benchmark"] == benchmark],
            minimum_stratum_states=minimum_stratum_states,
        )
        if summary is not None:
            by_benchmark[benchmark] = summary
    if set(by_benchmark) != EXPECTED_BENCHMARKS:
        missing = sorted(EXPECTED_BENCHMARKS - set(by_benchmark))
        raise ValueError(
            "binary visual strata lack support for benchmarks: "
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
            summary = summarize_binary_benchmark(
                sampled, minimum_stratum_states=1
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
    required_valid_draws = max(
        100, int(float(minimum_valid_fraction) * int(resamples))
    )
    if valid_draws < required_valid_draws:
        raise ValueError(
            f"only {valid_draws}/{resamples} bootstrap draws retained all benchmarks; "
            f"require {required_valid_draws}"
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
        "--analysis-role", choices=("development", "new_validation"), required=True
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

    eligibility_audit = prepare(
        rows, visual_metric="span2_mean_target_drop_fraction"
    )
    label_audit = assign_binary_label(rows)
    minimum = 3 if args.analysis_role == "development" else 20
    try:
        point, by_benchmark = macro_summary(
            rows, minimum_stratum_states=minimum
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
            "label_audit": label_audit,
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
    else:
        decision = "validated" if evidence["strict_crossover"] else "validation_no_go"
    payload = {
        "schema_version": 1,
        "analysis_role": args.analysis_role,
        "analysis_split": "all",
        "warning": (
            "Development is directional screening only; only an independently "
            "frozen new_validation run can support the strict claim."
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
        "eligibility_audit": eligibility_audit,
        "label_audit": label_audit,
        "frozen_rule": {
            "visual_metric": "whole_image_top1_change",
            "low_visual": "full-view top1 equals target and ablated top1 is unchanged",
            "high_visual": "full-view top1 equals target and ablated top1 changes",
            "uses_source_hit_outcomes": False,
            "minimum_stratum_states_per_benchmark": minimum,
            "minimum_matched_root_budget": MIN_ROOT_BUDGET,
            "maximum_root_budget": MAX_ROOT_BUDGET,
            "maximum_candidate_overlap_fraction": MAX_OVERLAP_FRACTION,
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
            "Low visual\n(top-1 unchanged)",
            "High visual\n(top-1 changed)",
        ],
        "figure_title": "Counterfactual visual-necessity source-conflict test",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(payload, args.output_dir / "binary_visual_necessity_triptych")
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
