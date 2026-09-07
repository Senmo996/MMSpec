"""Completion audit for unequal-count fresh visual-reliability confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.audit_consensus_counterfactual_run import (
    DEFAULT_POLICY,
    EXPECTED_PROTOCOL,
    LAYOUT,
    REQUIRED_SUMMARY_FIELDS,
    _question_ids,
    audit_counterfactual_traces,
    read_json,
    read_jsonl,
)


def audit(args) -> dict:
    failures = []
    benchmark_reports = {}
    for benchmark, relative in LAYOUT.items():
        expected = (
            args.mmspec_sample_num
            if benchmark == "MMSpec"
            else args.fixed_sample_num
        )
        benchmark_dir = args.output_root / relative
        loaded = {}
        policy_reports = {}
        for policy in ("target", args.policy):
            result_path = benchmark_dir / policy / "results.jsonl"
            summary_path = benchmark_dir / policy / "summary.json"
            try:
                rows = read_jsonl(result_path)
                summary = read_json(summary_path)
            except ValueError as error:
                failures.append(str(error))
                continue
            ids = _question_ids(rows)
            missing = sorted(REQUIRED_SUMMARY_FIELDS - set(summary))
            report = {
                "expected_records": expected,
                "num_records": len(rows),
                "unique_question_ids": len(set(ids)),
                "summary_num_records": summary.get("num_records"),
                "missing_summary_fields": missing,
                "result_path": str(result_path.resolve()),
                "summary_path": str(summary_path.resolve()),
            }
            if len(rows) != expected:
                failures.append(
                    f"{benchmark}/{policy}: expected {expected}, got {len(rows)}"
                )
            if len(ids) != len(set(ids)):
                failures.append(f"{benchmark}/{policy}: duplicate question IDs")
            if int(summary.get("num_records", -1)) != len(rows):
                failures.append(f"{benchmark}/{policy}: summary count mismatch")
            if missing:
                failures.append(
                    f"{benchmark}/{policy}: missing summary fields {missing}"
                )
            loaded[policy] = (rows, set(ids))
            policy_reports[policy] = report

        if set(loaded) == {"target", args.policy}:
            if loaded["target"][1] != loaded[args.policy][1]:
                failures.append(f"{benchmark}: target/candidate ID sets differ")
            trace = audit_counterfactual_traces(loaded[args.policy][0])
            policy_reports[args.policy]["counterfactual_trace_audit"] = trace
            if not trace["diagnostic_states"]:
                failures.append(f"{benchmark}: no diagnostic states")
            if trace["valid_states"] != trace["diagnostic_states"]:
                failures.append(f"{benchmark}: invalid diagnostic fields")
            if trace["pair_records"] < expected:
                failures.append(
                    f"{benchmark}: expected at least {expected} pair records, "
                    f"got {trace['pair_records']}"
                )
            for field in (
                "pair_schema_records",
                "distinct_pair_records",
            ):
                if trace[field] != trace["pair_records"]:
                    failures.append(f"{benchmark}: incomplete or invalid pair audit")
            if trace["reused_source_pair_records"]:
                failures.append(f"{benchmark}: wrong-image control reused")
            if not trace["all_source_identities_in_target_set"]:
                failures.append(
                    f"{benchmark}: wrong-image control escaped fresh target set"
                )
        benchmark_reports[benchmark] = policy_reports

    try:
        manifest = read_json(args.fixed_manifest_summary)
        if manifest.get("sample_num") != args.fixed_sample_num:
            failures.append("fixed manifest sample count mismatch")
        if manifest.get("excluded_benchmarks") != ["MME"]:
            failures.append("fixed manifest did not exclude MME")
        if manifest.get("all_source_index_overlaps_zero") is not True:
            failures.append("fixed manifest source-index overlap")
        if manifest.get("all_image_cluster_overlaps_zero") is not True:
            failures.append("fixed manifest image-cluster overlap")
    except ValueError as error:
        failures.append(str(error))

    try:
        range_audit = read_json(args.mmspec_range_audit)
        if range_audit.get("all_pairwise_overlaps_zero") is not True:
            failures.append("MMSpec range overlap")
        fresh = range_audit.get("ranges", {}).get(
            "visual_reliability_confirmation75", {}
        )
        if fresh.get("count") != args.mmspec_sample_num:
            failures.append("MMSpec fresh range count mismatch")
        if fresh.get("unique_image_clusters") != args.mmspec_sample_num:
            failures.append("MMSpec fresh range contains duplicate image clusters")
    except ValueError as error:
        failures.append(str(error))

    analysis_path = args.output_root / args.analysis_dir / "summary.json"
    try:
        analysis = read_json(analysis_path)
        if set(analysis.get("benchmarks", [])) != set(LAYOUT):
            failures.append("analysis benchmark set mismatch")
        if analysis.get("excluded_benchmarks") != ["MME"]:
            failures.append("analysis did not exclude MME")
        if analysis.get("visual_probe_protocols") != [EXPECTED_PROTOCOL]:
            failures.append("analysis protocol mismatch")
        frozen = analysis.get("frozen_rule", {})
        if frozen.get("uses_source_hit_outcomes") is not False:
            failures.append("analysis strata use source-hit outcomes")
        if frozen.get("uses_accepted_length_outcomes") is not False:
            failures.append("analysis strata use accepted-length outcomes")
        if frozen.get("visual_weight") != 0.5:
            failures.append("analysis visual weight mismatch")
        if frozen.get("tail_fraction") != 0.3:
            failures.append("analysis tail fraction mismatch")
        if analysis.get("support_complete") is True:
            for suffix in (".pdf", ".png"):
                figure = (
                    args.output_root
                    / args.analysis_dir
                    / f"visual_reliability_triptych{suffix}"
                )
                if not figure.is_file() or figure.stat().st_size == 0:
                    failures.append(f"missing figure: {figure}")
    except ValueError as error:
        analysis = None
        failures.append(str(error))

    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "output_root": str(args.output_root.resolve()),
        "expected_counts": {
            "MMSpec": args.mmspec_sample_num,
            "fixed_benchmarks": args.fixed_sample_num,
        },
        "fresh_wrong_image_pool": "selected target set",
        "excluded_benchmarks": ["MME"],
        "benchmarks": benchmark_reports,
        "analysis_decision": analysis.get("decision") if analysis else None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mmspec-sample-num", type=int, required=True)
    parser.add_argument("--fixed-sample-num", type=int, required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("visual_reliability_arbitration_validation"),
    )
    parser.add_argument("--fixed-manifest-summary", type=Path, required=True)
    parser.add_argument("--mmspec-range-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args)
    output = args.output or (args.output_root / "completion_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "num_failures": len(payload["failures"]),
                "output": str(output),
            },
            indent=2,
        )
    )
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
