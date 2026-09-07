"""Completion and integrity audit for v15 fixed-benchmark validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))

from evaluation.audit_consensus_counterfactual_run import (  # noqa: E402
    DEFAULT_POLICY,
    EXPECTED_PROTOCOL,
    REQUIRED_SUMMARY_FIELDS,
    _question_ids,
    audit_counterfactual_traces,
    read_json,
    read_jsonl,
)


FIXED_LAYOUT = {
    "MMT-Bench": Path("spatial/fixed/mmt_bench"),
    "SEEDBench": Path("spatial/fixed/seedbench"),
    "ScienceQA": Path("spatial/fixed/scienceqa"),
    "OCRBench": Path("spatial/fixed/ocrbench"),
    "ChartQA": Path("spatial/fixed/chartqa"),
    "MathVista": Path("spatial/fixed/mathvista"),
    "TextVQA": Path("spatial/fixed/textvqa"),
}
CANDIDATE_NUMERIC_FIELDS = (
    "visual_probe_u_candidate_consensus_support",
    "visual_probe_gc_candidate_consensus_support",
    "visual_probe_gc_minus_u_candidate_visual_support",
    "visual_probe_u_candidate_logmass_full",
    "visual_probe_u_candidate_logmass_mean",
    "visual_probe_u_candidate_logmass_wrong",
    "visual_probe_gc_candidate_logmass_full",
    "visual_probe_gc_candidate_logmass_mean",
    "visual_probe_gc_candidate_logmass_wrong",
)


def audit_candidate_fields(rows: list[dict]) -> dict:
    report = {
        "metadata_records": 0,
        "valid_metadata_records": 0,
        "diagnostic_states": 0,
        "states_with_candidate_flag": 0,
        "available_candidate_states": 0,
        "valid_available_candidate_states": 0,
        "states_using_target_outcome": 0,
    }
    for result in rows:
        for choice in result.get("choices", []):
            for metadata in choice.get("selective_probe_metadata", []):
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("protocol") != EXPECTED_PROTOCOL:
                    continue
                report["metadata_records"] += 1
                if (
                    metadata.get("candidate_set_visual_alignment") is True
                    and metadata.get(
                        "candidate_set_visual_alignment_uses_target_outcome"
                    )
                    is False
                ):
                    report["valid_metadata_records"] += 1
            for turn in choice.get("policy_trace", []):
                for trace in turn:
                    if not trace.get("selective_reuse_diagnostics"):
                        continue
                    report["diagnostic_states"] += 1
                    if isinstance(
                        trace.get("visual_probe_candidate_alignment_available"),
                        bool,
                    ):
                        report["states_with_candidate_flag"] += 1
                    if trace.get(
                        "visual_probe_candidate_alignment_uses_target_outcome"
                    ) is True:
                        report["states_using_target_outcome"] += 1
                    if not trace.get(
                        "visual_probe_candidate_alignment_available", False
                    ):
                        continue
                    report["available_candidate_states"] += 1
                    values = [trace.get(field) for field in CANDIDATE_NUMERIC_FIELDS]
                    numeric = all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in values
                    )
                    budget = int(
                        trace.get("visual_probe_candidate_alignment_budget", 0)
                    )
                    u = float(
                        trace.get(
                            "visual_probe_u_candidate_consensus_support", math.nan
                        )
                    )
                    gc = float(
                        trace.get(
                            "visual_probe_gc_candidate_consensus_support", math.nan
                        )
                    )
                    gap = float(
                        trace.get(
                            "visual_probe_gc_minus_u_candidate_visual_support",
                            math.nan,
                        )
                    )
                    consistent = math.isclose(
                        gap, gc - u, rel_tol=1e-6, abs_tol=1e-6
                    )
                    if numeric and budget > 0 and consistent:
                        report["valid_available_candidate_states"] += 1
    return report


def audit(args) -> dict:
    failures = []
    benchmark_reports = {}
    for benchmark, relative in FIXED_LAYOUT.items():
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
                "expected_records": args.sample_num,
                "num_records": len(rows),
                "unique_question_ids": len(set(ids)),
                "summary_num_records": summary.get("num_records"),
                "missing_summary_fields": missing,
                "result_path": str(result_path.resolve()),
                "summary_path": str(summary_path.resolve()),
            }
            if len(rows) != args.sample_num:
                failures.append(
                    f"{benchmark}/{policy}: expected {args.sample_num}, got {len(rows)}"
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
            candidate = audit_candidate_fields(loaded[args.policy][0])
            policy_reports[args.policy]["counterfactual_trace_audit"] = trace
            policy_reports[args.policy]["candidate_alignment_audit"] = candidate
            if not trace["diagnostic_states"]:
                failures.append(f"{benchmark}: no diagnostic states")
            if trace["valid_states"] != trace["diagnostic_states"]:
                failures.append(f"{benchmark}: invalid counterfactual fields")
            if trace["pair_records"] < args.sample_num:
                failures.append(f"{benchmark}: incomplete wrong-image pairs")
            for field in ("pair_schema_records", "distinct_pair_records"):
                if trace[field] != trace["pair_records"]:
                    failures.append(f"{benchmark}: invalid wrong-image pair schema")
            if trace["reused_source_pair_records"]:
                failures.append(f"{benchmark}: wrong-image control reused")
            if not trace["all_source_identities_in_target_set"]:
                failures.append(
                    f"{benchmark}: wrong-image control escaped fresh target set"
                )
            if candidate["metadata_records"] != candidate["valid_metadata_records"]:
                failures.append(f"{benchmark}: invalid candidate metadata")
            if candidate["states_with_candidate_flag"] != candidate["diagnostic_states"]:
                failures.append(f"{benchmark}: missing candidate state flags")
            if candidate["states_using_target_outcome"]:
                failures.append(f"{benchmark}: candidate score uses target outcome")
            if not candidate["available_candidate_states"]:
                failures.append(f"{benchmark}: no available candidate scores")
            if (
                candidate["valid_available_candidate_states"]
                != candidate["available_candidate_states"]
            ):
                failures.append(f"{benchmark}: invalid candidate score values")
        benchmark_reports[benchmark] = policy_reports

    try:
        manifest = read_json(args.manifest_summary)
        if manifest.get("sample_num") != args.sample_num:
            failures.append("manifest sample count mismatch")
        if manifest.get("excluded_benchmarks") != ["MME"]:
            failures.append("manifest did not exclude MME")
        if manifest.get("all_source_index_overlaps_zero") is not True:
            failures.append("manifest source-index overlap")
        if manifest.get("all_image_cluster_overlaps_zero") is not True:
            failures.append("manifest image-cluster overlap")
        if {row.get("dataset") for row in manifest.get("datasets", [])} != set(
            FIXED_LAYOUT
        ):
            failures.append("manifest benchmark set mismatch")
        observed_exclusions = {
            Path(path).name
            for path in manifest.get("additional_exclusion_manifest_dirs", [])
        }
        missing_exclusions = sorted(
            set(args.required_exclusion_dir) - observed_exclusions
        )
        if missing_exclusions:
            failures.append(
                "manifest missing exclusion directories " + repr(missing_exclusions)
            )
    except ValueError as error:
        failures.append(str(error))

    analysis_path = args.output_root / args.analysis_dir / "summary.json"
    try:
        analysis = read_json(analysis_path)
        if set(analysis.get("benchmarks", [])) != set(FIXED_LAYOUT):
            failures.append("analysis benchmark set mismatch")
        if analysis.get("excluded_benchmarks") != ["MME"]:
            failures.append("analysis did not exclude MME")
        if analysis.get("visual_probe_protocols") != [EXPECTED_PROTOCOL]:
            failures.append("analysis protocol mismatch")
        frozen = analysis.get("frozen_rule", {})
        for field in (
            "uses_source_hit_outcomes",
            "uses_accepted_length_outcomes",
            "uses_target_token_id",
        ):
            if frozen.get(field) is not False:
                failures.append(f"analysis violates outcome-blind field {field}")
        if frozen.get("visual_weight") != 0.5:
            failures.append("analysis visual weight mismatch")
        if frozen.get("tail_fraction") != 0.3:
            failures.append("analysis tail fraction mismatch")
        if analysis.get("bootstrap_resamples") != 10000:
            failures.append("analysis bootstrap count mismatch")
        if analysis.get("support_complete") is True:
            for suffix in (".pdf", ".png"):
                figure = (
                    args.output_root
                    / args.analysis_dir
                    / f"candidate_visual_alignment_triptych{suffix}"
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
        "expected_count_per_benchmark": args.sample_num,
        "fresh_wrong_image_pool": "selected target set",
        "excluded_benchmarks": ["MME", "MMSpec"],
        "benchmarks": benchmark_reports,
        "analysis_decision": analysis.get("decision") if analysis else None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-num", type=int, required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("candidate_visual_alignment_validation"),
    )
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument(
        "--required-exclusion-dir", action="append", default=[]
    )
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
