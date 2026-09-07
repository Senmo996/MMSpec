"""Completion audit for the eight-benchmark counterfactual-bank experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_POLICY = (
    "context-score-trigram-fusion-persistent-suffix4-visualcache-hotpath-"
    "cpp-depth10-node63-wide-plus4"
)
EXPECTED_PROTOCOL = "teacher_forced_mean_and_matched_wrong_image_v1"
LAYOUT = {
    "MMSpec": Path("spatial/mmspec"),
    "MMT-Bench": Path("spatial/fixed/mmt_bench"),
    "SEEDBench": Path("spatial/fixed/seedbench"),
    "ScienceQA": Path("spatial/fixed/scienceqa"),
    "OCRBench": Path("spatial/fixed/ocrbench"),
    "ChartQA": Path("spatial/fixed/chartqa"),
    "MathVista": Path("spatial/fixed/mathvista"),
    "TextVQA": Path("spatial/fixed/textvqa"),
}
REQUIRED_SUMMARY_FIELDS = {
    "num_records",
    "avg_accept_length",
    "median_accept_length",
    "max_accept_length",
    "p90_accept_length",
    "avg_tokens_per_iteration",
    "avg_sample_speedup",
    "median_sample_speedup",
    "max_sample_speedup",
    "sample_speedup_gt_1_ratio",
    "accept_le_0_5_ratio",
    "jsonl_path",
}
REQUIRED_TRACE_NUMERIC_FIELDS = (
    "visual_probe_span2_mean_mean_target_margin_drop",
    "visual_probe_span2_wrong_mean_target_margin_drop",
    "visual_probe_span2_mean_top1_change_rate",
    "visual_probe_span2_wrong_top1_change_rate",
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    return rows


def _question_ids(rows: list[dict]) -> list[str]:
    ids = []
    for row in rows:
        value = row.get("question_id", row.get("id"))
        if value is None:
            raise ValueError("result row lacks question_id/id")
        ids.append(str(value))
    return ids


def audit_counterfactual_traces(rows: list[dict]) -> dict:
    report = {
        "diagnostic_states": 0,
        "valid_states": 0,
        "invalid_states": 0,
        "pair_records": 0,
        "pair_schema_records": 0,
        "distinct_pair_records": 0,
        "category_fallback_pair_records": 0,
        "reused_source_pair_records": 0,
        "u_readiness_valid_states": 0,
        "u_ready_states": 0,
        "context_alignment_valid_states": 0,
    }
    target_pair_identities = set()
    source_pair_identities = set()
    for result in rows:
        for choice in result.get("choices", []):
            for metadata in choice.get("selective_probe_metadata", []):
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("protocol") != EXPECTED_PROTOCOL:
                    continue
                pair = metadata.get("wrong_image_pair", {})
                report["pair_records"] += 1
                if isinstance(pair.get("used_category_fallback"), bool) and isinstance(
                    pair.get("source_reused_within_run"), bool
                ):
                    report["pair_schema_records"] += 1
                if (
                    pair.get("target_image_identity")
                    and pair.get("source_image_identity")
                    and pair["target_image_identity"] != pair["source_image_identity"]
                ):
                    report["distinct_pair_records"] += 1
                if pair.get("target_image_identity"):
                    target_pair_identities.add(str(pair["target_image_identity"]))
                if pair.get("source_image_identity"):
                    source_pair_identities.add(str(pair["source_image_identity"]))
                if pair.get("used_category_fallback") is True:
                    report["category_fallback_pair_records"] += 1
                if pair.get("source_reused_within_run") is True:
                    report["reused_source_pair_records"] += 1
            for turn_trace in choice.get("policy_trace", []):
                for trace in turn_trace:
                    if not trace.get("selective_reuse_diagnostics"):
                        continue
                    report["diagnostic_states"] += 1
                    span_tokens = int(trace.get("visual_probe_span2_num_tokens", 0))
                    values = [trace.get(field) for field in REQUIRED_TRACE_NUMERIC_FIELDS]
                    numeric_valid = all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in values
                    )
                    rates_valid = all(
                        0.0 <= float(trace.get(field, -1.0)) <= 1.0
                        for field in (
                            "visual_probe_span2_mean_top1_change_rate",
                            "visual_probe_span2_wrong_top1_change_rate",
                        )
                    )
                    context_count = int(
                        trace.get("visual_probe_context3_num_tokens", -1)
                    )
                    expected_context_count = min(
                        max(int(trace.get("selective_output_position", 0)), 0),
                        3,
                    )
                    context_targets = trace.get(
                        "visual_probe_context3_target_token_ids"
                    )
                    context_full_top1 = trace.get(
                        "visual_probe_context3_full_top1_token_ids"
                    )
                    context_numeric_lists = [
                        trace.get(
                            "visual_probe_context3_mean_target_margin_drops"
                        ),
                        trace.get(
                            "visual_probe_context3_wrong_target_margin_drops"
                        ),
                    ]
                    context_bool_lists = [
                        trace.get("visual_probe_context3_mean_top1_changed"),
                        trace.get("visual_probe_context3_wrong_top1_changed"),
                    ]
                    context_valid = bool(
                        context_count == expected_context_count
                        and isinstance(context_targets, list)
                        and isinstance(context_full_top1, list)
                        and len(context_targets) == context_count
                        and len(context_full_top1) == context_count
                        and all(
                            isinstance(values, list)
                            and len(values) == context_count
                            and all(
                                isinstance(value, (int, float))
                                and math.isfinite(float(value))
                                for value in values
                            )
                            for values in context_numeric_lists
                        )
                        and all(
                            isinstance(values, list)
                            and len(values) == context_count
                            and all(isinstance(value, bool) for value in values)
                            for values in context_bool_lists
                        )
                        and trace.get(
                            "visual_probe_context3_same_text_trajectory"
                        )
                        is True
                    )
                    if context_valid:
                        report["context_alignment_valid_states"] += 1
                    valid = (
                        trace.get("visual_probe_protocol") == EXPECTED_PROTOCOL
                        and span_tokens in (1, 2)
                        and numeric_valid
                        and rates_valid
                        and trace.get("visual_probe_same_text_trajectory") is True
                        and trace.get("visual_probe_recomputed_vision_encoder") is True
                        and trace.get("visual_probe_span2_same_text_trajectory") is True
                        and bool(trace.get("visual_probe_span2_uses_future_tokens"))
                        == (span_tokens == 2)
                        and bool(trace.get("visual_probe_wrong_image_source_identity"))
                        and context_valid
                    )
                    u_available_before_request = trace.get(
                        "selective_u_available_before_request"
                    )
                    u_probability_before_request = trace.get(
                        "selective_u_row_top_probability_before_request"
                    )
                    u_readiness_valid = isinstance(
                        u_available_before_request, bool
                    ) and (
                        (
                            isinstance(u_probability_before_request, (int, float))
                            and math.isfinite(float(u_probability_before_request))
                        )
                        if u_available_before_request
                        else (
                            u_probability_before_request is None
                            or (
                                isinstance(
                                    u_probability_before_request, (int, float)
                                )
                                and math.isfinite(
                                    float(u_probability_before_request)
                                )
                            )
                        )
                    )
                    if u_readiness_valid:
                        report["u_readiness_valid_states"] += 1
                    if (
                        u_readiness_valid
                        and u_available_before_request
                        and float(u_probability_before_request) >= 0.50
                    ):
                        report["u_ready_states"] += 1
                    if valid:
                        report["valid_states"] += 1
                    else:
                        report["invalid_states"] += 1
    report["unique_target_pair_identities"] = len(target_pair_identities)
    report["unique_source_pair_identities"] = len(source_pair_identities)
    report["all_source_identities_in_target_set"] = bool(
        source_pair_identities
        and source_pair_identities.issubset(target_pair_identities)
    )
    return report


def audit(args) -> dict:
    failures = []
    benchmark_reports = {}
    for benchmark, relative in LAYOUT.items():
        benchmark_dir = args.output_root / relative
        policy_reports = {}
        loaded = {}
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
            missing_summary_fields = sorted(REQUIRED_SUMMARY_FIELDS - set(summary))
            report = {
                "result_path": str(result_path.resolve()),
                "summary_path": str(summary_path.resolve()),
                "num_records": len(rows),
                "unique_question_ids": len(set(ids)),
                "summary_num_records": summary.get("num_records"),
                "missing_summary_fields": missing_summary_fields,
            }
            if len(rows) != args.sample_num:
                failures.append(
                    f"{benchmark}/{policy}: expected {args.sample_num} records, "
                    f"got {len(rows)}"
                )
            if len(set(ids)) != len(ids):
                failures.append(f"{benchmark}/{policy}: duplicate question IDs")
            if int(summary.get("num_records", -1)) != len(rows):
                failures.append(f"{benchmark}/{policy}: summary count mismatch")
            if missing_summary_fields:
                failures.append(
                    f"{benchmark}/{policy}: missing summary fields "
                    f"{missing_summary_fields}"
                )
            policy_reports[policy] = report
            loaded[policy] = (rows, set(ids))

        if set(loaded) == {"target", args.policy}:
            if loaded["target"][1] != loaded[args.policy][1]:
                failures.append(f"{benchmark}: target/candidate ID sets differ")
            trace_report = audit_counterfactual_traces(loaded[args.policy][0])
            policy_reports[args.policy]["counterfactual_trace_audit"] = trace_report
            if trace_report["diagnostic_states"] == 0:
                failures.append(f"{benchmark}: no selective diagnostic states")
            if trace_report["valid_states"] != trace_report["diagnostic_states"]:
                failures.append(f"{benchmark}: invalid counterfactual trace fields")
            if (
                trace_report["u_readiness_valid_states"]
                != trace_report["diagnostic_states"]
            ):
                failures.append(f"{benchmark}: invalid U-readiness trace fields")
            if (
                trace_report["context_alignment_valid_states"]
                != trace_report["diagnostic_states"]
            ):
                failures.append(f"{benchmark}: invalid context-alignment fields")
            if trace_report["pair_records"] == 0:
                failures.append(f"{benchmark}: no wrong-image pair metadata")
            if trace_report["pair_schema_records"] != trace_report["pair_records"]:
                failures.append(f"{benchmark}: incomplete wrong-image pair schema")
            if trace_report["distinct_pair_records"] != trace_report["pair_records"]:
                failures.append(f"{benchmark}: non-distinct wrong-image pair")
            if trace_report["reused_source_pair_records"]:
                failures.append(f"{benchmark}: wrong-image source reused within run")
        benchmark_reports[benchmark] = policy_reports

    analysis_dir = args.output_root / args.analysis_dir
    root_summary_path = args.output_root / "summary.json"
    analysis_summary_path = analysis_dir / "summary.json"
    for path in (root_summary_path, analysis_summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {path}")

    analysis_summary = None
    if analysis_summary_path.is_file():
        try:
            analysis_summary = read_json(analysis_summary_path)
            if set(analysis_summary.get("benchmarks", [])) != set(LAYOUT):
                failures.append("analysis summary benchmark set is incomplete")
            if analysis_summary.get("excluded_benchmarks") != ["MME"]:
                failures.append("analysis summary does not explicitly exclude MME")
            if analysis_summary.get("visual_probe_protocols") != [EXPECTED_PROTOCOL]:
                failures.append("analysis summary probe protocol mismatch")
            if analysis_summary.get("frozen_rule", {}).get(
                "uses_source_hit_outcomes"
            ) is not False:
                failures.append("visual stratum rule is not outcome-independent")
            if analysis_summary.get("support_complete") is True:
                for suffix in (".pdf", ".png"):
                    figure = analysis_dir / f"consensus_visual_triptych{suffix}"
                    if not figure.is_file() or figure.stat().st_size == 0:
                        failures.append(f"missing or empty artifact: {figure}")
        except ValueError as error:
            failures.append(str(error))

    u_ready_analysis_dir = args.output_root / args.u_ready_analysis_dir
    u_ready_summary_path = u_ready_analysis_dir / "summary.json"
    u_ready_summary = None
    if not u_ready_summary_path.is_file() or u_ready_summary_path.stat().st_size == 0:
        failures.append(f"missing or empty artifact: {u_ready_summary_path}")
    else:
        try:
            u_ready_summary = read_json(u_ready_summary_path)
            if u_ready_summary.get("source_population") != "pre_request_confident_u":
                failures.append("U-ready analysis source population mismatch")
            if set(u_ready_summary.get("benchmarks", [])) != set(LAYOUT):
                failures.append("U-ready analysis benchmark set is incomplete")
            if u_ready_summary.get("excluded_benchmarks") != ["MME"]:
                failures.append("U-ready analysis does not explicitly exclude MME")
            if u_ready_summary.get("frozen_rule", {}).get(
                "uses_source_hit_outcomes"
            ) is not False:
                failures.append("U-ready stratum rule is not outcome-independent")
            if u_ready_summary.get("support_complete") is True:
                for suffix in (".pdf", ".png"):
                    figure = u_ready_analysis_dir / (
                        f"consensus_visual_triptych{suffix}"
                    )
                    if not figure.is_file() or figure.stat().st_size == 0:
                        failures.append(f"missing or empty artifact: {figure}")
        except ValueError as error:
            failures.append(str(error))

    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "output_root": str(args.output_root.resolve()),
        "expected_sample_num_per_benchmark": args.sample_num,
        "expected_probe_protocol": EXPECTED_PROTOCOL,
        "excluded_benchmarks": ["MME"],
        "benchmarks": benchmark_reports,
        "analysis_decision": (
            analysis_summary.get("decision") if analysis_summary else None
        ),
        "u_ready_analysis_decision": (
            u_ready_summary.get("decision") if u_ready_summary else None
        ),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-num", type=int, required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("consensus_visual_analysis")
    )
    parser.add_argument(
        "--u-ready-analysis-dir",
        type=Path,
        default=Path("consensus_visual_u_ready_analysis"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_num <= 0:
        parser.error("--sample-num must be positive")
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
