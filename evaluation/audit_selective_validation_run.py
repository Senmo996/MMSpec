"""Strict completion audit for an eight-benchmark selective-reuse run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_POLICY = (
    "context-score-trigram-fusion-persistent-suffix4-visualcache-hotpath-"
    "cpp-depth10-node63-wide-plus4"
)
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
CONTENT_ABLATION_PROTOCOL = "teacher_forced_whole_image_mean_ablation_v1"


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


def question_ids(rows: list[dict]) -> list[str]:
    output = []
    for row in rows:
        value = row.get("question_id", row.get("id"))
        if value is None:
            raise ValueError("result row lacks question_id/id")
        output.append(str(value))
    return output


def trace_protocols(rows: list[dict]) -> tuple[set[str], int]:
    protocols = set()
    states = 0
    for row in rows:
        for choice in row.get("choices", []):
            for turn_trace in choice.get("policy_trace", []):
                for trace in turn_trace:
                    if not trace.get("selective_reuse_diagnostics"):
                        continue
                    states += 1
                    protocol = trace.get("visual_probe_protocol")
                    if protocol is not None:
                        protocols.add(str(protocol))
    return protocols, states


def content_span_audit(rows: list[dict]) -> dict:
    required_numeric = (
        "visual_probe_span2_mean_jsd",
        "visual_probe_span2_mean_target_logprob_drop",
        "visual_probe_span2_mean_target_drop_fraction",
    )
    report = {
        "protocol_states": 0,
        "valid_span_states": 0,
        "missing_or_invalid_span_states": 0,
        "one_token_terminal_states": 0,
        "two_token_states": 0,
    }
    for row in rows:
        for choice in row.get("choices", []):
            for turn_trace in choice.get("policy_trace", []):
                for trace in turn_trace:
                    if trace.get("visual_probe_protocol") != CONTENT_ABLATION_PROTOCOL:
                        continue
                    report["protocol_states"] += 1
                    span_tokens = int(trace.get("visual_probe_span2_num_tokens", 0))
                    numeric_values = [trace.get(field) for field in required_numeric]
                    numeric_valid = all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        for value in numeric_values
                    )
                    future_flag = bool(
                        trace.get("visual_probe_span2_uses_future_tokens", False)
                    )
                    valid = (
                        span_tokens in (1, 2)
                        and numeric_valid
                        and trace.get("visual_probe_span2_same_text_trajectory") is True
                        and future_flag == (span_tokens == 2)
                    )
                    if valid:
                        report["valid_span_states"] += 1
                        key = (
                            "two_token_states"
                            if span_tokens == 2
                            else "one_token_terminal_states"
                        )
                        report[key] += 1
                    else:
                        report["missing_or_invalid_span_states"] += 1
    return report


def audit(args) -> dict:
    failures = []
    benchmark_reports = {}
    for benchmark, relative in LAYOUT.items():
        benchmark_dir = args.output_root / relative
        policy_reports = {}
        loaded = {}
        for policy in ("target", args.policy):
            policy_dir = benchmark_dir / policy
            result_path = policy_dir / "results.jsonl"
            summary_path = policy_dir / "summary.json"
            try:
                rows = read_jsonl(result_path)
                summary = read_json(summary_path)
            except ValueError as error:
                failures.append(str(error))
                continue
            ids = question_ids(rows)
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
            target_ids = loaded["target"][1]
            candidate_rows, candidate_ids = loaded[args.policy]
            if target_ids != candidate_ids:
                failures.append(f"{benchmark}: target/candidate ID sets differ")
            protocols, trace_states = trace_protocols(candidate_rows)
            policy_reports[args.policy]["trace_states"] = trace_states
            policy_reports[args.policy]["visual_probe_protocols"] = sorted(protocols)
            if protocols != {args.expected_probe_protocol}:
                failures.append(
                    f"{benchmark}: expected probe protocol "
                    f"{args.expected_probe_protocol}, got {sorted(protocols)}"
                )
            if trace_states == 0:
                failures.append(f"{benchmark}: no selective diagnostic states")
            if args.expected_probe_protocol == CONTENT_ABLATION_PROTOCOL:
                span_report = content_span_audit(candidate_rows)
                policy_reports[args.policy]["exact_span2_audit"] = span_report
                if span_report["protocol_states"] != trace_states:
                    failures.append(
                        f"{benchmark}: not every diagnostic state has the "
                        "content-ablation protocol"
                    )
                if span_report["valid_span_states"] != trace_states:
                    failures.append(
                        f"{benchmark}: exact two-token visual fields are "
                        "missing or invalid"
                    )
        benchmark_reports[benchmark] = policy_reports

    analysis_dir = args.output_root / args.analysis_dir
    root_summary_path = args.output_root / "summary.json"
    analysis_summary_path = analysis_dir / "summary.json"
    figure_pdf = analysis_dir / "selective_conflict_triptych.pdf"
    figure_png = analysis_dir / "selective_conflict_triptych.png"
    for path in (root_summary_path, analysis_summary_path, figure_pdf, figure_png):
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty artifact: {path}")
    evidence_gate = None
    if analysis_summary_path.is_file():
        try:
            analysis_summary = read_json(analysis_summary_path)
            evidence_gate = analysis_summary.get("evidence_gate")
            if sorted(analysis_summary.get("benchmarks", [])) != sorted(LAYOUT):
                failures.append("analysis summary benchmark set is incomplete")
            if analysis_summary.get("excluded_benchmarks") != ["MME"]:
                failures.append("analysis summary does not explicitly exclude MME")
            if analysis_summary.get("visual_probe_protocols") != [
                args.expected_probe_protocol
            ]:
                failures.append("analysis summary probe protocol mismatch")
            if (
                args.expected_visual_metric
                and analysis_summary.get("frozen_rule", {}).get("visual_metric")
                != args.expected_visual_metric
            ):
                failures.append("analysis summary visual metric mismatch")
        except ValueError as error:
            failures.append(str(error))

    return {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "output_root": str(args.output_root.resolve()),
        "expected_sample_num_per_benchmark": args.sample_num,
        "expected_probe_protocol": args.expected_probe_protocol,
        "expected_visual_metric": args.expected_visual_metric,
        "excluded_benchmarks": ["MME"],
        "benchmarks": benchmark_reports,
        "evidence_gate": evidence_gate,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-num", type=int, required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--expected-probe-protocol", required=True)
    parser.add_argument("--expected-visual-metric")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_num <= 0:
        parser.error("--sample-num must be positive")
    payload = audit(args)
    output = args.output or (args.output_root / "completion_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
