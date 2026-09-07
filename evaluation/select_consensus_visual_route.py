"""Select one preregistered development route by least-conditioned priority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
ROUTES = (
    (
        "current_span2_all_conflicts",
        Path("consensus_visual_analysis"),
        "current_span2",
        "all_conflicts",
        "target_margin_drop",
    ),
    (
        "grounded_context_all_conflicts",
        Path("grounded_context_visual_analysis"),
        "grounded_context",
        "all_conflicts",
        "target_margin_drop",
    ),
    (
        "current_span2_pre_request_confident_u",
        Path("consensus_visual_u_ready_analysis"),
        "current_span2",
        "pre_request_confident_u",
        "target_margin_drop",
    ),
    (
        "grounded_context_pre_request_confident_u",
        Path("grounded_context_visual_u_ready_analysis"),
        "grounded_context",
        "pre_request_confident_u",
        "target_margin_drop",
    ),
    (
        "distributional_current_span2_all_conflicts",
        Path("distributional_visual_analysis"),
        "current_span2",
        "all_conflicts",
        "distribution_jsd",
    ),
    (
        "distributional_current_span2_pre_request_confident_u",
        Path("distributional_visual_u_ready_analysis"),
        "current_span2",
        "pre_request_confident_u",
        "distribution_jsd",
    ),
)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def select_route(output_root: Path) -> dict:
    reports = []
    failures = []
    selected = None
    for priority, (name, relative, alignment, population, signal) in enumerate(
        ROUTES, 1
    ):
        summary_path = output_root / relative / "summary.json"
        try:
            summary = read_json(summary_path)
        except ValueError as error:
            failures.append(str(error))
            continue
        route_failures = []
        if summary.get("analysis_role") != "development":
            route_failures.append("analysis_role is not development")
        if summary.get("visual_alignment", "current_span2") != alignment:
            route_failures.append("visual_alignment mismatch")
        if summary.get("source_population") != population:
            route_failures.append("source_population mismatch")
        if summary.get("visual_signal", "target_margin_drop") != signal:
            route_failures.append("visual_signal mismatch")
        if set(summary.get("benchmarks", [])) != EXPECTED_BENCHMARKS:
            route_failures.append("benchmark set mismatch")
        if summary.get("excluded_benchmarks") != ["MME"]:
            route_failures.append("MME exclusion mismatch")
        if summary.get("frozen_rule", {}).get("uses_source_hit_outcomes") is not False:
            route_failures.append("stratum rule is not source-outcome blind")
        passed = bool(
            not route_failures
            and summary.get("support_complete") is True
            and summary.get("development_directional_crossover") is True
        )
        report = {
            "priority": priority,
            "route": name,
            "summary_path": str(summary_path.resolve()),
            "visual_alignment": alignment,
            "source_population": population,
            "visual_signal": signal,
            "support_complete": bool(summary.get("support_complete", False)),
            "development_directional_crossover": bool(
                summary.get("development_directional_crossover", False)
            ),
            "decision": summary.get("decision"),
            "point_estimates": summary.get("point_estimates", {}),
            "insufficient_support_benchmarks": summary.get(
                "insufficient_support_benchmarks", []
            ),
            "route_failures": route_failures,
            "passes_development_gate": passed,
        }
        reports.append(report)
        if selected is None and passed:
            selected = report

    return {
        "schema_version": 1,
        "analysis_role": "development_route_selection",
        "selection_rule": "first passing route in frozen least-conditioned priority",
        "uses_sample_or_benchmark_selection": False,
        "excluded_benchmarks": ["MME"],
        "route_priority": [name for name, *_rest in ROUTES],
        "routes": reports,
        "selected_route": selected["route"] if selected else None,
        "selected_summary_path": selected["summary_path"] if selected else None,
        "decision": (
            "go_to_image_disjoint_confirmation"
            if selected is not None
            else "development_no_go"
        ),
        "audit_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = select_route(args.output_root)
    output = args.output or (args.output_root / "route_selection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "selected_route": payload["selected_route"],
                "audit_failures": payload["audit_failures"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if payload["audit_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
