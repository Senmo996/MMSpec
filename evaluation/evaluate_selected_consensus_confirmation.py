"""Evaluate only the development-selected route on independent confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from evaluation.select_consensus_visual_route import ROUTES


ROUTE_INDEX = {
    name: {
        "relative": relative,
        "visual_alignment": alignment,
        "source_population": population,
        "visual_signal": signal,
    }
    for name, relative, alignment, population, signal in ROUTES
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error


def evaluate(dev_selection_path: Path, confirmation_root: Path) -> dict:
    failures = []
    development = read_json(dev_selection_path)
    selected_route = development.get("selected_route")
    if development.get("decision") != "go_to_image_disjoint_confirmation":
        failures.append("development decision did not authorize confirmation")
    if development.get("audit_failures"):
        failures.append("development route-selection audit failed")
    if selected_route not in ROUTE_INDEX:
        failures.append(f"unknown or absent selected route: {selected_route}")
        return {
            "schema_version": 1,
            "decision": "confirmation_invalid",
            "selected_route": selected_route,
            "validated": False,
            "failures": failures,
        }

    route = ROUTE_INDEX[selected_route]
    summary_path = confirmation_root / route["relative"] / "summary.json"
    completion_audit_path = confirmation_root / "completion_audit.json"
    summary = read_json(summary_path)
    completion_audit = read_json(completion_audit_path)
    if completion_audit.get("status") != "passed":
        failures.append("confirmation completion audit did not pass")
    if summary.get("analysis_role") != "new_validation":
        failures.append("confirmation analysis role mismatch")
    if summary.get("visual_alignment", "current_span2") != route["visual_alignment"]:
        failures.append("confirmation visual alignment mismatch")
    if summary.get("source_population") != route["source_population"]:
        failures.append("confirmation source population mismatch")
    if summary.get("visual_signal", "target_margin_drop") != route["visual_signal"]:
        failures.append("confirmation visual signal mismatch")
    if summary.get("excluded_benchmarks") != ["MME"]:
        failures.append("confirmation did not explicitly exclude MME")
    if summary.get("frozen_rule", {}).get("uses_source_hit_outcomes") is not False:
        failures.append("confirmation strata are not source-outcome blind")

    validated = bool(
        not failures
        and summary.get("support_complete") is True
        and summary.get("strict_confirmatory_crossover") is True
        and summary.get("decision") == "validated"
    )
    return {
        "schema_version": 1,
        "analysis_role": "frozen_route_independent_confirmation",
        "development_route_selection": str(dev_selection_path.resolve()),
        "confirmation_root": str(confirmation_root.resolve()),
        "selected_route": selected_route,
        "selected_summary_path": str(summary_path.resolve()),
        "visual_alignment": route["visual_alignment"],
        "source_population": route["source_population"],
        "visual_signal": route["visual_signal"],
        "support_complete": bool(summary.get("support_complete", False)),
        "strict_confirmatory_crossover": bool(
            summary.get("strict_confirmatory_crossover", False)
        ),
        "point_estimates": summary.get("point_estimates", {}),
        "cluster_bootstrap_95_ci": summary.get("cluster_bootstrap_95_ci", {}),
        "validated": validated,
        "decision": "validated" if validated else "confirmation_no_go",
        "failures": failures,
    }


def promote_if_validated(payload: dict, confirmation_root: Path) -> None:
    if not payload.get("validated"):
        return
    source_summary = Path(payload["selected_summary_path"])
    source_stem = source_summary.parent / "consensus_visual_triptych"
    destination = confirmation_root / "validated_selected_route"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_summary, destination / "summary.json")
    for suffix in (".pdf", ".png"):
        source = source_stem.with_suffix(suffix)
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"validated route is missing figure: {source}")
        shutil.copy2(source, destination / f"selective_reuse_triptych{suffix}")
    (destination / "provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-route-selection", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = evaluate(
            args.development_route_selection,
            args.confirmation_root,
        )
        promote_if_validated(payload, args.confirmation_root)
    except ValueError as error:
        payload = {
            "schema_version": 1,
            "decision": "confirmation_invalid",
            "validated": False,
            "failures": [str(error)],
        }
    output = args.output or (args.confirmation_root / "confirmation_decision.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "validated": payload["validated"],
                "failures": payload["failures"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
