import json

from evaluation.evaluate_selected_consensus_confirmation import evaluate


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_confirmation_evaluates_only_the_development_selected_route(tmp_path):
    selection = tmp_path / "development" / "route_selection.json"
    confirmation = tmp_path / "confirmation"
    _write(
        selection,
        {
            "decision": "go_to_image_disjoint_confirmation",
            "selected_route": "grounded_context_all_conflicts",
            "audit_failures": [],
        },
    )
    _write(confirmation / "completion_audit.json", {"status": "passed"})
    _write(
        confirmation / "grounded_context_visual_analysis" / "summary.json",
        {
            "analysis_role": "new_validation",
            "visual_alignment": "grounded_context",
            "source_population": "all_conflicts",
            "excluded_benchmarks": ["MME"],
            "frozen_rule": {"uses_source_hit_outcomes": False},
            "support_complete": True,
            "strict_confirmatory_crossover": True,
            "decision": "validated",
        },
    )

    payload = evaluate(selection, confirmation)

    assert payload["selected_route"] == "grounded_context_all_conflicts"
    assert payload["validated"] is True
    assert payload["decision"] == "validated"


def test_confirmation_cannot_substitute_a_different_passing_route(tmp_path):
    selection = tmp_path / "development" / "route_selection.json"
    confirmation = tmp_path / "confirmation"
    _write(
        selection,
        {
            "decision": "go_to_image_disjoint_confirmation",
            "selected_route": "current_span2_all_conflicts",
            "audit_failures": [],
        },
    )
    _write(confirmation / "completion_audit.json", {"status": "passed"})
    _write(
        confirmation / "consensus_visual_analysis" / "summary.json",
        {
            "analysis_role": "new_validation",
            "visual_alignment": "current_span2",
            "source_population": "all_conflicts",
            "excluded_benchmarks": ["MME"],
            "frozen_rule": {"uses_source_hit_outcomes": False},
            "support_complete": True,
            "strict_confirmatory_crossover": False,
            "decision": "validation_no_go",
        },
    )
    # A different route looks positive, but is deliberately not consulted.
    _write(
        confirmation / "grounded_context_visual_analysis" / "summary.json",
        {
            "strict_confirmatory_crossover": True,
            "decision": "validated",
        },
    )

    payload = evaluate(selection, confirmation)

    assert payload["validated"] is False
    assert payload["decision"] == "confirmation_no_go"
