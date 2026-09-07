import json

from evaluation.select_consensus_visual_route import ROUTES, select_route


BENCHMARKS = [
    "MMSpec",
    "MMT-Bench",
    "SEEDBench",
    "ScienceQA",
    "OCRBench",
    "ChartQA",
    "MathVista",
    "TextVQA",
]


def _write_route(root, relative, alignment, population, signal, *, passed):
    output = root / relative / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "analysis_role": "development",
                "visual_alignment": alignment,
                "source_population": population,
                "visual_signal": signal,
                "benchmarks": BENCHMARKS,
                "excluded_benchmarks": ["MME"],
                "frozen_rule": {"uses_source_hit_outcomes": False},
                "support_complete": passed,
                "development_directional_crossover": passed,
                "decision": (
                    "go_to_independent_confirmation"
                    if passed
                    else "development_no_go"
                ),
                "point_estimates": {},
                "insufficient_support_benchmarks": [] if passed else BENCHMARKS,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_selector_uses_frozen_least_conditioned_priority(tmp_path):
    for index, (
        _name,
        relative,
        alignment,
        population,
        signal,
    ) in enumerate(ROUTES):
        _write_route(
            tmp_path,
            relative,
            alignment,
            population,
            signal,
            passed=index in (1, 3),
        )

    payload = select_route(tmp_path)

    assert payload["selected_route"] == "grounded_context_all_conflicts"
    assert payload["decision"] == "go_to_image_disjoint_confirmation"
    assert payload["audit_failures"] == []


def test_selector_retains_a_complete_development_no_go(tmp_path):
    for _name, relative, alignment, population, signal in ROUTES:
        _write_route(
            tmp_path,
            relative,
            alignment,
            population,
            signal,
            passed=False,
        )

    payload = select_route(tmp_path)

    assert payload["selected_route"] is None
    assert payload["decision"] == "development_no_go"
    assert len(payload["routes"]) == 6
