import pytest

from evaluation.analyze_visual_reliability_components import (
    assign_component_strata,
    summarize_component_benchmark,
    weighted_factorial_regression,
)


def _rows():
    rows = []
    for index in range(20):
        visual = index / 19.0
        reliability = (19 - index) / 19.0
        rows.append(
            {
                "benchmark": "ScienceQA",
                "cluster_id": f"ScienceQA:{index}",
                "arbitration_visual_rank": visual,
                "arbitration_gc_confidence_advantage_rank": reliability,
                "arbitration_score": 0.5 * (visual + reliability),
                "frozen_delta": -1 if index < 6 else 1 if index >= 14 else 0,
                "matched_gc_minus_u_accept": (
                    -2 if index < 6 else 2 if index >= 14 else 0
                ),
                "u_matched_accept": 2 if index < 6 else 0,
                "gc_matched_accept": 2 if index >= 14 else 0,
            }
        )
    return rows


def test_component_strata_use_only_the_requested_score():
    rows = _rows()
    audit = assign_component_strata(rows, route_name="visual_only")
    label_field = audit["label_field"]
    before = [row[label_field] for row in rows]

    assert before == ["low"] * 6 + ["ambiguous"] * 8 + ["high"] * 6
    assert audit["uses_source_hit_outcomes"] is False
    assert audit["uses_accepted_length_outcomes"] is False

    for row in rows:
        row["frozen_delta"] *= -1
        row["u_matched_accept"], row["gc_matched_accept"] = (
            row["gc_matched_accept"],
            row["u_matched_accept"],
        )
    assign_component_strata(rows, route_name="visual_only")
    assert [row[label_field] for row in rows] == before


def test_component_summary_reports_both_outcomes():
    rows = _rows()
    assign_component_strata(rows, route_name="visual_only")

    summary = summarize_component_benchmark(rows, route_name="visual_only")

    assert summary is not None
    assert summary["low_root_delta"] == pytest.approx(-1.0)
    assert summary["high_root_delta"] == pytest.approx(1.0)
    assert summary["low_accept_delta"] == pytest.approx(-2.0)
    assert summary["high_accept_delta"] == pytest.approx(2.0)


def test_weighted_factorial_regression_recovers_known_coefficients():
    rows = []
    for benchmark in ("ScienceQA", "MathVista"):
        for visual in (0.1, 0.3, 0.7, 0.9):
            for reliability in (0.2, 0.8):
                v = visual - 0.5
                r = reliability - 0.5
                outcome = 0.2 + 1.5 * v - 0.7 * r + 2.0 * v * r
                rows.append(
                    {
                        "benchmark": benchmark,
                        "arbitration_visual_rank": visual,
                        "arbitration_gc_confidence_advantage_rank": reliability,
                        "arbitration_score": 0.5 * (visual + reliability),
                        "known_outcome": outcome,
                    }
                )

    coefficients = weighted_factorial_regression(rows, outcome="known_outcome")

    assert coefficients["intercept"] == pytest.approx(0.2)
    assert coefficients["visual"] == pytest.approx(1.5)
    assert coefficients["reliability"] == pytest.approx(-0.7)
    assert coefficients["interaction"] == pytest.approx(2.0)
