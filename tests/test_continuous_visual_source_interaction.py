import numpy as np
import pytest

from evaluation.analyze_continuous_visual_source_interaction import (
    assign_analysis_visual_percentiles,
    empirical_conflict_tail_macro_summary,
    fit_linear_probability,
    summarize_empirical_conflict_tails_benchmark,
    summarize_continuous_benchmark,
)


def test_linear_probability_model_centers_visual_percentile_at_half():
    x = np.asarray([0.0, 0.25, 0.75, 1.0])
    y = 0.4 + 0.3 * (x - 0.5)

    intercept, slope = fit_linear_probability(x, y)

    assert intercept == pytest.approx(0.4)
    assert slope == pytest.approx(0.3)


def test_continuous_summary_recovers_source_preference_crossover():
    rows = []
    for percentile, u_hit, gc_hit in (
        (0.1, 1, 0),
        (0.3, 1, 0),
        (0.7, 0, 1),
        (0.9, 0, 1),
    ):
        rows.append(
            {
                "frozen_eligible": True,
                "frozen_visual_percentile": percentile,
                "frozen_u_hit": u_hit,
                "frozen_gc_hit": gc_hit,
                "frozen_delta": gc_hit - u_hit,
            }
        )

    summary = summarize_continuous_benchmark(rows)

    assert summary is not None
    assert summary["delta_slope"] > 0.0
    assert summary["delta_p10"] < 0.0
    assert summary["delta_p90"] > 0.0


def test_continuous_summary_requires_visual_variation():
    rows = [
        {
            "frozen_eligible": True,
            "frozen_visual_percentile": 0.5,
            "frozen_u_hit": 1,
            "frozen_gc_hit": 0,
            "frozen_delta": -1,
        }
        for _ in range(3)
    ]

    assert summarize_continuous_benchmark(rows) is None


def test_conflict_conditioned_percentiles_ignore_ineligible_extremes():
    rows = []
    for index, (score, eligible) in enumerate(
        ((-100.0, False), (1.0, True), (2.0, True), (100.0, False))
    ):
        rows.append(
            {
                "benchmark": "ScienceQA",
                "frozen_visual_score": score,
                "frozen_visual_percentile": (index + 0.5) / 4.0,
                "frozen_eligible": eligible,
            }
        )

    audit = assign_analysis_visual_percentiles(
        rows, reference_population="eligible_conflicts"
    )

    assert audit["uses_source_hit_outcomes"] is False
    assert audit["num_reference_states"] == 2
    assert rows[0]["analysis_visual_percentile"] is None
    assert rows[1]["analysis_visual_percentile"] == pytest.approx(0.25)
    assert rows[2]["analysis_visual_percentile"] == pytest.approx(0.75)
    assert rows[3]["analysis_visual_percentile"] is None
    assert rows[0]["all_state_visual_percentile"] == pytest.approx(0.125)


def test_continuous_summary_accepts_separate_analysis_percentile():
    rows = []
    for percentile, u_hit, gc_hit in (
        (0.1, 1, 0),
        (0.3, 1, 0),
        (0.7, 0, 1),
        (0.9, 0, 1),
    ):
        rows.append(
            {
                "frozen_eligible": True,
                "analysis_visual_percentile": percentile,
                "frozen_u_hit": u_hit,
                "frozen_gc_hit": gc_hit,
                "frozen_delta": gc_hit - u_hit,
            }
        )

    summary = summarize_continuous_benchmark(
        rows, percentile_field="analysis_visual_percentile"
    )

    assert summary is not None
    assert summary["delta_p10"] < 0.0 < summary["delta_p90"]


def test_empirical_tails_do_not_inherit_linear_endpoint_crossover():
    rows = []
    for percentile, delta in (
        (0.05, 1),
        (0.15, 0),
        (0.35, -1),
        (0.65, 0),
        (0.85, 1),
        (0.95, 1),
    ):
        rows.append(
            {
                "benchmark": "ScienceQA",
                "frozen_eligible": True,
                "analysis_visual_percentile": percentile,
                "frozen_u_hit": int(delta < 0),
                "frozen_gc_hit": int(delta > 0),
                "frozen_delta": delta,
            }
        )

    summary = summarize_empirical_conflict_tails_benchmark(
        rows, percentile_field="analysis_visual_percentile"
    )

    assert summary is not None
    assert summary["low_delta"] == pytest.approx(0.5)
    assert summary["high_delta"] == pytest.approx(1.0)
    assert summary["interaction"] == pytest.approx(0.5)


def test_empirical_tail_macro_uses_equal_benchmark_weight():
    rows = []
    for benchmark, low_delta, high_delta in (
        ("A", -1, 1),
        ("B", 1, 1),
    ):
        for percentile, delta in ((0.1, low_delta), (0.9, high_delta)):
            rows.append(
                {
                    "benchmark": benchmark,
                    "frozen_eligible": True,
                    "analysis_visual_percentile": percentile,
                    "frozen_u_hit": int(delta < 0),
                    "frozen_gc_hit": int(delta > 0),
                    "frozen_delta": delta,
                }
            )

    macro, by_benchmark = empirical_conflict_tail_macro_summary(
        rows, percentile_field="analysis_visual_percentile"
    )

    assert set(by_benchmark) == {"A", "B"}
    assert macro["low_delta"] == pytest.approx(0.0)
    assert macro["high_delta"] == pytest.approx(1.0)
    assert macro["interaction"] == pytest.approx(1.0)
