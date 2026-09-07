import pytest

from evaluation.analyze_visual_reliability_arbitration import (
    assign_arbitration_strata,
    summarize_benchmark,
)


def _rows():
    rows = []
    for index in range(20):
        low = index < 6
        high = index >= 14
        rows.append(
            {
                "benchmark": "ScienceQA",
                "frozen_eligible": True,
                "visual_probe_full_top1_matches_target": True,
                "visual_probe_span2_mean_mean_target_margin_drop": float(index),
                "visual_probe_span2_wrong_mean_target_margin_drop": float(index),
                "u_row_top_probability": 1.0 - index / 25.0,
                "root_transition_top_probability": index / 25.0,
                "root_transition_context_order": 3,
                "frozen_u_hit": int(low),
                "frozen_gc_hit": int(high),
                "u_matched_accept": 2 if low else 0,
                "gc_matched_accept": 2 if high else 0,
            }
        )
    return rows


def test_arbitration_strata_are_outcome_blind_and_recover_double_crossover():
    rows = _rows()
    audit = assign_arbitration_strata(rows)
    before = [row["arbitration_stratum"] for row in rows]
    summary = summarize_benchmark(rows)

    assert audit["uses_source_hit_outcomes"] is False
    assert audit["uses_accepted_length_outcomes"] is False
    assert before == ["low"] * 6 + ["ambiguous"] * 8 + ["high"] * 6
    assert summary is not None
    assert summary["low_root_delta"] == pytest.approx(-1.0)
    assert summary["high_root_delta"] == pytest.approx(1.0)
    assert summary["low_accept_delta"] == pytest.approx(-2.0)
    assert summary["high_accept_delta"] == pytest.approx(2.0)

    for row in rows:
        row["frozen_u_hit"], row["frozen_gc_hit"] = (
            row["frozen_gc_hit"],
            row["frozen_u_hit"],
        )
        row["u_matched_accept"], row["gc_matched_accept"] = (
            row["gc_matched_accept"],
            row["u_matched_accept"],
        )
    assign_arbitration_strata(rows)
    assert [row["arbitration_stratum"] for row in rows] == before


def test_arbitration_requires_a_gc_context_row():
    rows = _rows()
    rows[0]["root_transition_context_order"] = 1

    audit = assign_arbitration_strata(rows)

    assert rows[0]["arbitration_stratum"] == "invalid_gc_context_order"
    assert audit["invalid_reasons"]["invalid_gc_context_order"] == 1
