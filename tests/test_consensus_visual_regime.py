import pytest

from evaluation.analyze_consensus_visual_regime import (
    MEAN_JSD_FIELD,
    MEAN_MARGIN_FIELD,
    MEAN_TOP1_FIELD,
    WRONG_JSD_FIELD,
    WRONG_MARGIN_FIELD,
    WRONG_TOP1_FIELD,
    assign_consensus_visual_strata,
    plot,
    summarize_consensus_benchmark,
)


def _rows():
    rows = []
    for index in range(10):
        if index < 4:
            mean_top1 = wrong_top1 = 0.0
            u_hit, gc_hit = 1, 0
        elif index >= 6:
            mean_top1 = wrong_top1 = 0.5
            u_hit, gc_hit = 0, 1
        else:
            mean_top1 = wrong_top1 = 0.0
            u_hit = gc_hit = 0
        rows.append(
            {
                "benchmark": "ScienceQA",
                "frozen_eligible": True,
                "visual_probe_full_top1_matches_target": True,
                MEAN_MARGIN_FIELD: float(index),
                WRONG_MARGIN_FIELD: float(index) * 2.0,
                MEAN_TOP1_FIELD: mean_top1,
                WRONG_TOP1_FIELD: wrong_top1,
                "frozen_u_hit": u_hit,
                "frozen_gc_hit": gc_hit,
                "frozen_delta": gc_hit - u_hit,
            }
        )
    return rows


def test_consensus_strata_require_both_counterfactuals_and_top1_agreement():
    rows = _rows()

    audit = assign_consensus_visual_strata(rows)

    assert audit["uses_source_hit_outcomes"] is False
    assert [row["consensus_visual_stratum"] for row in rows] == [
        "low",
        "low",
        "low",
        "low",
        "ambiguous",
        "ambiguous",
        "high",
        "high",
        "high",
        "high",
    ]


def test_consensus_strata_are_unchanged_when_source_outcomes_are_flipped():
    rows = _rows()
    assign_consensus_visual_strata(rows)
    before = [row["consensus_visual_stratum"] for row in rows]
    for row in rows:
        row["frozen_u_hit"], row["frozen_gc_hit"] = (
            row["frozen_gc_hit"],
            row["frozen_u_hit"],
        )
        row["frozen_delta"] *= -1

    assign_consensus_visual_strata(rows)

    assert [row["consensus_visual_stratum"] for row in rows] == before


def test_consensus_summary_recovers_observed_source_crossover():
    rows = _rows()
    assign_consensus_visual_strata(rows)

    summary = summarize_consensus_benchmark(rows)

    assert summary is not None
    assert summary["low_delta"] == pytest.approx(-1.0)
    assert summary["high_delta"] == pytest.approx(1.0)
    assert summary["interaction"] == pytest.approx(2.0)
    assert summary["low_u_recall"] == pytest.approx(1.0)
    assert summary["high_gc_recall"] == pytest.approx(1.0)


def test_discordant_counterfactual_ranks_remain_ambiguous():
    rows = _rows()
    rows[0][WRONG_MARGIN_FIELD] = 100.0

    assign_consensus_visual_strata(rows)

    assert rows[0]["mean_margin_percentile"] < 0.4
    assert rows[0]["wrong_margin_percentile"] >= 0.6
    assert rows[0]["consensus_visual_stratum"] == "ambiguous"


def test_distributional_jsd_can_define_strata_without_target_margin():
    rows = _rows()
    for index, row in enumerate(rows):
        row[MEAN_JSD_FIELD] = float(index) / 100.0
        row[WRONG_JSD_FIELD] = float(index) / 50.0
        # Reverse the target-margin order to prove that JSD, not margin, is
        # the selected source-outcome-blind signal.
        row[MEAN_MARGIN_FIELD] = float(9 - index)
        row[WRONG_MARGIN_FIELD] = float(9 - index)

    audit = assign_consensus_visual_strata(
        rows, visual_signal="distribution_jsd"
    )
    summary = summarize_consensus_benchmark(
        rows, visual_signal="distribution_jsd"
    )

    assert audit["visual_signal"] == "distribution_jsd"
    assert [row["consensus_visual_stratum"] for row in rows] == [
        "low",
        "low",
        "low",
        "low",
        "ambiguous",
        "ambiguous",
        "high",
        "high",
        "high",
        "high",
    ]
    assert summary is not None
    assert summary["low_delta"] == pytest.approx(-1.0)
    assert summary["high_delta"] == pytest.approx(1.0)


def test_distributional_jsd_rejects_unrecorded_context_alignment():
    with pytest.raises(ValueError, match="available only for current_span2"):
        assign_consensus_visual_strata(
            _rows(),
            visual_alignment="grounded_context",
            visual_signal="distribution_jsd",
        )


def test_pre_request_confident_u_population_is_outcome_blind():
    rows = _rows()
    for index, row in enumerate(rows):
        row["u_available_before_request"] = index != 0
        row["u_row_top_probability_before_request"] = (
            0.75 if index != 1 else 0.49
        )

    audit = assign_consensus_visual_strata(
        rows, source_population="pre_request_confident_u"
    )

    assert rows[0]["consensus_visual_stratum"] == "u_not_ready"
    assert rows[1]["consensus_visual_stratum"] == "u_not_ready"
    assert audit["source_population_states"] == 8
    assert audit["u_not_ready_states"] == 2
    before = [row["consensus_visual_stratum"] for row in rows]
    for row in rows:
        row["frozen_u_hit"], row["frozen_gc_hit"] = (
            row["frozen_gc_hit"],
            row["frozen_u_hit"],
        )
        row["frozen_delta"] *= -1
    assign_consensus_visual_strata(
        rows, source_population="pre_request_confident_u"
    )
    assert [row["consensus_visual_stratum"] for row in rows] == before


def test_consensus_triptych_renders_for_u_ready_population(tmp_path):
    point = {
        "low_mean_margin_drop": 0.1,
        "low_wrong_margin_drop": 0.2,
        "high_mean_margin_drop": 0.8,
        "high_wrong_margin_drop": 0.9,
        "low_u_recall": 0.7,
        "low_gc_recall": 0.5,
        "high_u_recall": 0.4,
        "high_gc_recall": 0.7,
        "low_delta": -0.2,
        "high_delta": 0.3,
    }
    intervals = {
        metric: [value - 0.05, value + 0.05]
        for metric, value in point.items()
    }
    payload = {
        "point_estimates": point,
        "cluster_bootstrap_95_ci": intervals,
        "analysis_role": "development",
        "benchmarks": ["ScienceQA"],
        "source_population": "pre_request_confident_u",
    }

    plot(payload, tmp_path / "triptych")

    assert (tmp_path / "triptych.pdf").stat().st_size > 0
    assert (tmp_path / "triptych.png").stat().st_size > 0


def test_grounded_context_alignment_uses_the_tokens_that_key_gc():
    rows = _rows()
    for index, row in enumerate(rows):
        changed = index >= 6
        row.update(
            {
                "g_available": True,
                "c_available": True,
                "visual_probe_context3_num_tokens": 3,
                "visual_probe_context3_target_token_ids": [10, 11, 12],
                "visual_probe_context3_full_top1_token_ids": [10, 11, 12],
                "visual_probe_context3_mean_target_margin_drops": [
                    float(index)
                ]
                * 3,
                "visual_probe_context3_wrong_target_margin_drops": [
                    float(index) * 2.0
                ]
                * 3,
                "visual_probe_context3_mean_top1_changed": [changed] * 3,
                "visual_probe_context3_wrong_top1_changed": [changed] * 3,
            }
        )
        # Deliberately reverse the current-token score; it must not define the
        # grounded-context strata.
        row[MEAN_MARGIN_FIELD] = float(9 - index)
        row[WRONG_MARGIN_FIELD] = float(9 - index)

    audit = assign_consensus_visual_strata(
        rows, visual_alignment="grounded_context"
    )
    summary = summarize_consensus_benchmark(
        rows, visual_alignment="grounded_context"
    )

    assert audit["visual_alignment"] == "grounded_context"
    assert [row["consensus_visual_stratum"] for row in rows] == [
        "low",
        "low",
        "low",
        "low",
        "ambiguous",
        "ambiguous",
        "high",
        "high",
        "high",
        "high",
    ]
    assert summary is not None
    assert summary["low_delta"] == pytest.approx(-1.0)
    assert summary["high_delta"] == pytest.approx(1.0)
