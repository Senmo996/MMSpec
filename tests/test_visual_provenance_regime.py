import pytest

from evaluation.analyze_visual_provenance_regime import (
    assign_visual_provenance_strata,
    summarize_benchmark,
)


def _row(position, mean_jsd, wrong_jsd, *, eligible=True):
    return {
        "analysis_split": "development",
        "benchmark": "ScienceQA",
        "question_id": "scienceqa:test",
        "choice_index": 0,
        "turn_index": 0,
        "iteration": position,
        "output_position": position,
        "frozen_eligible": eligible,
        "visual_probe_full_top1_matches_target": True,
        "visual_probe_mean_jsd": mean_jsd,
        "visual_probe_wrong_jsd": wrong_jsd,
        "frozen_u_hit": 0,
        "frozen_gc_hit": 1,
        "u_matched_accept": 0,
        "gc_matched_accept": 1,
    }


def test_recent_visual_evidence_is_carried_into_an_instantaneously_low_state():
    rows = [
        _row(0, 0.04, 0.04),
        _row(4, 0.001, 0.001),
        _row(13, 0.001, 0.001),
    ]

    audit = assign_visual_provenance_strata(rows)

    assert [row["visual_provenance_stratum"] for row in rows] == [
        "high",
        "high",
        "low",
    ]
    assert rows[1]["visual_current_union_jsd"] == pytest.approx(0.001)
    assert rows[1]["visual_provenance_consensus_jsd"] == pytest.approx(0.02)
    assert audit["instantaneous_low_states"] == 2
    assert audit["instantaneous_low_relabeled_high"] == 1


def test_provenance_label_is_outcome_blind():
    rows = [
        _row(0, 0.001, 0.001),
        _row(12, 0.04, 0.04),
    ]
    assign_visual_provenance_strata(rows)
    before = [row["visual_provenance_stratum"] for row in rows]

    for row in rows:
        row["frozen_u_hit"], row["frozen_gc_hit"] = (
            row["frozen_gc_hit"],
            row["frozen_u_hit"],
        )
        row["u_matched_accept"], row["gc_matched_accept"] = (
            row["gc_matched_accept"],
            row["u_matched_accept"],
        )
    assign_visual_provenance_strata(rows)

    assert [row["visual_provenance_stratum"] for row in rows] == before


def test_consensus_high_and_union_low_leave_disagreement_ambiguous():
    rows = [
        _row(0, 0.03, 0.001),
        _row(20, 0.001, 0.001),
        _row(40, 0.03, 0.03),
    ]

    assign_visual_provenance_strata(rows)

    assert [row["visual_provenance_stratum"] for row in rows] == [
        "ambiguous",
        "low",
        "high",
    ]


def test_summary_reports_root_and_acceptance_crossovers():
    low = [_row(index, 0.001, 0.001) for index in range(4)]
    high = [_row(100 + index, 0.04, 0.04) for index in range(4)]
    for row in low:
        row["frozen_u_hit"] = 1
        row["frozen_gc_hit"] = 0
        row["u_matched_accept"] = 2
        row["gc_matched_accept"] = 0
    for row in high:
        row["frozen_u_hit"] = 0
        row["frozen_gc_hit"] = 1
        row["u_matched_accept"] = 0
        row["gc_matched_accept"] = 2
    rows = low + high
    assign_visual_provenance_strata(rows)

    summary = summarize_benchmark(rows)

    assert summary is not None
    assert summary["low_root_delta"] == pytest.approx(-1.0)
    assert summary["high_root_delta"] == pytest.approx(1.0)
    assert summary["low_accept_delta"] == pytest.approx(-2.0)
    assert summary["high_accept_delta"] == pytest.approx(2.0)
