from evaluation.analyze_conflict_conditioned_tail_regime import (
    assign_conflict_reference_percentiles,
)


def test_percentiles_use_only_eligible_conflicts_without_hits():
    rows = [
        {
            "benchmark": "MMSpec",
            "frozen_eligible": True,
            "frozen_visual_score": 0.0,
        },
        {
            "benchmark": "MMSpec",
            "frozen_eligible": True,
            "frozen_visual_score": 1.0,
        },
        {
            "benchmark": "MMSpec",
            "frozen_eligible": False,
            "frozen_visual_score": 0.5,
        },
    ]

    audit = assign_conflict_reference_percentiles(rows)

    assert rows[0]["frozen_visual_percentile"] == 0.25
    assert rows[1]["frozen_visual_percentile"] == 0.75
    assert rows[2]["frozen_visual_percentile"] == 0.5
    assert audit["states_by_benchmark"] == {"MMSpec": 2}
    assert audit["uses_source_hit_outcomes"] is False


def test_ties_receive_average_empirical_percentile():
    rows = [
        {
            "benchmark": "MMSpec",
            "frozen_eligible": True,
            "frozen_visual_score": value,
        }
        for value in (0.0, 0.0, 1.0, 1.0)
    ]

    assign_conflict_reference_percentiles(rows)

    assert [row["frozen_visual_percentile"] for row in rows] == [
        0.25,
        0.25,
        0.75,
        0.75,
    ]
