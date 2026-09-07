import numpy as np

from evaluation.analyze_candidate_pairwise_win_panel import (
    summarize_pairwise_win,
)


def _row(benchmark, stratum, u_accept, gc_accept):
    return {
        "benchmark": benchmark,
        "arbitration_stratum": stratum,
        "u_matched_accept": u_accept,
        "gc_matched_accept": gc_accept,
    }


def test_pairwise_win_uses_half_credit_for_ties_and_pairs_sum_to_one():
    rows = []
    for benchmark in ("A", "B"):
        rows.extend(
            [
                _row(benchmark, "low", 2, 1),
                _row(benchmark, "low", 1, 1),
                _row(benchmark, "high", 0, 2),
                _row(benchmark, "high", 1, 1),
            ]
        )

    point, by_benchmark = summarize_pairwise_win(rows)

    assert np.isclose(point["low_u_pairwise_win"], 0.75)
    assert np.isclose(point["low_gc_pairwise_win"], 0.25)
    assert np.isclose(point["high_u_pairwise_win"], 0.25)
    assert np.isclose(point["high_gc_pairwise_win"], 0.75)
    assert np.isclose(
        point["low_u_pairwise_win"] + point["low_gc_pairwise_win"], 1.0
    )
    assert np.isclose(
        point["high_u_pairwise_win"] + point["high_gc_pairwise_win"], 1.0
    )
    assert by_benchmark["A"]["low_tie_rate"] == 0.5
    assert by_benchmark["A"]["high_tie_rate"] == 0.5
