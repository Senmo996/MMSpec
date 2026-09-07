import numpy as np

from evaluation.analyze_candidate_alignment_quantiles import (
    assign_quantile_bins,
    signed_support_margin,
)


def test_assign_quantile_bins_covers_endpoints() -> None:
    rows = [
        {"arbitration_visual_rank": 0.0},
        {"arbitration_visual_rank": 0.19},
        {"arbitration_visual_rank": 0.20},
        {"arbitration_visual_rank": 0.99},
        {"arbitration_visual_rank": 1.0},
        {"arbitration_visual_rank": None},
    ]
    assert assign_quantile_bins(rows, 5) == 5
    assert [row["candidate_alignment_bin"] for row in rows] == [
        0,
        0,
        1,
        4,
        4,
        None,
    ]


def test_signed_support_margin_preserves_interval_order() -> None:
    payload = {
        "point_estimates": {"u_relative_support": [0.8, 0.3]},
        "cluster_bootstrap_95_ci": {
            "u_relative_support": [[0.7, 0.9], [0.2, 0.4]]
        },
    }
    values, intervals = signed_support_margin(payload)
    assert np.allclose(values, [-0.6, 0.4])
    assert np.allclose(intervals, [[-0.8, -0.4], [0.2, 0.6]])
