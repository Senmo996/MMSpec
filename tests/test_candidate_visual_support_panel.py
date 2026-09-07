import numpy as np

from evaluation.analyze_candidate_visual_alignment import (
    GC_SUPPORT_FIELD,
    U_SUPPORT_FIELD,
)
from evaluation.analyze_candidate_visual_support_panel import summarize_support


def test_support_summary_has_four_source_by_stratum_values():
    rows = []
    for benchmark, shift in (("A", 0.0), ("B", 1.0)):
        for stratum, u_value, gc_value in (
            ("low", 2.0 + shift, -1.0 + shift),
            ("high", -2.0 + shift, 3.0 + shift),
        ):
            for index in range(2):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "arbitration_stratum": stratum,
                        U_SUPPORT_FIELD: u_value + index,
                        GC_SUPPORT_FIELD: gc_value + index,
                    }
                )

    point, by_benchmark = summarize_support(rows)

    assert set(point) == {
        "low_u_support",
        "low_gc_support",
        "high_u_support",
        "high_gc_support",
        "low_u_relative_support",
        "low_gc_relative_support",
        "high_u_relative_support",
        "high_gc_relative_support",
    }
    assert np.isclose(point["low_u_support"], 3.0)
    assert np.isclose(point["low_gc_support"], 0.0)
    assert np.isclose(point["high_u_support"], -1.0)
    assert np.isclose(point["high_gc_support"], 4.0)
    assert np.isclose(
        point["low_u_relative_support"] + point["low_gc_relative_support"],
        1.0,
    )
    assert np.isclose(
        point["high_u_relative_support"] + point["high_gc_relative_support"],
        1.0,
    )
    assert point["low_u_relative_support"] > point["low_gc_relative_support"]
    assert point["high_gc_relative_support"] > point["high_u_relative_support"]
    assert set(by_benchmark) == {"A", "B"}
