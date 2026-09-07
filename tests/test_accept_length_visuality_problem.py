import numpy as np
import pytest

from evaluation.analyze_accept_length_visuality_problem import (
    DUAL_FLIP,
    ONE_FLIP,
    STABLE,
    length_bin,
    summarize_benchmark,
    summarize_macro,
)


def _row(benchmark, accepted, visual, flip_class, index):
    return {
        "benchmark": benchmark,
        "cluster_id": f"{benchmark}:{index}",
        "accept_length": accepted,
        "length_bin": length_bin(accepted),
        "conservative_visual_dependence": visual,
        "flip_class": flip_class,
        "image_sensitive": flip_class != STABLE,
    }


def test_length_bins_are_fixed():
    assert [length_bin(value) for value in (0, 1, 2, 3, 4, 5, 9)] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5+",
        "5+",
    ]
    with pytest.raises(ValueError):
        length_bin(-1)


def test_summary_exposes_visual_decay_and_shorter_sensitive_paths():
    rows = []
    for benchmark in ("A", "B"):
        rows.extend(
            [
                _row(benchmark, 0, 0.9, DUAL_FLIP, 0),
                _row(benchmark, 1, 0.8, DUAL_FLIP, 1),
                _row(benchmark, 1, 0.6, ONE_FLIP, 2),
                _row(benchmark, 2, 0.5, ONE_FLIP, 3),
                _row(benchmark, 2, 0.4, STABLE, 4),
                _row(benchmark, 3, 0.3, STABLE, 5),
                _row(benchmark, 3, 0.2, STABLE, 6),
                _row(benchmark, 4, 0.1, STABLE, 7),
                _row(benchmark, 5, 0.0, STABLE, 8),
                # Keep the sensitive type represented at exactly length four.
                _row(benchmark, 4, 0.1, ONE_FLIP, 9),
                _row(benchmark, 5, 0.05, ONE_FLIP, 10),
            ]
        )

    benchmark_summary = summarize_benchmark(
        [row for row in rows if row["benchmark"] == "A"]
    )
    point, by_benchmark, missing = summarize_macro(rows, ("A", "B"))

    assert benchmark_summary is not None
    assert missing == []
    assert set(by_benchmark) == {"A", "B"}
    assert point["accept_1_visual_dependence"] > point[
        "accept_5plus_visual_dependence"
    ]
    assert point["accept_0_visual_dependence"] > point[
        "accept_1_visual_dependence"
    ]
    assert point["accept_1_sensitive_rate"] > point[
        "accept_5plus_sensitive_rate"
    ]
    assert point["stable_long_rate"] > point["sensitive_long_rate"]
    assert point["sensitive_reject_rate"] > point["stable_reject_rate"]
    assert point["rejected_vs_long_flip_risk_ratio"] > 1.0
    assert point["rejected_vs_long_visual_relative_decline"] > 0.0
    assert np.isclose(
        point["accept_1_stable_rate"]
        + point["accept_1_one_flip_rate"]
        + point["accept_1_dual_flip_rate"],
        1.0,
    )
