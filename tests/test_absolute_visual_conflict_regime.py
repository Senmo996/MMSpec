from evaluation.analyze_absolute_visual_conflict_regime import (
    HIGH_MIN,
    LOW_MAX,
    macro_summary,
    summarize_absolute_benchmark,
)

import pytest


def row(score, delta, *, eligible=True):
    return {
        "frozen_visual_score": score,
        "frozen_eligible": eligible,
        "u_available": True,
        "gc_available": True,
        "frozen_u_hit": int(delta < 0),
        "frozen_gc_hit": int(delta > 0),
        "frozen_delta": delta,
    }


def test_absolute_bands_do_not_depend_on_relative_rank():
    rows = [row(LOW_MAX, -1), row(HIGH_MIN, 1)]

    summary = summarize_absolute_benchmark(rows, minimum_tail_states=1)

    assert summary is not None
    assert summary["low_delta"] == -1
    assert summary["high_delta"] == 1
    assert summary["interaction"] == 2


def test_absolute_bands_enforce_support_after_eligibility():
    rows = [row(0.1, -1, eligible=False), row(0.9, 1)]

    assert summarize_absolute_benchmark(rows, minimum_tail_states=1) is None


def test_macro_summary_requires_every_frozen_benchmark():
    rows = []
    for _ in range(8):
        low = row(0.1, -1)
        low["benchmark"] = "ScienceQA"
        high = row(0.9, 1)
        high["benchmark"] = "ScienceQA"
        rows.extend((low, high))

    with pytest.raises(ValueError, match="lack frozen support"):
        macro_summary(rows)
