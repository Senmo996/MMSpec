import copy

from evaluation.analyze_candidate_visual_alignment_only import (
    assign_candidate_only_strata,
)


def _row(index: int) -> dict:
    return {
        "benchmark": "Toy",
        "frozen_eligible": True,
        "frozen_root_budget": 2,
        "visual_probe_full_top1_matches_target": True,
        "visual_probe_candidate_alignment_available": True,
        "visual_probe_candidate_alignment_budget": 2,
        "visual_probe_candidate_alignment_uses_target_outcome": False,
        "visual_probe_u_candidate_consensus_support": 0.0,
        "visual_probe_gc_candidate_consensus_support": index / 10.0,
        "visual_probe_gc_minus_u_candidate_visual_support": index / 10.0,
        # Deliberately reverse reliability to prove it is ignored.
        "u_row_top_probability": index / 10.0,
        "root_transition_top_probability": 1.0 - index / 10.0,
        "root_transition_context_order": 2,
        "frozen_u_hit": int(index % 2 == 0),
        "frozen_gc_hit": int(index % 3 == 0),
        "u_matched_accept": index % 4,
        "gc_matched_accept": index % 5,
    }


def test_candidate_only_assignment_ignores_reliability_and_outcomes():
    rows = [_row(index) for index in range(10)]
    audit = assign_candidate_only_strata(rows)
    labels = [row["arbitration_stratum"] for row in rows]

    changed = copy.deepcopy(rows)
    for row in changed:
        row["u_row_top_probability"] = 1.0 - row["u_row_top_probability"]
        row["root_transition_top_probability"] = (
            1.0 - row["root_transition_top_probability"]
        )
        row["frozen_u_hit"] = 1 - row["frozen_u_hit"]
        row["frozen_gc_hit"] = 1 - row["frozen_gc_hit"]
        row["u_matched_accept"] += 100
        row["gc_matched_accept"] -= 100
    assign_candidate_only_strata(changed)

    assert audit["uses_source_reliability"] is False
    assert audit["uses_source_hit_outcomes"] is False
    assert audit["uses_accepted_length_outcomes"] is False
    assert labels == [row["arbitration_stratum"] for row in changed]
    assert rows[0]["arbitration_stratum"] == "low"
    assert rows[-1]["arbitration_stratum"] == "high"


def test_candidate_only_assignment_accepts_sensitivity_tail_fraction():
    default_rows = [_row(index) for index in range(11)]
    wider_rows = copy.deepcopy(default_rows)

    default = assign_candidate_only_strata(default_rows)
    wider = assign_candidate_only_strata(wider_rows, tail_fraction=0.35)

    assert wider["tail_fraction"] == 0.35
    assert wider["by_benchmark"]["Toy"]["low"] > default["by_benchmark"]["Toy"]["low"]
    assert wider["by_benchmark"]["Toy"]["high"] > default["by_benchmark"]["Toy"]["high"]
