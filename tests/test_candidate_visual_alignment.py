import copy

from evaluation.analyze_candidate_visual_alignment import (
    assign_candidate_alignment_strata,
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
        "visual_probe_u_candidate_consensus_support": 0.2 - index * 0.01,
        "visual_probe_gc_candidate_consensus_support": index * 0.01,
        "visual_probe_gc_minus_u_candidate_visual_support": index * 0.02 - 0.2,
        "u_row_top_probability": 0.9 - index * 0.03,
        "root_transition_top_probability": 0.1 + index * 0.03,
        "root_transition_context_order": 2,
        "frozen_u_hit": int(index % 2 == 0),
        "frozen_gc_hit": int(index % 3 == 0),
        "u_matched_accept": index % 4,
        "gc_matched_accept": index % 5,
    }


def test_candidate_alignment_assignment_is_outcome_blind():
    rows = [_row(index) for index in range(12)]
    audit = assign_candidate_alignment_strata(rows)
    labels = [row["arbitration_stratum"] for row in rows]
    scores = [row["arbitration_score"] for row in rows]

    changed = copy.deepcopy(rows)
    for row in changed:
        row["frozen_u_hit"] = 1 - row["frozen_u_hit"]
        row["frozen_gc_hit"] = 1 - row["frozen_gc_hit"]
        row["u_matched_accept"] += 100
        row["gc_matched_accept"] -= 100
    assign_candidate_alignment_strata(changed)

    assert audit["uses_source_hit_outcomes"] is False
    assert audit["uses_accepted_length_outcomes"] is False
    assert audit["uses_target_token_id"] is False
    assert labels == [row["arbitration_stratum"] for row in changed]
    assert scores == [row["arbitration_score"] for row in changed]
    assert labels.count("low") > 0
    assert labels.count("high") > 0
    assert rows[0]["arbitration_stratum"] == "low"
    assert rows[-1]["arbitration_stratum"] == "high"


def test_candidate_alignment_rejects_budget_mismatch():
    rows = [_row(index) for index in range(12)]
    rows[0]["visual_probe_candidate_alignment_budget"] = 1
    audit = assign_candidate_alignment_strata(rows)

    assert rows[0]["arbitration_stratum"] == "candidate_budget_mismatch"
    assert audit["invalid_reasons"]["candidate_budget_mismatch"] == 1
    assert audit["valid_states"] == 11
