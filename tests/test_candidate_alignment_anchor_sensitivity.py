import copy

from evaluation.audit_candidate_alignment_anchor_sensitivity import (
    assign_without_anchor_filter,
)


def _row(index: int, anchor_matches: bool) -> dict:
    return {
        "benchmark": "Toy",
        "frozen_eligible": True,
        "frozen_root_budget": 2,
        "visual_probe_full_top1_matches_target": anchor_matches,
        "visual_probe_candidate_alignment_available": True,
        "visual_probe_candidate_alignment_budget": 2,
        "visual_probe_candidate_alignment_uses_target_outcome": False,
        "visual_probe_u_candidate_consensus_support": 0.0,
        "visual_probe_gc_candidate_consensus_support": index / 10.0,
        "visual_probe_gc_minus_u_candidate_visual_support": index / 10.0,
        "u_row_top_probability": 0.8,
        "root_transition_top_probability": 0.2,
        "root_transition_context_order": 2,
        "frozen_u_hit": index % 2,
        "frozen_gc_hit": index % 3,
        "u_matched_accept": index,
        "gc_matched_accept": 10 - index,
    }


def test_anchor_sensitivity_assignment_includes_failed_anchor_and_is_outcome_blind():
    rows = [_row(index, anchor_matches=index != 0) for index in range(10)]
    audit = assign_without_anchor_filter(rows)
    labels = [row["arbitration_stratum"] for row in rows]

    changed = copy.deepcopy(rows)
    for row in changed:
        row["visual_probe_full_top1_matches_target"] = not row[
            "visual_probe_full_top1_matches_target"
        ]
        row["frozen_u_hit"] = 1 - row["frozen_u_hit"]
        row["frozen_gc_hit"] = 1 - int(bool(row["frozen_gc_hit"]))
        row["u_matched_accept"] += 100
        row["gc_matched_accept"] -= 100
    assign_without_anchor_filter(changed)

    assert rows[0]["arbitration_stratum"] == "low"
    assert audit["by_benchmark"]["Toy"]["failed_full_anchor_states_included"] == 1
    assert labels == [row["arbitration_stratum"] for row in changed]
