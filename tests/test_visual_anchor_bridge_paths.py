from evaluation.analyze_visual_anchor_bridge_paths import (
    build_path_cases,
    summarize_cases,
)


def _record(
    step,
    root,
    target,
    baseline,
    visual,
    *,
    counterfactual_history=True,
    high_visual=True,
):
    return {
        "question_id": "q1",
        "step_index": step,
        "root_token": root,
        "target_token": target,
        "has_counterfactual_history": counterfactual_history,
        "high_visual_state": high_visual,
        "grounding_score": 0.9 if high_visual else 0.1,
        "coverage": {
            "4": {
                "baseline_candidates": baseline,
                "cover_candidates": visual,
                "baseline_hit": target in baseline,
                "cover_hit": target in visual,
                "candidate_row_changed": baseline != visual,
            }
        },
    }


def test_visual_anchor_can_add_a_path_while_language_bridge_stays_full_view():
    current = _record(0, 10, 20, [21, 22, 23, 24], [20, 22, 23, 24])
    following = _record(1, 20, 30, [30, 31, 32, 33], [99, 31, 32, 33])
    groups = {("source.jsonl", "q1"): [current, following]}

    cases = build_path_cases(groups, anchor_budget=4, bridge_budget=4)
    summary = summarize_cases(cases)

    assert len(cases) == 1
    assert not cases[0]["baseline_path_hit"]
    assert cases[0]["visual_anchor_language_bridge_path_hit"]
    assert not cases[0]["recursive_cover_path_hit"]
    assert summary["added_paths"] == 1
    assert summary["lost_paths"] == 0
    assert summary["path_coverage_gain"] == 1.0
    assert summary["path_coverage_gain_pp_given_bridge_available"] == 100.0
    assert summary["oracle_bridge_path_gain_upper_bound_pp"] == 100.0


def test_self_transition_is_excluded_because_bridge_row_was_updated():
    current = _record(0, 10, 10, [10, 11, 12, 13], [10, 11, 12, 13])
    following = _record(1, 10, 20, [20, 21, 22, 23], [20, 21, 22, 23])
    groups = {("source.jsonl", "q1"): [current, following]}

    assert build_path_cases(groups, 4, 4, exclude_self_transitions=True) == []
    assert len(build_path_cases(groups, 4, 4, exclude_self_transitions=False)) == 1


def test_missing_language_row_counts_as_a_path_miss_for_both_methods():
    current = _record(0, 10, 20, [20, 21, 22, 23], [20, 21, 22, 23])
    following = _record(1, 20, 30, [30, 31, 32, 33], [30, 31, 32, 33])
    following["coverage"]["4"] = None
    groups = {("source.jsonl", "q1"): [current, following]}

    cases = build_path_cases(groups, 4, 4)

    assert len(cases) == 1
    assert not cases[0]["bridge_row_available"]
    assert not cases[0]["baseline_path_hit"]
    assert not cases[0]["visual_anchor_language_bridge_path_hit"]
