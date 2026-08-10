from evaluation.analyze_visual_lexical_backoff_paths import (
    build_path_cases,
    summarize_cases,
)


def _state(step, root, target, baseline, *, available, high=True):
    return {
        "question_id": "q1",
        "step_index": step,
        "root_token": root,
        "target_token": target,
        "baseline_candidates": baseline,
        "baseline_row_available": available,
        "high_visual_state": high,
        "grounding_score": 0.9 if high else 0.1,
    }


def test_visual_inventory_adds_first_hop_and_recycled_row_bridges_second_hop():
    key = ("run", 0)
    groups = {
        key: [
            _state(0, 10, 20, [], available=False),
            _state(1, 20, 30, [30, 31, 32, 33], available=True),
        ]
    }
    inventories = {
        key: {"inventories": {"visual_max": [20, 21, 22, 23]}}
    }

    cases = build_path_cases(groups, inventories, "visual_max", 4, 4, 4)
    summary = summarize_cases(cases)

    assert len(cases) == 1
    assert cases[0]["backoff_used"]
    assert cases[0]["first_hop_added"]
    assert cases[0]["path_added"]
    assert summary["path_coverage_gain_pp"] == 100.0
    assert summary["bridge_conversion_of_first_hop_additions"] == 1.0


def test_existing_recycled_row_is_never_replaced_or_lost():
    key = ("run", 0)
    groups = {
        key: [
            _state(0, 10, 20, [20, 21, 22, 23], available=True),
            _state(1, 20, 30, [30, 31, 32, 33], available=True),
        ]
    }
    inventories = {
        key: {"inventories": {"visual_max": [99, 98, 97, 96]}}
    }

    cases = build_path_cases(groups, inventories, "visual_max", 4, 4, 4)

    assert len(cases) == 1
    assert not cases[0]["backoff_used"]
    assert cases[0]["baseline_path_hit"]
    assert cases[0]["backoff_path_hit"]
    assert not cases[0]["first_hop_lost"]
    assert not cases[0]["path_lost"]


def test_self_transition_is_excluded_from_temporal_bridge_reconstruction():
    key = ("run", 0)
    groups = {
        key: [
            _state(0, 10, 10, [], available=False),
            _state(1, 10, 20, [20, 21, 22, 23], available=True),
        ]
    }
    inventories = {
        key: {"inventories": {"visual_max": [10, 11, 12, 13]}}
    }

    assert build_path_cases(groups, inventories, "visual_max", 4, 4, 4) == []
    assert len(
        build_path_cases(
            groups,
            inventories,
            "visual_max",
            4,
            4,
            4,
            exclude_self_transitions=False,
        )
    ) == 1
