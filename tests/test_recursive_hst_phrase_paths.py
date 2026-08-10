from evaluation.analyze_recursive_hst_phrase_paths import (
    build_phrase_policy_candidates,
    fuse_sources,
    prompt_lookup_candidates,
    select_dev_action,
)


def test_prompt_lookup_uses_longest_suffix_and_recent_successors():
    source = [1, 2, 3, 4, 2, 3, 5]

    assert prompt_lookup_candidates(
        source, [9, 2, 3], max_ngram=4, limit=3
    ) == [5, 4]
    assert prompt_lookup_candidates(
        source, [9, 3], max_ngram=4, limit=1
    ) == [5]
    assert prompt_lookup_candidates(
        source, [9, 8], max_ngram=4, limit=3
    ) == []


def test_prompt_lookup_filters_special_successors():
    assert prompt_lookup_candidates(
        [1, 2, 9, 1, 2, 3],
        [1, 2],
        max_ngram=2,
        limit=3,
        excluded_token_ids=[9],
    ) == [3]


def test_phrase_fusion_keeps_width_and_fills_missing_sources():
    assert fuse_sources(
        [[1, 2, 3], [2, 4], [5]], [1, 1, 1], width=3
    ) == [1, 2, 5]
    policies = build_phrase_policy_candidates(
        bridge=[], phrase=[7], hybrid=[8, 9, 10], width=3
    )
    assert policies["bridge2_phrase1"] == [7, 8, 9]
    assert policies["phrase_only"] == [7]


def test_dev_action_selection_applies_first_path_and_specificity_gates():
    dev = {
        "configs": ["good", "generic"],
        "first_hop_metrics": {
            "good": {"top3_hit_rate": 0.2},
            "generic": {"top3_hit_rate": 0.3},
        },
        "paired_specificity": {
            "good": {"observed_minus_control_pp": 1.0},
            "generic": {"observed_minus_control_pp": -1.0},
        },
        "phrase_path_metrics": {
            "good": {
                "phrase_only": {
                    "path_hit_rate": 0.06,
                    "phrase_available_given_first_rate": 0.5,
                }
            },
            "generic": {
                "phrase_only": {
                    "path_hit_rate": 0.1,
                    "phrase_available_given_first_rate": 0.5,
                }
            },
        },
    }

    config, policy, eligible = select_dev_action(
        dev, min_top3_hit_rate=0.15, min_path_hit_rate=0.05
    )

    assert (config, policy) == ("good", "phrase_only")
    assert len(eligible) == 1
