import math
from types import SimpleNamespace

import pytest
import torch

from method.sam_grounded.controller import (
    GroundedDraftController,
    VisualGroundingCalibrator,
    confidence_from_logits,
)
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel
from evaluation.eval_sam_grounded_mmspec import _parse_policies, _select_topic_indices


def test_visual_grounding_calibration_separates_visual_and_text_states():
    prompt_hidden = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    visual_mask = torch.tensor([True, True, False, False])
    calibrator = VisualGroundingCalibrator.from_prompt(prompt_hidden, visual_mask)

    assert calibrator is not None
    assert calibrator.score(torch.tensor([1.0, 0.0])) > 0.9
    assert calibrator.score(torch.tensor([0.0, 1.0])) < 0.1


def test_visual_soft_budget_decreases_monotonically():
    controller = GroundedDraftController(
        "visual-soft", min_draft_tokens=2, max_draft_tokens=40
    )
    budgets = [controller.decide(score).budget for score in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[0] == 40
    assert budgets[-1] == 2


def test_target_and_fixed_policies_are_exact_caps():
    target = GroundedDraftController("target", min_draft_tokens=2, max_draft_tokens=40)
    fixed = GroundedDraftController("fixed", min_draft_tokens=2, max_draft_tokens=40)
    broad = GroundedDraftController("broad", min_draft_tokens=2, max_draft_tokens=40)
    assert target.decide(0.0).budget == 0
    assert target.decide(1.0).budget == 0
    assert fixed.decide(0.0).budget == 40
    assert fixed.decide(1.0).budget == 40
    assert broad.decide(0.5).budget == 40


def test_cover_policy_prefix_validates_underlying_tree_policy():
    assert _parse_policies("cover-visual-wide-plus2,broad") == [
        "cover-visual-wide-plus2",
        "broad",
    ]
    with pytest.raises(Exception):
        _parse_policies("cover-does-not-exist")


def test_trigram_node_budget_policies_parse_and_cap_verifier_batch():
    policies = [
        f"context-score-trigram-deeper-wide-plus4-node{budget}"
        for budget in (23, 31, 39, 47, 55)
    ]
    assert _parse_policies(",".join(policies)) == policies
    for policy, expected in zip(policies, (23, 31, 39, 47, 55)):
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 63, None
        ) == expected
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 15, 0.99
        ) == 15


def test_persistent_fusion_node_budget_policies_parse_and_cap_verifier_batch():
    policies = [
        f"context-score-trigram-fusion-persistent-node{budget}-deepest-wide-plus4"
        for budget in (39, 47, 55, 63, 79, 95)
    ]
    assert _parse_policies(",".join(policies)) == policies
    for policy, expected in zip(policies, (39, 47, 55, 63, 79, 95)):
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 95, None
        ) == expected
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 31, 0.99
        ) == 31


def test_persistent_committed_policy_parses_and_keeps_full_budget():
    policy = (
        "context-score-trigram-fusion-persistent-committed-"
        "deepest-wide-plus4"
    )
    assert _parse_policies(policy) == [policy]
    assert TreeRecyclingSpecModel._effective_tree_node_budget(
        policy, 63, 0.99
    ) == 63


def test_persistent_stable_policies_parse_and_cap_verifier_batch():
    policies = [
        f"context-score-trigram-fusion-persistent-stable-node{budget}-deepest-wide-plus4"
        for budget in (55, 63)
    ]
    assert _parse_policies(",".join(policies)) == policies
    for policy, expected in zip(policies, (55, 63)):
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 95, None
        ) == expected


def test_persistent_depth_policies_parse_and_cap_configured_budget():
    configs = (
        (7, 63),
        (8, 55),
        (8, 63),
        (10, 47),
        (10, 55),
        (10, 63),
        (10, 79),
        (10, 95),
    )
    policies = [
        f"context-score-trigram-fusion-persistent-depth{depth}-node{budget}-wide-plus4"
        for depth, budget in configs
    ]
    assert _parse_policies(",".join(policies)) == policies
    for policy, (_, expected_budget) in zip(policies, configs):
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 95, None
        ) == expected_budget


def test_persistent_adaptive95_policy_expands_only_mid_confidence_roots():
    policy = (
        "context-score-trigram-fusion-persistent-adaptive95-"
        "depth10-wide-plus4"
    )
    assert _parse_policies(policy) == [policy]
    choose = TreeRecyclingSpecModel._effective_tree_node_budget
    assert choose(policy, 95, None) == 63
    assert choose(policy, 95, 0.39) == 63
    assert choose(policy, 95, 0.40) == 95
    assert choose(policy, 95, 0.69) == 95
    assert choose(policy, 95, 0.70) == 63


def test_persistent_optimized_depth_policies_parse_and_use_node63_depth10():
    policies = [
        "context-score-trigram-fusion-persistent-contextcal-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-shadow-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-hotpath-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-hotpath-cpp-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes95-hotpath-cpp-"
        "depth10-node95-wide-plus4",
    ]
    assert _parse_policies(",".join(policies)) == policies
    for policy in policies[:2]:
        assert TreeRecyclingSpecModel._effective_tree_node_budget(
            policy, 95, None
        ) == 63
    choose_budget = TreeRecyclingSpecModel._effective_tree_node_budget
    assert choose_budget(policies[2], 95, 0.5, 1) == 47
    assert choose_budget(policies[2], 95, 0.5, 2) == 63
    assert choose_budget(policies[3], 95, 0.5, 1) == 47
    assert choose_budget(policies[3], 95, 0.5, 3) == 63
    assert choose_budget(policies[4], 95, 0.5, 1) == 47
    assert choose_budget(policies[4], 95, 0.5, 2) == 63
    assert choose_budget(policies[4], 95, 0.5, 3) == 63
    assert choose_budget(policies[5], 95, 0.5, 1) == 47
    assert choose_budget(policies[5], 95, 0.5, 2) == 63
    assert choose_budget(policies[5], 95, 0.5, 3) == 95

    choose_masses = TreeRecyclingSpecModel._score_priority_hit_masses
    assert choose_masses(policies[0], 3)[:3] == (0.78, 0.76, 0.68)
    assert choose_masses(policies[0], 2)[:3] == (0.74, 0.72, 0.64)
    assert choose_masses(policies[0], 1)[:3] == (0.68, 0.69, 0.60)
    assert choose_masses(policies[1], 3) is None


def test_prompt_transition_row_selection_can_skip_persistent_hits():
    choose = TreeRecyclingSpecModel._select_prompt_transition_rows
    assert choose([7, 8, 7, 9]) == ([0, 1, 3], [7, 8, 9])
    assert choose([7, 8, 7, 9], cached_tokens={8, 9}) == ([0], [7])


def test_existing_adaptive_node_budgets_keep_their_thresholds():
    choose = TreeRecyclingSpecModel._effective_tree_node_budget
    assert choose("context-score-adaptive-safe-deeper-wide-plus2", 63, None) == 63
    assert choose("context-score-adaptive-safe-deeper-wide-plus2", 63, 0.84) == 63
    assert choose("context-score-adaptive-safe-deeper-wide-plus2", 63, 0.85) == 47
    assert choose("score-adaptive-deeper-wide-plus2", 63, 0.74) == 63
    assert choose("score-adaptive-deeper-wide-plus2", 63, 0.75) == 47
    assert choose("score-adaptive-deeper-wide-plus2", 63, 0.90) == 31


def test_reset_persistent_recycling_cache_clears_warmup_state():
    model = TreeRecyclingSpecModel.__new__(TreeRecyclingSpecModel)
    object.__setattr__(
        model,
        "_gwtr_persistent_unigram_caches",
        {"policy": {"transitions": {1: [2]}, "scores": {1: [1.0]}}},
    )
    object.__setattr__(
        model,
        "_gwtr_persistent_unigram_banks",
        {
            "policy": {
                "transitions": {1: [2]},
                "scores": {1: [1.0]},
                "counts": {1: 1},
            }
        },
    )

    model.reset_persistent_recycling_cache()

    assert model._gwtr_persistent_unigram_caches == {}
    assert model._gwtr_persistent_unigram_banks == {}


def test_persistent_transition_merge_is_bounded_and_rewards_agreement():
    row, scores, count = TreeRecyclingSpecModel._merge_persistent_transition_row(
        [1, 2, 3],
        [0.6, 0.3, 0.1],
        3,
        [2, 4, 1],
        [0.5, 0.3, 0.2],
        3,
    )

    assert row == [1, 2, 3]
    assert count == 4
    assert sum(scores) == pytest.approx(1.0)


def test_bounded_transition_row_refreshes_and_evicts_oldest():
    rows = {"old": [1], "keep": [2]}
    scores = {"old": [1.0], "keep": [1.0]}

    TreeRecyclingSpecModel._store_bounded_transition_row(
        rows, scores, "new", [3], [1.0], 2
    )
    TreeRecyclingSpecModel._store_bounded_transition_row(
        rows, scores, "keep", [4], [1.0], 2
    )

    assert list(rows) == ["new", "keep"]
    assert rows["keep"] == [4]
    assert set(scores) == {"new", "keep"}


def test_visual_lexical_backoff_policies_parse_and_route_only_unseen_visual_roots():
    assert _parse_policies(
        "visual-lexical-backoff,visual-hst-backoff,visual-hst-backoff-gated,visual-wide-plus2-hst-backoff,visual-wide-plus2-hst-backoff-gated,visual-rootwide-plus2,visual-rootwide-plus2-hst-backoff,visual-wide-plus2-vli-backoff"
    ) == [
        "visual-lexical-backoff",
        "visual-hst-backoff",
        "visual-hst-backoff-gated",
        "visual-wide-plus2-hst-backoff",
        "visual-wide-plus2-hst-backoff-gated",
        "visual-rootwide-plus2",
        "visual-rootwide-plus2-hst-backoff",
        "visual-wide-plus2-vli-backoff",
    ]
    route = TreeRecyclingSpecModel._visual_lexical_backoff_active
    assert route("visual-lexical-backoff", 0.8, 0.55, root_row_valid=False)
    assert route(
        "visual-wide-plus2-vli-backoff", 0.8, 0.55, root_row_valid=False
    )
    assert route("visual-hst-backoff", 0.8, 0.55, root_row_valid=False)
    assert route("visual-hst-backoff-gated", 0.8, 0.55, root_row_valid=False)
    assert route(
        "visual-wide-plus2-hst-backoff", 0.8, 0.55, root_row_valid=False
    )
    assert route(
        "visual-wide-plus2-hst-backoff-gated",
        0.8,
        0.55,
        root_row_valid=False,
    )
    assert route(
        "visual-rootwide-plus2-hst-backoff",
        0.8,
        0.55,
        root_row_valid=False,
    )
    assert not route("visual-lexical-backoff", 0.4, 0.55, root_row_valid=False)
    assert not route("visual-lexical-backoff", 0.8, 0.55, root_row_valid=True)
    assert not route("visual-hst-backoff", 0.8, 0.55, root_row_valid=True)
    assert not route("broad", 0.8, 0.55, root_row_valid=False)


def test_visual_accept_reacts_to_online_rejections():
    controller = GroundedDraftController(
        "visual-accept",
        min_draft_tokens=2,
        max_draft_tokens=40,
        acceptance_ema_decay=0.5,
    )
    before = controller.decide(0.2).budget
    controller.observe(accepted_tokens=0, proposed_tokens=10)
    after = controller.decide(0.2).budget
    assert after < before


def test_visual_anchor_only_shortens_ambiguous_grounded_states():
    controller = GroundedDraftController(
        "visual-anchor",
        min_draft_tokens=2,
        max_draft_tokens=40,
        visual_threshold=0.55,
        confidence_threshold=0.75,
    )
    assert controller.decide(grounding_score=0.8, confidence=0.2).budget == 2
    assert controller.decide(grounding_score=0.8, confidence=0.9).budget == 40
    assert controller.decide(grounding_score=0.2, confidence=0.2).budget == 40


def test_confidence_from_logits_uses_top_two_margin():
    low = confidence_from_logits(torch.tensor([1.0, 0.99, -2.0]))
    high = confidence_from_logits(torch.tensor([10.0, 0.0, -2.0]))
    assert 0.0 <= low < high <= 1.0
    assert math.isfinite(low)


def test_calibrator_returns_none_without_both_modalities():
    hidden = torch.randn(4, 8)
    assert VisualGroundingCalibrator.from_prompt(
        hidden, torch.zeros(4, dtype=torch.bool)
    ) is None


def test_calibrator_vectorized_scores_match_scalar_scores():
    hidden = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    calibrator = VisualGroundingCalibrator.from_prompt(
        hidden, torch.tensor([True, True, False, False])
    )
    vectorized = calibrator.scores(hidden)
    scalar = torch.tensor([calibrator.score(row) for row in hidden])
    assert torch.allclose(vectorized.cpu(), scalar, atol=1e-6)
    assert VisualGroundingCalibrator.from_prompt(
        hidden, torch.ones(4, dtype=torch.bool)
    ) is None


def test_final_hidden_state_fast_path_preserves_explicit_layer_lookup():
    final_hidden = torch.randn(1, 3, 4)
    layer_zero = torch.randn(1, 3, 4)
    output = SimpleNamespace(
        last_hidden_state=final_hidden,
        hidden_states=(layer_zero, torch.zeros_like(final_hidden)),
    )

    assert TreeRecyclingSpecModel._layer_hidden(output, -1) is final_hidden
    assert TreeRecyclingSpecModel._layer_hidden(output, 0) is layer_zero

    fallback = SimpleNamespace(hidden_states=(layer_zero, final_hidden))
    assert TreeRecyclingSpecModel._layer_hidden(fallback, -1) is final_hidden


def test_layerwise_verification_diagnostics_localizes_first_difference():
    shared = torch.tensor([[[1.0, 2.0, 3.0]]])
    packed_layer = torch.tensor([[[1.0, 2.5, 3.0]]])
    single_layer = torch.tensor([[[1.0, 2.0, 3.0]]])
    packed = SimpleNamespace(
        hidden_states=(shared, packed_layer),
        logits=torch.tensor([[[0.0, 2.0, 1.5]]]),
    )
    single = SimpleNamespace(
        hidden_states=(shared, single_layer),
        logits=torch.tensor([[[0.0, 1.5, 2.0]]]),
    )

    result = TreeRecyclingSpecModel._layerwise_verification_diagnostics(
        packed, single
    )

    assert result["num_hidden_states"] == 2
    assert result["first_different_layer"] == 1
    assert result["layers"][0]["max_abs"] == 0.0
    assert result["layers"][1]["max_abs"] == pytest.approx(0.5)
    assert result["logits"]["packed_top_ids"] == [1, 2]
    assert result["logits"]["single_top_ids"] == [2, 1]
    assert not result["logits"]["same_argmax"]


def test_controller_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        GroundedDraftController("visual-soft", min_draft_tokens=5, max_draft_tokens=4)


def test_tree_shape_uses_configured_visual_thresholds():
    common = (0.6, 0.4, 0.24, 0.7, 0.3, 2, 4, 4, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-hard", *common) == (2, 4)
    assert TreeRecyclingSpecModel._tree_shape("grounded-residual", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("grounded-residual-reverse", *common) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-wide-plus2", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-wide-plus2-vli-backoff", *common
    ) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-wide-plus2-reverse", *common) == (6, 2)

    common = (0.8, 0.4, 0.32, 0.7, 0.3, 2, 4, 4, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-hard", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-anchor", *common) == (2, 4)
    assert TreeRecyclingSpecModel._tree_shape("broad", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-lexical-backoff", *common
    ) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-hst-backoff", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-hst-backoff-gated", *common
    ) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-wide-plus2-hst-backoff", *common
    ) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-wide-plus2-hst-backoff-gated", *common
    ) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-rootwide-plus2", *common
    ) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-rootwide-plus2-hst-backoff", *common
    ) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape("wide-plus2", *common) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "rank-prior-wide-plus2", *common
    ) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "rank-prior-deep-wide-plus2", *common
    ) == (6, 3)
    assert TreeRecyclingSpecModel._tree_shape(
        "rank-prior-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "rank-prior-deepest-wide-plus2", *common
    ) == (6, 5)
    assert TreeRecyclingSpecModel._tree_shape(
        "score-prior-deep-wide-plus2", *common
    ) == (6, 3)
    assert TreeRecyclingSpecModel._tree_shape(
        "score-prior-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus4", *common
    ) == (8, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-deeper-wide-plus4", *common
    ) == (8, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-residual2-deeper-wide-plus4", *common
    ) == (8, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-residual2-deepest-wide-plus4", *common
    ) == (8, 5)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-fusion-deeper-wide-plus4", *common
    ) == (8, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-fusion-deepest-wide-plus4", *common
    ) == (8, 5)
    for policy in (
        "context-score-trigram-fusion55-deepest-wide-plus4",
        "context-score-trigram-fusion-adaptive-deepest-wide-plus4",
        "context-score-trigram-fusion-calibrated-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-committed-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-stable-node55-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-stable-node63-deepest-wide-plus4",
        "context-score-trigram-fusion-bank-deepest-wide-plus4",
        "context-score-trigram-fusion-bank-global15-deepest-wide-plus4",
    ):
        assert TreeRecyclingSpecModel._tree_shape(policy, *common) == (8, 5)
    for budget in (39, 47, 55, 63, 79, 95):
        assert TreeRecyclingSpecModel._tree_shape(
            f"context-score-trigram-fusion-persistent-node{budget}-deepest-wide-plus4",
            *common,
        ) == (8, 5)
    for depth, budget in (
        (7, 63),
        (8, 55),
        (8, 63),
        (10, 47),
        (10, 55),
        (10, 63),
        (10, 79),
        (10, 95),
    ):
        assert TreeRecyclingSpecModel._tree_shape(
            f"context-score-trigram-fusion-persistent-depth{depth}-node{budget}-wide-plus4",
            *common,
        ) == (8, depth)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-fusion-persistent-adaptive95-"
        "depth10-wide-plus4",
        *common,
    ) == (8, 10)
    for policy in (
        "context-score-trigram-fusion-persistent-contextcal-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-shadow-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-hotpath-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-hotpath-cpp-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes95-hotpath-cpp-"
        "depth10-node95-wide-plus4",
    ):
        assert TreeRecyclingSpecModel._tree_shape(
            policy, *common
        ) == (8, 10)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-fusion-global7-deepest-wide-plus4", *common
    ) == (8, 5)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-fusion-global15-deepest-wide-plus4", *common
    ) == (8, 5)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-deepest-wide-plus4", *common
    ) == (8, 5)
    for budget in (23, 31, 39, 47, 55):
        assert TreeRecyclingSpecModel._tree_shape(
            f"context-score-trigram-deeper-wide-plus4-node{budget}", *common
        ) == (8, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-deeper-wide-plus6", *common
    ) == (10, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-trigram-deeper-wide-plus8", *common
    ) == (12, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-fourgram-deeper-wide-plus4", *common
    ) == (8, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-fourgram-deepest-wide-plus4", *common
    ) == (8, 5)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus5", *common
    ) == (9, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus6", *common
    ) == (10, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus8", *common
    ) == (12, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-calibrated-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus2-node55", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-prior-deeper-wide-plus2-node47", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "context-score-adaptive-safe-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "score-prior-deepest-wide-plus2", *common
    ) == (6, 5)
    assert TreeRecyclingSpecModel._tree_shape(
        "score-prior-maxdeep-wide-plus2", *common
    ) == (6, 9)
    assert TreeRecyclingSpecModel._tree_shape(
        "score-adaptive-safe-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape(
        "score-adaptive-deeper-wide-plus2", *common
    ) == (6, 4)
    assert TreeRecyclingSpecModel._tree_shape("visual-wide-plus2", *common) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape(
        "visual-wide-plus2-vli-backoff", *common
    ) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-wide-plus2-reverse", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("narrow", *common) == (2, 2)
    assert TreeRecyclingSpecModel._tree_shape("spine", *common) == (4, 4)
    assert TreeRecyclingSpecModel._tree_shape("visual-reverse", *common) == (2, 4)
    assert TreeRecyclingSpecModel._tree_shape("visual-spine", *common) == (4, 4)
    assert TreeRecyclingSpecModel._tree_shape("visual-spine-reverse", *common) == (2, 4)
    assert TreeRecyclingSpecModel._tree_shape("modal-broad", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("modal-hard", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("modal-spine", *common) == (4, 4)
    assert TreeRecyclingSpecModel._tree_shape("grounded-backoff", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("grounded-backoff-reverse", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("grounded-residual", *common) == (6, 2)
    assert TreeRecyclingSpecModel._tree_shape("grounded-residual-reverse", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("hybrid", *common) == (4, 4)
    assert TreeRecyclingSpecModel._tree_shape("modal-hybrid", *common) == (4, 4)
    assert TreeRecyclingSpecModel._tree_shape("visual-hybrid", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("grounded-hybrid", *common) == (4, 4)
    assert TreeRecyclingSpecModel._tree_shape("grounded-hybrid-reverse", *common) == (2, 4)
    assert TreeRecyclingSpecModel._tree_shape("visual-width", *common) == (4, 2)
    assert TreeRecyclingSpecModel._tree_shape("visual-width-reverse", *common) == (2, 2)


def test_visual_rootwide_policy_narrows_only_grounded_child_branches():
    branch_width = TreeRecyclingSpecModel._tree_branch_width
    assert branch_width("visual-rootwide-plus2", 0.8, 0.55, 6, 2) == 2
    assert branch_width(
        "visual-rootwide-plus2-hst-backoff", 0.8, 0.55, 6, 2
    ) == 2
    assert branch_width("visual-rootwide-plus2", 0.4, 0.55, 4, 2) == 4
    assert branch_width("visual-wide-plus2", 0.8, 0.55, 6, 2) == 6
    assert branch_width(
        "context-score-prior-deeper-wide-plus6", 0.0, 0.55, 10, 2
    ) == 8
    assert branch_width(
        "context-score-prior-deeper-wide-plus5", 0.0, 0.55, 9, 2
    ) == 8
    assert branch_width(
        "context-score-prior-deeper-wide-plus8", 0.0, 0.55, 12, 2
    ) == 6


def test_tree_builder_preserves_branches_and_blocks_edge_cycles():
    transitions = torch.zeros((1, 8, 2), dtype=torch.long)
    valid = torch.zeros((1, 8), dtype=torch.bool)
    transitions[0, 1] = torch.tensor([2, 3])
    transitions[0, 2] = torch.tensor([1, 4])
    transitions[0, 3] = torch.tensor([4, 5])
    valid[0, [1, 2, 3, 4, 5]] = True

    tokens, mask, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=1,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=3,
        node_budget=15,
        blocked_token_id=None,
    )

    assert tokens[:3] == [1, 2, 3]
    assert mask.shape == (1, 1, len(tokens), len(tokens))
    assert int(positions.max().item()) == 3
    assert any(len(path) == 3 for path in paths)


def test_tree_builder_reuses_cached_topology_tensors():
    transitions = torch.zeros((1, 8, 2), dtype=torch.long)
    valid = torch.zeros((1, 8), dtype=torch.bool)
    transitions[0, 1] = torch.tensor([2, 3])
    transitions[0, 2] = torch.tensor([4, 5])
    transitions[0, 3] = torch.tensor([6, 7])
    valid[0, 1:8] = True
    cache = {}

    first = TreeRecyclingSpecModel._build_tree(
        root_token=1,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=6,
        blocked_token_id=None,
        topology_cache=cache,
    )
    second = TreeRecyclingSpecModel._build_tree(
        root_token=1,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=6,
        blocked_token_id=None,
        topology_cache=cache,
    )

    assert len(cache) == 1
    assert first[1].data_ptr() == second[1].data_ptr()
    assert first[2].data_ptr() == second[2].data_ptr()
    assert first[3] is second[3]


def test_tree_builder_accepts_equivalent_host_transition_rows():
    transitions = torch.zeros((1, 8, 2), dtype=torch.long)
    valid = torch.zeros((1, 8), dtype=torch.bool)
    transitions[0, 1] = torch.tensor([2, 3])
    transitions[0, 2] = torch.tensor([4, 5])
    transitions[0, 3] = torch.tensor([6, 7])
    valid[0, 1:8] = True
    host = {1: [2, 3], 2: [4, 5], 3: [6, 7]}

    device_tree = TreeRecyclingSpecModel._build_tree(
        root_token=1,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=6,
        blocked_token_id=None,
    )
    host_tree = TreeRecyclingSpecModel._build_tree(
        root_token=1,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=6,
        blocked_token_id=None,
        host_transitions=host,
    )

    assert host_tree[0] == device_tree[0]
    assert torch.equal(host_tree[1], device_tree[1])
    assert torch.equal(host_tree[2], device_tree[2])
    assert host_tree[3] == device_tree[3]


def test_rank_prior_tree_reallocates_siblings_to_deeper_paths():
    vocab_size = 400
    transitions = torch.zeros((1, vocab_size, 6), dtype=torch.long)
    valid = torch.ones((1, vocab_size), dtype=torch.bool)
    for token in range(vocab_size):
        transitions[0, token] = torch.tensor(
            [(token * 6 + offset + 1) % vocab_size for offset in range(6)]
        )

    tokens, mask, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=6,
        depth=3,
        node_budget=63,
        blocked_token_id=None,
        priority_layout=True,
    )

    assert len(tokens) - 1 == 63
    assert [(positions == level).sum().item() for level in range(1, 4)] == [
        6,
        26,
        31,
    ]
    assert mask.shape == (1, 1, 64, 64)
    assert max(map(len, paths)) == 3
    assert paths[:3] == [[1], [1, 2], [1, 2, 3]]


def test_score_prior_tree_follows_a_confident_contiguous_chain():
    transitions = torch.zeros((1, 16, 2), dtype=torch.long)
    valid = torch.ones((1, 16), dtype=torch.bool)
    host = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7, 8],
    }
    scores = {token: [0.9, 0.1] for token in host}

    tokens, _, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=3,
        node_budget=4,
        blocked_token_id=None,
        host_transitions=host,
        host_transition_scores=scores,
        score_priority_layout=True,
    )

    assert tokens == [0, 1, 3, 7, 2]
    assert positions.tolist() == [0, 1, 2, 3, 1]
    assert paths[:3] == [[1], [1, 2], [1, 2, 3]]


def test_context_score_tree_prefers_exact_bigram_rows_and_records_context():
    transitions = torch.empty(0, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)
    host = {0: [2, 1], 1: [4, 3]}
    scores = {0: [0.9, 0.1], 1: [0.9, 0.1]}
    context = {(9, 0): [1, 2], (0, 1): [3, 4]}
    context_scores = {(9, 0): [0.9, 0.1], (0, 1): [0.9, 0.1]}
    metadata = {}

    tokens, _, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=3,
        blocked_token_id=None,
        host_transitions=host,
        host_transition_scores=scores,
        root_previous_token=9,
        host_context_transitions=context,
        host_context_transition_scores=context_scores,
        metadata_out=metadata,
        score_priority_layout=True,
    )

    assert tokens == [0, 1, 3, 2]
    assert positions.tolist() == [0, 1, 2, 1]
    assert paths[:2] == [[1], [1, 2]]
    assert metadata["semantic_previous_tokens"] == [9, 0, 1, 0]


def test_trigram_score_tree_precedes_bigram_and_records_two_token_context():
    transitions = torch.empty(0, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)
    host = {0: [1, 2], 2: [3, 4]}
    scores = {0: [0.9, 0.1], 2: [0.9, 0.1]}
    context = {(9, 0): [1, 2], (0, 2): [3, 4]}
    context_scores = {(9, 0): [0.9, 0.1], (0, 2): [0.9, 0.1]}
    trigrams = {(8, 9, 0): [2, 1], (9, 0, 2): [4, 3]}
    trigram_scores = {(8, 9, 0): [0.9, 0.1], (9, 0, 2): [0.9, 0.1]}
    metadata = {}

    tokens, _, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=3,
        blocked_token_id=None,
        host_transitions=host,
        host_transition_scores=scores,
        root_previous_token=9,
        root_previous_previous_token=8,
        host_context_transitions=context,
        host_context_transition_scores=context_scores,
        host_trigram_transitions=trigrams,
        host_trigram_transition_scores=trigram_scores,
        metadata_out=metadata,
        score_priority_layout=True,
    )

    assert tokens == [0, 2, 4, 1]
    assert positions.tolist() == [0, 1, 2, 1]
    assert paths[:2] == [[1], [1, 2]]
    assert metadata["semantic_previous_tokens"] == [9, 0, 2, 0]
    assert metadata["semantic_previous_previous_tokens"] == [8, 9, 0, 9]


def test_fourgram_score_tree_precedes_trigram_and_records_three_token_context():
    transitions = torch.empty(0, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)
    host = {0: [1, 2], 2: [3, 4]}
    scores = {0: [0.9, 0.1], 2: [0.9, 0.1]}
    context = {(9, 0): [1, 2], (0, 2): [3, 4]}
    context_scores = {(9, 0): [0.9, 0.1], (0, 2): [0.9, 0.1]}
    trigrams = {(8, 9, 0): [2, 1], (9, 0, 2): [4, 3]}
    trigram_scores = {(8, 9, 0): [0.9, 0.1], (9, 0, 2): [0.9, 0.1]}
    fourgrams = {(7, 8, 9, 0): [1, 2], (8, 9, 0, 1): [3, 4]}
    fourgram_scores = {
        (7, 8, 9, 0): [0.9, 0.1],
        (8, 9, 0, 1): [0.9, 0.1],
    }
    metadata = {}

    tokens, _, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=2,
        node_budget=3,
        blocked_token_id=None,
        host_transitions=host,
        host_transition_scores=scores,
        root_previous_token=9,
        root_previous_previous_token=8,
        root_previous_previous_previous_token=7,
        host_context_transitions=context,
        host_context_transition_scores=context_scores,
        host_trigram_transitions=trigrams,
        host_trigram_transition_scores=trigram_scores,
        host_fourgram_transitions=fourgrams,
        host_fourgram_transition_scores=fourgram_scores,
        metadata_out=metadata,
        score_priority_layout=True,
    )

    assert tokens == [0, 1, 3, 2]
    assert positions.tolist() == [0, 1, 2, 1]
    assert paths[:2] == [[1], [1, 2]]
    assert metadata["semantic_previous_tokens"] == [9, 0, 1, 0]
    assert metadata["semantic_previous_previous_tokens"] == [8, 9, 0, 9]
    assert metadata["semantic_previous_previous_previous_tokens"] == [
        7,
        8,
        9,
        8,
    ]


@pytest.mark.parametrize(
    ("mode", "expected_tokens"),
    [
        ("strict", [0, 1, 2, 3, 4]),
        ("residual2", [0, 1, 2, 5, 6]),
        ("fusion", [0, 1, 2, 3, 5]),
        ("fusion55", [0, 1, 2, 5, 3]),
        ("fusion_adaptive", [0, 1, 2, 5, 3]),
    ],
)
def test_context_candidate_modes_mix_lower_order_fallbacks(
    mode, expected_tokens
):
    transitions = torch.empty(0, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)
    host = {0: [7, 8, 1, 2]}
    scores = {0: [0.6, 0.2, 0.1, 0.1]}
    contexts = {(9, 0): [5, 6, 1, 2]}
    context_scores = {(9, 0): [0.6, 0.2, 0.1, 0.1]}
    trigrams = {(8, 9, 0): [1, 2, 3, 4]}
    trigram_scores = {(8, 9, 0): [0.4, 0.3, 0.2, 0.1]}

    tokens, _, _, _ = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=4,
        depth=1,
        node_budget=4,
        blocked_token_id=None,
        host_transitions=host,
        host_transition_scores=scores,
        root_previous_token=9,
        root_previous_previous_token=8,
        host_context_transitions=contexts,
        host_context_transition_scores=context_scores,
        host_trigram_transitions=trigrams,
        host_trigram_transition_scores=trigram_scores,
        context_candidate_mode=mode,
        score_priority_layout=True,
    )

    assert tokens == expected_tokens


def test_global_candidates_only_fill_an_unseen_host_row():
    transitions = torch.empty(0, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)

    tokens, _, positions, paths = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=4,
        depth=1,
        node_budget=3,
        blocked_token_id=None,
        host_transitions={},
        host_transition_scores={},
        host_global_candidates=[5, 6, 7, 8],
        host_global_candidate_scores=[0.4, 0.3, 0.2, 0.1],
        score_priority_layout=True,
    )

    assert tokens == [0, 5, 6, 7]
    assert positions.tolist() == [0, 1, 1, 1]
    assert paths == [[1], [2], [3]]

    tokens, _, _, _ = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=1,
        node_budget=2,
        blocked_token_id=None,
        host_transitions={0: [1, 2]},
        host_transition_scores={0: [0.75, 0.25]},
        host_global_candidates=[5, 6],
        host_global_candidate_scores=[0.6, 0.4],
        score_priority_layout=True,
    )
    assert tokens == [0, 1, 2]


def test_persistent_candidates_are_a_lower_order_fallback():
    transitions = torch.empty(0, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)

    tokens, _, _, _ = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=1,
        node_budget=2,
        blocked_token_id=None,
        host_transitions={},
        host_transition_scores={},
        host_persistent_transitions={0: [5, 6]},
        host_persistent_transition_scores={0: [0.7, 0.3]},
        context_candidate_mode="fusion",
        score_priority_layout=True,
    )
    assert tokens == [0, 5, 6]

    tokens, _, _, _ = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=2,
        depth=1,
        node_budget=2,
        blocked_token_id=None,
        host_transitions={0: [1, 2]},
        host_transition_scores={0: [0.7, 0.3]},
        host_persistent_transitions={0: [5, 6]},
        host_persistent_transition_scores={0: [0.7, 0.3]},
        context_candidate_mode="strict",
        score_priority_layout=True,
    )
    assert tokens == [0, 1, 2]

    tokens, _, _, _ = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=4,
        depth=1,
        node_budget=4,
        blocked_token_id=None,
        host_transitions={0: [1, 2]},
        host_transition_scores={0: [0.7, 0.3]},
        host_persistent_transitions={0: [5, 6]},
        host_persistent_transition_scores={0: [0.7, 0.3]},
        context_candidate_mode="fusion",
        score_priority_layout=True,
    )
    assert tokens == [0, 1, 2, 5, 6]


def test_best_verified_path_uses_longest_matching_branch():
    # Root predicts token 11.  Its packed node then predicts token 13, so the
    # first two-node branch is accepted and node 13 supplies the correction.
    logits = torch.full((1, 4, 20), -10.0)
    logits[0, 0, 11] = 5.0
    logits[0, 1, 13] = 5.0
    logits[0, 2, 14] = 5.0
    logits[0, 3, 15] = 5.0
    accepted_path, accepted_tokens, correction, query_index = (
        TreeRecyclingSpecModel._best_verified_path(
            logits,
            flat_tokens=[7, 11, 12, 13],
            paths=[[1], [2], [1, 3]],
            remaining_tokens=8,
        )
    )

    assert accepted_path == [1, 3]
    assert accepted_tokens == [11, 13]
    assert correction == 15
    assert query_index == 3
    queued = torch.argmax(logits[0], dim=-1)
    assert TreeRecyclingSpecModel._best_verified_path(
        logits,
        flat_tokens=[7, 11, 12, 13],
        paths=[[1], [2], [1, 3]],
        remaining_tokens=8,
        predictions=queued,
    ) == (accepted_path, accepted_tokens, correction, query_index)

def test_cache_compaction_is_skipped_for_already_contiguous_paths():
    requires = TreeRecyclingSpecModel._requires_cache_compaction
    assert not requires([])
    assert not requires([1])
    assert not requires([1, 2, 3])
    assert requires([2])
    assert requires([1, 4])


def test_target_only_path_stops_without_building_recycling_state():
    def logits_for(token_id):
        logits = torch.full((1, 1, 10), -10.0)
        logits[0, 0, token_id] = 10.0
        return logits

    class FakeTarget:
        def __init__(self):
            self.tokens = iter([3, 4])

        def __call__(self, **kwargs):
            assert kwargs["output_hidden_states"] is False
            return SimpleNamespace(logits=logits_for(next(self.tokens)))

    fake_model = SimpleNamespace(base_model=FakeTarget())
    output, new_tokens, idx, acceptance, trace = (
        TreeRecyclingSpecModel._target_only_from_prefill(
            fake_model,
            input_ids=torch.tensor([[8, 9]]),
            prefill_output=SimpleNamespace(logits=logits_for(2)),
            past_key_values=object(),
            current_length_data=torch.zeros(1, dtype=torch.long),
            prompt_length=2,
            max_new_tokens=5,
            max_length=7,
            stop_token_ids={4},
            log=True,
            return_acceptance_len=True,
            return_decode_time=False,
            return_policy_trace=True,
        )
    )

    assert output.tolist() == [[8, 9, 2, 3, 4]]
    assert new_tokens == 3
    assert idx == 2
    assert acceptance == [0, 0]
    assert len(trace) == 2


def test_spine_tree_is_wide_only_at_the_root():
    transitions = torch.zeros((1, 16, 4), dtype=torch.long)
    valid = torch.ones((1, 16), dtype=torch.bool)
    for token in range(16):
        transitions[0, token] = torch.tensor(
            [(token + offset + 1) % 16 for offset in range(4)]
        )

    tokens, _, positions, _ = TreeRecyclingSpecModel._build_tree(
        root_token=0,
        transitions=transitions,
        transition_valid=valid,
        width=4,
        depth=4,
        node_budget=31,
        blocked_token_id=None,
        branch_width=1,
    )

    assert len(tokens) - 1 == 16
    assert [(positions == depth).sum().item() for depth in range(1, 5)] == [4, 4, 4, 4]


def test_promote_transition_prefers_suffix_edge_and_keeps_alternatives():
    transitions = torch.tensor([[[0, 0, 0], [2, 3, 4], [0, 0, 0]]])
    valid = torch.tensor([[False, True, False]])
    TreeRecyclingSpecModel._promote_transition(
        transitions,
        valid,
        transition_bin=0,
        source_token=1,
        next_token=4,
    )
    assert transitions[0, 1].tolist() == [4, 2, 3]


def test_grounded_backoff_fuses_conditional_residual_with_global_tail():
    transitions = torch.zeros((3, 10, 4), dtype=torch.long)
    valid = torch.zeros((3, 10), dtype=torch.bool)
    counts = torch.zeros((3, 10), dtype=torch.int32)
    transitions[0, 1] = torch.tensor([2, 3, 4, 5])
    transitions[2, 1] = torch.tensor([2, 6, 7, 8])
    valid[0, 1] = True
    valid[2, 1] = True
    counts[2, 1] = 2

    candidates = TreeRecyclingSpecModel._transition_candidates(
        transitions,
        valid,
        parent_token=1,
        primary_bin=2,
        limit=4,
        fallback_bin=0,
        transition_counts=counts,
    )

    assert candidates == [2, 3, 6, 4]


def test_grounded_backoff_uses_global_row_when_condition_is_unseen():
    transitions = torch.zeros((3, 8, 3), dtype=torch.long)
    valid = torch.zeros((3, 8), dtype=torch.bool)
    transitions[0, 1] = torch.tensor([2, 3, 4])
    valid[0, 1] = True

    candidates = TreeRecyclingSpecModel._transition_candidates(
        transitions,
        valid,
        parent_token=1,
        primary_bin=2,
        limit=2,
        fallback_bin=0,
    )

    assert candidates == [2, 3]


def test_grounded_residual_preserves_global_topk_before_adding_candidates():
    transitions = torch.zeros((3, 12, 6), dtype=torch.long)
    valid = torch.zeros((3, 12), dtype=torch.bool)
    counts = torch.zeros((3, 12), dtype=torch.int32)
    transitions[0, 1] = torch.tensor([2, 3, 4, 5, 9, 10])
    transitions[2, 1] = torch.tensor([2, 6, 7, 8, 9, 10])
    valid[0, 1] = True
    valid[2, 1] = True
    counts[2, 1] = 2

    candidates = TreeRecyclingSpecModel._transition_candidates(
        transitions,
        valid,
        parent_token=1,
        primary_bin=2,
        limit=6,
        fallback_bin=0,
        transition_counts=counts,
        preserve_fallback_limit=4,
    )

    assert candidates == [2, 3, 4, 5, 6, 7]


def test_grounded_residual_requires_repeated_conditional_observation():
    transitions = torch.zeros((3, 12, 6), dtype=torch.long)
    valid = torch.zeros((3, 12), dtype=torch.bool)
    counts = torch.zeros((3, 12), dtype=torch.int32)
    transitions[0, 1] = torch.tensor([2, 3, 4, 5, 9, 10])
    transitions[2, 1] = torch.tensor([6, 7, 8, 9, 10, 11])
    valid[0, 1] = True
    valid[2, 1] = True
    counts[2, 1] = 1

    candidates = TreeRecyclingSpecModel._transition_candidates(
        transitions,
        valid,
        parent_token=1,
        primary_bin=2,
        limit=6,
        fallback_bin=0,
        transition_counts=counts,
        preserve_fallback_limit=4,
    )

    assert candidates == [2, 3, 4, 5]


def test_grounded_residual_never_exceeds_requested_width():
    transitions = torch.zeros((3, 12, 6), dtype=torch.long)
    valid = torch.zeros((3, 12), dtype=torch.bool)
    counts = torch.zeros((3, 12), dtype=torch.int32)
    transitions[0, 1] = torch.tensor([2, 3, 4, 5, 9, 10])
    transitions[2, 1] = torch.tensor([6, 7, 8, 9, 10, 11])
    valid[0, 1] = True
    valid[2, 1] = True
    counts[2, 1] = 3

    candidates = TreeRecyclingSpecModel._transition_candidates(
        transitions,
        valid,
        parent_token=1,
        primary_bin=2,
        limit=4,
        fallback_bin=0,
        transition_counts=counts,
        preserve_fallback_limit=4,
    )

    assert candidates == [2, 3, 4, 5]


def test_balanced_topic_selection_supports_disjoint_holdout_offsets():
    data = [
        {"topic": "a", "id": "a0"},
        {"topic": "b", "id": "b0"},
        {"topic": "a", "id": "a1"},
        {"topic": "a", "id": "a2"},
        {"topic": "b", "id": "b1"},
        {"topic": "b", "id": "b2"},
    ]

    assert _select_topic_indices(data, samples_per_topic=1, topic_offset=0) == [0, 1]
    assert _select_topic_indices(data, samples_per_topic=2, topic_offset=1) == [2, 3, 4, 5]
