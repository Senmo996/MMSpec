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
