"""Regressions for committed-state suffix lookup and single-pass allocation."""

import copy
import random

import pytest
import torch

from method.sam.samd.sam_draft import DraftModel
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model


def tree_kwargs():
    return dict(
        root_token=1, transitions=torch.empty(0, dtype=torch.long),
        transition_valid=torch.empty(0, dtype=torch.bool), width=2, depth=4,
        blocked_token_id=-1, host_transitions={1: [2, 8], 2: [3, 9], 3: [4, 10]},
        host_transition_scores={1: [.6, .4], 2: [.6, .4], 3: [.6, .4]},
        score_priority_layout=True,
    )


def assert_same_tree(left, right):
    assert left[0] == right[0]
    assert torch.equal(left[1], right[1])
    assert torch.equal(left[2], right[2])
    assert left[3] == right[3]


def past_suffix_oracle(tokens, max_tokens=4):
    """Earliest previous occurrence of the longest already-committed suffix."""
    for length in range(len(tokens) - 1, 0, -1):
        for start in range(len(tokens) - length):
            if tokens[start:start + length] == tokens[-length:]:
                end = start + length
                return length, tokens[end:end + min(max_tokens, 39, 4 * length)]
    return 0, []


def test_committed_root_is_not_consumed_twice():
    draft = DraftModel(device="cpu")
    draft.update_tokens([11, 22, 33, 44, 55, 22, 66, 44, 55, 22])
    assert draft.sam_dyn.cur_length == 3
    assert draft.sam_dyn.lookup(22)[1] == 1
    assert draft.lookup_tokens(22)[0][:4] == [33, 44, 55, 22]
    assert draft.lookup_continuation_tokens() == [66, 44, 55, 22]


def test_continuation_is_read_only_and_creates_no_position_tensor(monkeypatch):
    draft = DraftModel(device="cpu")
    draft.update_tokens([1, 2, 3, 1, 2])
    before = copy.deepcopy(vars(draft.sam_dyn))

    def forbidden(*args, **kwargs):
        raise AssertionError("continuation lookup must not transfer or allocate a tensor")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "arange", forbidden)
        patch.setattr(type(draft.sam_dyn), "lookup", forbidden)
        patch.setattr(type(draft.sam_dyn), "gen_dyn_draft", forbidden)
        assert draft.lookup_continuation_tokens(2) == [3, 1]
        assert draft.lookup_continuation_tokens(2) == [3, 1]
    assert vars(draft.sam_dyn) == before


def test_empty_unseen_reset_and_length_limits():
    draft = DraftModel(device="cpu")
    assert draft.lookup_continuation_tokens() == []
    draft.update_tokens([10, 20, 30])
    assert draft.lookup_continuation_tokens() == []
    draft.update_tokens([10])
    assert draft.lookup_continuation_tokens(0) == []
    assert draft.lookup_continuation_tokens(1) == [20]
    assert draft.lookup_continuation_tokens(100) == [20, 30, 10]
    draft.sam_dyn.n_predicts = 2
    assert draft.lookup_continuation_tokens(100) == [20]
    with pytest.raises(ValueError):
        draft.lookup_continuation_tokens(-1)
    draft.reset()
    assert draft.lookup_continuation_tokens() == []


def test_incremental_suffix_state_matches_past_only_oracle():
    rng = random.Random(731)
    for history in (
        [1] * 40, [1, 2] * 30, [1, 2, 3, 4] * 20,
        *[[rng.randrange(7) for _ in range(100)] for _ in range(20)],
    ):
        draft = DraftModel(device="cpu")
        for end, token in enumerate(history, start=1):
            draft.update_tokens([token])
            length, expected = past_suffix_oracle(history[:end])
            assert draft.sam_dyn.cur_length == length
            assert draft.lookup_continuation_tokens() == expected
        batched = DraftModel(device="cpu")
        for start in range(0, len(history), 7):
            batched.update_tokens(history[start:start + 7])
        assert batched.lookup_continuation_tokens() == draft.lookup_continuation_tokens()


def test_overlap_is_refilled_with_one_tree_build(monkeypatch):
    original = Model._build_tree
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(Model, "_build_tree", staticmethod(counted))
    args = tree_kwargs() | dict(node_budget=5, reserved_path_tokens=[2, 3])
    legacy = Model._build_tree_budgeted(**args)
    assert len(calls) == 2
    calls.clear()
    fixed = Model._build_tree_budgeted_single_pass(**args)
    assert len(calls) == 1
    assert len(fixed[0]) - 1 == 5
    assert_same_tree(legacy, fixed)


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 5, 10])
@pytest.mark.parametrize("suffix", [[], [2, 3], [9, 8, 7, 6], [-1, 2], [2, -1], [2, 1, 2, 3]])
def test_single_pass_budget_edges(budget, suffix):
    args = tree_kwargs() | dict(node_budget=budget, reserved_path_tokens=suffix)
    original = copy.deepcopy(args)
    legacy = Model._build_tree_budgeted(**args)
    fixed = Model._build_tree_budgeted_single_pass(**args)
    assert_same_tree(legacy, fixed)
    assert len(fixed[0]) - 1 <= budget
    assert args["reserved_path_tokens"] == original["reserved_path_tokens"]
    assert args["host_transitions"] == original["host_transitions"]


def test_single_pass_preserves_ranked_selection_and_metadata_randomized():
    rng = random.Random(20260910)
    for _ in range(750):
        vocab = list(range(9))
        rows = {token: rng.sample(vocab, rng.randrange(6)) for token in vocab if rng.random() < .8}
        scores = {}
        for token, row in rows.items():
            weights = [rng.uniform(.05, 1) for _ in row]
            scores[token] = [v / sum(weights) for v in weights]
        context_rows = {(a, b): rng.sample(vocab, 3) for a, b in ((0, 1), (1, 2), (2, 3))}
        args = tree_kwargs() | dict(
            root_token=rng.choice(vocab), root_previous_token=rng.choice(vocab),
            root_previous_previous_token=rng.choice(vocab),
            width=rng.randrange(1, 6), branch_width=rng.randrange(1, 5),
            depth=rng.randrange(1, 7), node_budget=rng.randrange(25),
            blocked_token_id=rng.choice([-1, 0]), host_transitions=rows,
            host_transition_scores=scores, host_context_transitions=context_rows,
            host_context_transition_scores={key: [.6, .3, .1] for key in context_rows},
            context_candidate_mode=rng.choice(["strict", "fusion", "fusion_uniform"]),
            reserved_path_tokens=[rng.choice(vocab + [-1]) for _ in range(rng.randrange(6))],
        )
        legacy_metadata = {"_candidate_trace_diagnostics": True}
        fixed_metadata = {"_candidate_trace_diagnostics": True}
        legacy = Model._build_tree_budgeted(**args, metadata_out=legacy_metadata)
        fixed = Model._build_tree_budgeted_single_pass(**args, metadata_out=fixed_metadata)
        assert_same_tree(legacy, fixed)
        assert legacy_metadata == fixed_metadata
        assert len(fixed[0]) - 1 <= args["node_budget"]
        # Every packed row attends exactly to the root and its own ancestry.
        for row_index, path in enumerate(fixed[3], start=1):
            assert fixed[2][row_index].item() == len(path)
            assert set(torch.where(fixed[1][0, 0, row_index])[0].tolist()) == {0, *path}


def test_single_pass_sparse_pool_cache_and_backend_fallback():
    for layout in ("score", "breadth", "rank"):
        args = tree_kwargs() | dict(
            width=2, depth=3, node_budget=63,
            reserved_path_tokens=[8, 9, 10],
            score_priority_layout=layout == "score", priority_layout=layout == "rank",
            host_transitions={}, host_transition_scores={},
        )
        cache = {}
        legacy = Model._build_tree_budgeted(**args)
        fixed = Model._build_tree_budgeted_single_pass(**args, topology_cache=cache)
        cached = Model._build_tree_budgeted_single_pass(**args, topology_cache=cache)
        assert_same_tree(legacy, fixed)
        assert_same_tree(fixed, cached)
        assert fixed[1] is cached[1]
        assert len(fixed[0]) - 1 == 3


def test_suffix_comparison_changes_only_the_two_approved_components():
    from evaluation.eval_recap_suffix_fixes import (
        CONTROL_POLICIES, EXPERIMENT, control_kwargs,
    )

    policies = CONTROL_POLICIES[EXPERIMENT]
    assert list(policies) == ["target", "no_suffix", "legacy_suffix", "fixed_suffix"]
    controls = {name: control_kwargs(EXPERIMENT, name) for name in policies}
    baseline = controls["no_suffix"]
    for name in ("legacy_suffix", "fixed_suffix"):
        for key, value in baseline.items():
            if key not in ("recycling_suffix_length", "recycling_suffix_implementation"):
                assert controls[name][key] == value
        assert controls[name]["recycling_suffix_length"] == 4
        assert controls[name]["recycling_total_node_budget"] == 63
    assert controls["legacy_suffix"]["recycling_suffix_implementation"] == "legacy"
    assert controls["fixed_suffix"]["recycling_suffix_implementation"] == "fixed"
    assert control_kwargs("suffix", "fusion_suffix")["recycling_suffix_implementation"] == "legacy"
