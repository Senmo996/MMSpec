import copy
import random

import pytest
import torch

from method.sam_grounded.root_semantic_backoff import RootBackoffMemory, predecessor_keys
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model
from evaluation.eval_recap_mmspec_semantic import next_turn, options, summarize


def test_predecessor_alignment_prefill_and_branched_tree():
    hidden = torch.arange(15).reshape(5, 3).float()
    prompt = predecessor_keys(hidden, [-1, 0, 3])
    assert prompt.tolist() == [[0., 0., 0.], hidden[0].tolist(), hidden[3].tolist()]
    previous = torch.tensor([100., 101., 102.])
    tree = predecessor_keys(hidden, [-1, 0, 1, 0, 3], previous)
    assert torch.equal(tree[0], previous)
    assert torch.equal(tree[3], hidden[0])  # Not flattened row 2.
    assert torch.equal(tree[4], hidden[3])
    hidden[-1].fill_(-999)
    assert torch.equal(tree[4], torch.tensor([9., 10., 11.]))


def test_cosine_and_recent_use_the_same_pool_but_different_rankings():
    banks = [RootBackoffMemory(mode, capacity=4, neighbors=1) for mode in ("cosine", "recent")]
    for bank in banks:
        bank.update([[10], [20]], [[1.], [1.]], [[("U", 1)], [("U", 2)]], torch.eye(2))
    assert banks[0].retrieve(torch.tensor([1., 0.])) == ([10], [1.])
    assert banks[1].retrieve() == ([20], [1.])
    assert set(banks[0].owners) == set(banks[1].owners)


@pytest.mark.parametrize("mode", ["cosine", "recent"])
def test_only_live_lexical_rows_remain_eligible(mode):
    bank = RootBackoffMemory(mode, 4, 3)
    bank.update([[10]], [[1.]], [[("U", 1), ("C", (2, 1))]], torch.tensor([[1., 0.]]))
    bank.update([[20]], [[1.]], [[("U", 1)]], torch.tensor([[0., 1.]]))
    assert len(bank.entries) == 2  # Old row survives through C.
    bank.update([[30]], [[1.]], [[("C", (2, 1))]], torch.tensor([[-1., 0.]]))
    assert len(bank.entries) == 2
    assert 10 not in bank.retrieve(torch.tensor([1., 0.]))[0]
    assert bank.owners[("U", 1)] != bank.owners[("C", (2, 1))]


def test_bounded_eviction_and_repeated_scatter_slots():
    bank = RootBackoffMemory("cosine", capacity=1, neighbors=1)
    bank.update([[10], [20], [30]], [[1.]] * 3,
                [[("U", 1)], [("U", 2)], [("U", 2)]], torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
    assert bank.retrieve(torch.tensor([-1., 0.])) == ([30], [1.])
    assert bank.stats()["backoff_index_evicted"] == 1
    assert list(bank.owners) == [("U", 2)]
    assert bank.keys[0].tolist() == [-1., 0.]


def test_fusion_uses_fixed_weights_and_deduplicated_top_k():
    bank = RootBackoffMemory("recent", 4, 3, top_k=2)
    bank.update([[10, 11], [10, 12], [13, 10]], [[.6, .4]] * 3,
                [[("U", n)] for n in range(3)])
    candidates, scores = bank.retrieve()
    assert candidates == [10, 13]
    assert scores == pytest.approx([.46 / .88, .42 / .88])


def tree_args():
    return dict(root_token=1, transitions=torch.empty(0, dtype=torch.long),
                transition_valid=torch.empty(0, dtype=torch.bool), width=3, branch_width=3,
                depth=8, node_budget=20, blocked_token_id=-1,
                host_transitions={2: [1]}, host_transition_scores={2: [1.]},
                score_priority_layout=True, context_candidate_mode="fusion")


@pytest.mark.parametrize("cpp", [False, True])
def test_backoff_is_root_only_does_not_pollute_lexical_cache(cpp):
    args = tree_args() | dict(fast_score_priority_layout=cpp)
    before = copy.deepcopy(args["host_transitions"])
    result = Model._build_tree(**args, root_backoff_distribution=([2, 3], [.8, .2]))
    assert Model._tree_path_token_ids(result[0], result[3]) == [[2], [2, 1], [3]]
    assert args["host_transitions"] == before


@pytest.mark.parametrize("cpp", [False, True])
def test_backoff_never_overrides_existing_lexical_candidates(cpp):
    args = tree_args() | dict(fast_score_priority_layout=cpp, host_transitions={1: [7, 8], 7: [9]})
    original = Model._build_tree(**args)
    extended = Model._build_tree(**args, root_backoff_distribution=([2, 3], [.8, .2]))
    assert original[0] == extended[0]
    assert torch.equal(original[1], extended[1])
    assert original[3] == extended[3]


def test_cpp_python_parity_and_budget_with_root_fallback_and_suffix():
    rng = random.Random(419)
    for _ in range(80):
        rows = {token: rng.sample(range(10), rng.randrange(1, 4)) for token in range(10) if rng.random() < .6}
        args = tree_args() | dict(root_token=rng.randrange(10), host_transitions=rows,
                host_transition_scores={t: [1 / len(row)] * len(row) for t, row in rows.items()},
                root_backoff_distribution=([3, 5, 7], [.7, .2, .1]),
                width=rng.randrange(1, 4), node_budget=rng.randrange(1, 30),
                reserved_path_tokens=rng.sample(range(10), 4))
        python = Model._build_tree_budgeted(**args)
        cpp = Model._build_tree_budgeted(**args, fast_score_priority_layout=True)
        assert python[0] == cpp[0]
        assert torch.equal(python[1], cpp[1])
        assert torch.equal(python[2], cpp[2])
        assert python[3] == cpp[3]
        assert len(cpp[0]) - 1 <= args["node_budget"]


def test_namespace_and_other_components_are_matched():
    baseline = options("original")
    for variant in ("semantic_root", "recent_root"):
        candidate = options(variant)
        assert candidate["recycling_cache_namespace"] != baseline["recycling_cache_namespace"]
        for key in baseline.keys() - {"recycling_cache_namespace", "recycling_root_backoff"}:
            assert baseline[key] == candidate[key]


def test_multiturn_history_uses_ids_without_decoding():
    class Tokenizer:
        all_special_ids = [90]
        def encode(self, value, add_special_tokens=False):
            assert "next question" in value
            return [100, 101]
        def decode(self, *_args, **_kwargs):
            raise AssertionError("formal timing never decodes generated IDs")
    inputs = dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2), pixel_values="original image")
    result = next_turn(inputs, [5, 6, 90], "next question", Tokenizer())
    assert result["input_ids"].tolist() == [[1, 2, 5, 6, 100, 101]]
    assert result["pixel_values"] == inputs["pixel_values"]
    assert inputs["input_ids"].tolist() == [[1, 2]]


def test_summary_reports_raw_acceptance_and_plus_one_separately():
    rows = [dict(sample_id="a", num_tokens=7, seconds=2., accepted_lengths=[1, 3],
                 average_accept_length=2., average_tokens_per_iteration=7/3,
                 verification_iterations=2, speedup=1.5, relative_original_speed=1.1,
                 matches_target=True, target_seconds=3., target_num_tokens=7,
                 counters=dict(root_lexical_misses=1, root_backoff_triggers=1))]
    result = summarize(rows, "results.jsonl")
    assert result["avg_accept_length"] == 2. and result["tau"] == 3.
    assert result["pooled_speedup"] == 1.5
    assert result["root_backoff_trigger_rate"] == .5
