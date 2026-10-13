"""Spare-budget suffix bridging and semantically equivalent SAM cloning."""

import copy
import json
import random

import pytest
import torch

from method.sam.samd.dyn_sam import DynSAM
from method.sam.samd.sam_draft import DraftModel
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model


def bridge_args():
    return dict(
        root_token=1, transitions=torch.empty(0, dtype=torch.long),
        transition_valid=torch.empty(0, dtype=torch.bool), width=2, depth=6,
        node_budget=8, blocked_token_id=-1,
        host_transitions={3: [4, 5], 4: [6]},
        host_transition_scores={3: [.8, .2], 4: [1.]},
        score_priority_layout=True, reserved_path_tokens=[2, 3],
    )


def assert_base_is_unchanged(base, extended):
    size = len(base[0])
    assert extended[0][:size] == base[0]
    assert torch.equal(extended[1][..., :size, :size], base[1])
    assert not extended[1][..., :size, size:].any()
    assert torch.equal(extended[2][:size], base[2])
    assert extended[3][:size - 1] == base[3]


def semantic_state(sam):
    return {key: value for key, value in vars(sam).items() if key != "clone_mode"}


def test_bridge_reaches_cached_predictions_without_another_tree_build(monkeypatch):
    args = bridge_args()
    baseline = Model._build_tree_budgeted_single_pass(**args)
    assert Model._tree_path_token_ids(baseline[0], baseline[3]) == [[2], [2, 3]]
    original = Model._build_tree
    calls = []

    def counted(*values, **kwargs):
        calls.append(1)
        return original(*values, **kwargs)

    monkeypatch.setattr(Model, "_build_tree", staticmethod(counted))
    metadata = {"_suffix_bridge_diagnostics": True}
    extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True, metadata_out=metadata)
    assert len(calls) == 1
    assert_base_is_unchanged(baseline, extended)
    paths = Model._tree_path_token_ids(extended[0], extended[3])
    assert [2, 3, 4, 6] in paths and [2, 3, 5] in paths
    assert len(paths) == 5
    assert metadata["suffix_added_nodes"] == 2
    assert metadata["pre_bridge_candidate_nodes"] == 2
    assert metadata["suffix_bridge_opportunity"]
    assert metadata["suffix_bridge_added_nodes"] == 3
    assert metadata["suffix_bridge_node_start"] == 3


def test_bridge_uses_the_suffix_context_not_the_original_root_context():
    args = bridge_args() | dict(
        root_previous_token=90, root_previous_previous_token=80,
        host_trigram_transitions={(1, 2, 3): [7], (2, 3, 7): [8], (90, 1, 3): [99]},
        host_trigram_transition_scores={(1, 2, 3): [1.], (2, 3, 7): [1.], (90, 1, 3): [1.]},
        context_candidate_mode="strict", node_budget=4,
    )
    metadata = {"_suffix_bridge_diagnostics": True}
    extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True, metadata_out=metadata)
    assert Model._tree_path_token_ids(extended[0], extended[3])[-1] == [2, 3, 7, 8]
    assert 99 not in extended[0]
    assert metadata["semantic_previous_tokens"] == [90, 1, 2, 3, 7]
    assert metadata["semantic_previous_previous_tokens"] == [80, 90, 1, 2, 3]


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 8, 63])
@pytest.mark.parametrize("suffix", [[], [2], [2, 3], [-1], [2, -1], [2, 1, 2]])
def test_bridge_respects_caps_and_valid_suffix_prefixes(budget, suffix):
    args = bridge_args() | dict(node_budget=budget, reserved_path_tokens=suffix)
    baseline = Model._build_tree_budgeted_single_pass(**args)
    extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True)
    assert_base_is_unchanged(baseline, extended)
    assert len(extended[0]) - 1 <= budget
    for path in Model._tree_path_token_ids(extended[0], extended[3]):
        edges = list(zip([args["root_token"], *path[:-1]], path))
        assert len(set(edges)) == len(edges)
        assert -1 not in path
        assert len(path) <= args["depth"]


def test_bridge_never_displaces_a_full_tree():
    args = bridge_args() | dict(node_budget=3, host_transitions={1: [8, 9], 3: [4, 5]})
    baseline = Model._build_tree_budgeted_single_pass(**args)
    metadata = {"_suffix_bridge_diagnostics": True}
    extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True, metadata_out=metadata)
    assert_base_is_unchanged(baseline, extended)
    assert len(extended[0]) == len(baseline[0]) == 4
    assert metadata["suffix_bridge_added_nodes"] == 0
    assert not metadata["suffix_bridge_opportunity"]


def test_bridge_depth_and_repeated_edges_and_duplicate_candidates():
    args = bridge_args() | dict(
        host_transitions={3: [1, 1], 1: []}, host_transition_scores={3: [.5, .5]},
        depth=3,
    )
    # The first root row must be empty. The bridge can revisit token 1 but
    # cannot repeat an edge already traversed by its ancestry.
    extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True)
    paths = Model._tree_path_token_ids(extended[0], extended[3])
    assert paths == [[2], [2, 3], [2, 3, 1]]
    args.update(host_context_transitions={(3, 1): [2]}, depth=6)
    extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True)
    assert Model._tree_path_token_ids(extended[0], extended[3]) == paths


def test_bridge_randomized_stable_prefix_budget_and_ancestry():
    rng = random.Random(914)
    for _ in range(500):
        rows = {i: rng.sample(range(10), rng.randrange(4)) for i in range(10) if rng.random() < .5}
        args = bridge_args() | dict(
            root_token=rng.randrange(10), host_transitions=rows,
            host_transition_scores={i: [1 / len(row) for _ in row] for i, row in rows.items()},
            width=rng.randrange(1, 4), branch_width=rng.randrange(1, 4),
            depth=rng.randrange(1, 8), node_budget=rng.randrange(20),
            reserved_path_tokens=[rng.randrange(10) for _ in range(rng.randrange(5))],
        )
        before_rows = copy.deepcopy(rows)
        baseline = Model._build_tree_budgeted_single_pass(**args)
        extended = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True)
        assert_base_is_unchanged(baseline, extended)
        assert rows == before_rows
        assert len(extended[0]) - 1 <= args["node_budget"]
        token_paths = Model._tree_path_token_ids(extended[0], extended[3])
        assert len(set(map(tuple, token_paths))) == len(token_paths)
        for index, path in enumerate(extended[3], 1):
            assert path[-1] == index and path == sorted(path)
            assert set(torch.where(extended[1][0, 0, index])[0].tolist()) == {0, *path}
            assert extended[2][index].item() == len(path)


def test_bridge_rejects_an_unbounded_or_unsupported_builder():
    with pytest.raises(ValueError, match="inclusive budget"):
        Model._build_tree(**bridge_args(), suffix_bridge=True)
    with pytest.raises(ValueError, match="inclusive budget"):
        Model._build_tree_budgeted_single_pass(**(bridge_args() | dict(score_priority_layout=False)), suffix_bridge=True)


def test_flat_clone_matches_all_sam_states_incrementally():
    rng = random.Random(719)
    for _ in range(30):
        history = [rng.randrange(24) for _ in range(400)]
        baseline, optimized = DynSAM(device="cpu"), DynSAM(device="cpu", clone_mode="flat")
        for start in range(0, len(history), 13):
            baseline.add_tokens(history[start:start + 13])
            optimized.add_tokens(history[start:start + 13])
            assert semantic_state(baseline) == semantic_state(optimized)
            for token in (0, 3, 7):
                assert baseline.lookup(token) == optimized.lookup(token)
        assert len({id(state.next) for state in optimized.states}) == len(optimized.states)


def test_clone_mode_is_explicit_and_survives_reset_without_leaking_state():
    draft = DraftModel(device="cpu")
    assert draft.sam_dyn.clone_mode == "deepcopy"
    history = [11, 22, 33, 44, 55, 22, 66, 44, 55, 22]
    draft.update_tokens(history)
    before = copy.deepcopy(semantic_state(draft.sam_dyn))
    draft.sam_dyn.set_clone_mode("flat")
    assert semantic_state(draft.sam_dyn) == before
    assert draft.lookup_continuation_tokens() == [66, 44, 55, 22]
    draft.reset()
    assert draft.sam_dyn.clone_mode == "flat" and draft.sam_dyn.max_length == 0
    draft.update_tokens(history)
    assert semantic_state(draft.sam_dyn) == before
    draft.sam_dyn.set_clone_mode("deepcopy")
    assert semantic_state(draft.sam_dyn) == before
    with pytest.raises(ValueError):
        draft.sam_dyn.set_clone_mode("unknown")
    assert draft.sam_dyn.clone_mode == "deepcopy"


def test_factorial_controls_change_only_bridge_and_copy_mode():
    from evaluation.eval_recap_suffix_bridge_clone import (
        CONTROL_POLICIES, EXPERIMENT, control_kwargs,
    )

    configs = {name: control_kwargs(EXPERIMENT, name) for name in CONTROL_POLICIES[EXPERIMENT]}
    assert len(configs) == 6
    baseline = configs["fixed_suffix"]
    for name in ("clone_only", "bridge_only", "bridge_and_clone"):
        for key, value in baseline.items():
            if key not in ("recycling_sam_clone_mode", "recycling_suffix_bridge"):
                assert configs[name][key] == value
    assert configs["clone_only"]["recycling_sam_clone_mode"] == "flat"
    assert not configs["clone_only"]["recycling_suffix_bridge"]
    assert configs["bridge_only"]["recycling_suffix_bridge"]
    assert configs["bridge_only"]["recycling_sam_clone_mode"] == "deepcopy"
    assert configs["no_suffix"]["recycling_suffix_length"] == 0
    assert all(config["recycling_total_node_budget"] == 63 for config in configs.values())


def test_summary_enrichment_checks_and_saves_all_policy_summaries(tmp_path):
    from evaluation.eval_recap_suffix_bridge_clone import (
        CONTROL_POLICIES, COUNTER_KEYS, DATASETS, EXPERIMENT,
        REQUIRED_SUMMARY_FIELDS, dataset_slug, enrich_summaries,
    )

    result = {"complete": True, "datasets": {}}
    for dataset in DATASETS:
        result["datasets"][dataset] = {}
        for name, config in CONTROL_POLICIES[EXPERIMENT].items():
            directory = tmp_path / dataset_slug(dataset) / name
            directory.mkdir(parents=True)
            bridge = int(config["recycling_suffix_bridge"])
            counter = {key: 0 for key in COUNTER_KEYS}
            counter.update(
                iterations=1, total_candidate_nodes=2 + bridge,
                max_candidate_nodes=2 + bridge, total_pre_bridge_candidate_nodes=2,
                suffix_bridge_added_nodes=bridge, suffix_bridge_active_iterations=bridge,
                suffix_bridge_opportunities=1, suffix_bridge_accepted_tokens=bridge,
                sam_clone_mode=config["recycling_sam_clone_mode"],
            )
            choice = dict(
                output_token_ids=[[1, 2]], acceptance_length=[[1]], new_tokens=[2],
                idxs=[1], recap_counters=counter,
            )
            rows = [dict(question_id=str(i), choices=[choice]) for i in range(2)]
            path = directory / "results.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = {field: 0 for field in REQUIRED_SUMMARY_FIELDS}
            summary.update(num_records=2, jsonl_path=str(path))
            result["datasets"][dataset][name] = summary
    checks = enrich_summaries(tmp_path, result, 2)
    assert len(list(tmp_path.glob("*/*/summary.json"))) == 24
    for dataset in DATASETS:
        assert checks[dataset]["clone_only_vs_fixed_suffix"]["identical_output_and_acceptance_samples"] == 2
        assert checks[dataset]["bridge_and_clone_vs_bridge_only"]["identical_tree_counter_samples"] == 2
        summary = result["datasets"][dataset]["bridge_only"]["suffix_component_counters"]
        assert summary["suffix_bridge_added_nodes"] == 2
        assert summary["suffix_bridge_accepted_tokens"] == 2
        assert summary["bridge_active_iteration_ratio"] == 1
