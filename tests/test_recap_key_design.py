"""Candidate-budget, causal-indexing and fixed-pool tests; no model loading."""

import copy
import json
import random

import pytest
import torch

from evaluation.eval_recap_key_design import (
    MODES, dense_rank, first_prompt_positions, fuse_rows, normalize_rows,
    replay_keys, summarize,
)


def toy():
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    rows = {p: ([10 + p, 30 + p], [.8, .2]) for p in range(len(tokens)-1)}
    positions = sorted(rows)
    features = torch.randn(len(positions), 6, generator=torch.Generator().manual_seed(3))
    features[0] = 0  # empty-prefix source key
    return tokens, rows, positions, features


def test_query_precedes_update_and_only_uses_predecessor_features():
    tokens, rows, positions, features = toy()
    before = copy.deepcopy(rows)
    original = replay_keys(tokens, 3, rows, positions, features, 2, -1)
    changed_rows = {p: ([99, 98], [.6, .4]) if p >= 3 else row for p, row in rows.items()}
    changed_features = features.clone()
    # Row-index 3 is h_2 and is legitimately available at the first query.
    # Row-index 4 is h_3 and is NOT available there.
    changed_features[4:] *= -100
    changed_tokens = tokens[:4] + [90] * (len(tokens)-4)
    changed = replay_keys(changed_tokens, 3, changed_rows, positions, changed_features, 2, -1)
    for mode in MODES:
        for field in ("candidate_token_ids", "source_positions", "source_similarities"):
            assert original[0]["modes"][mode][field] == changed[0]["modes"][mode][field]
    assert rows == before
    assert original[0]["query_key_position"] == 2
    assert original[0]["pool_positions"] == [0, 1, 2]
    for state in original:
        assert max(state["pool_positions"]) < state["position"]
        for result in state["modes"].values():
            assert set(result["source_positions"]) <= set(state["pool_positions"])
            assert result["source_key_positions"] == [p-1 for p in result["source_positions"]]
            assert len(result["candidate_token_ids"]) <= 2


def test_dense_token_filter_and_past_only_centering():
    features = torch.tensor([[0., 1.], [1., 0.], [2., 0.], [1000., -900.]])
    index = {i: i for i in range(4)}
    tokens = [5, 6, 5, 9]
    query = torch.tensor([1., 0.])
    positions, values = dense_rank(query, [0, 1, 2], index, features, tokens, 5, True, False)
    assert positions == [2, 0] and values == [1., 0.]
    centered = dense_rank(query, [0, 1, 2], index, features, tokens, 5, True, True)
    changed = features.clone()
    changed[3] *= -1e6
    assert dense_rank(query, [0, 1, 2], index, changed, tokens, 5, True, True) == centered
    assert dense_rank(query, [0], index, features, tokens, 8, True, False) == ([], [])


def test_fusion_matches_production_on_random_rows():
    from evaluation.eval_recap_memory_source import root_candidates
    rng = random.Random(731)
    for _ in range(120):
        top_k = rng.randrange(1, 9)
        banks = {order: {} for order in (1, 2, 3)}
        rows, sources = {}, []
        tokens = [1, 2, 3, 4, 1, 2, 3]
        for order in (3, 2, 1):
            if rng.random() < .3:
                continue
            ids = rng.sample(range(5, 40), top_k)
            values = sorted([rng.random() + .01 for _ in ids], reverse=True)
            scores = [x / sum(values) for x in values]
            source_position = order - 1
            rows[source_position] = ids, scores
            banks[order][tuple(tokens[-order:])] = (source_position, ids, scores)
            sources.append(source_position)
        for mode, selected in (("fusion", sources), ("backoff", sources[:1])):
            expected = root_candidates(tokens, 6, banks, mode, top_k, -1)
            assert set(fuse_rows(selected, rows, top_k, -1)) == set(expected)


def test_source_slot_budget_and_lexical_production_parity():
    tokens, rows, positions, features = toy()
    states = replay_keys(tokens, 3, rows, positions, features, 2, -1, verify_production=True)
    for state in states:
        assert state["modes"]["dense_token_fusion"]["num_source_slots"] <= state["lexical_source_slot_budget"]
        assert state["modes"]["dense_centered_token_fusion"]["num_source_slots"] <= state["lexical_source_slot_budget"]
        assert state["modes"]["dense_fusion"]["num_source_slots"] <= 3
        for mode in ("dense_token_top1", "dense_token_fusion", "dense_centered_token_fusion"):
            assert all(tokens[p] == tokens[state["position"]] for p in state["modes"][mode]["source_positions"])


def test_full_query_recall_counts_uncovered_states_as_misses():
    tokens, rows, positions, features = toy()
    states = replay_keys(tokens, 3, rows, positions, features, 2, -1)
    states[0]["modes"]["lexical_fusion"].update(candidate_token_ids=[13], available=True, hit=True)
    for state in states[1:]:
        state["modes"]["lexical_fusion"].update(candidate_token_ids=[], available=False, hit=False)
    records = [{"dataset": "toy", "target_image_identity": "im1", "states": states}]
    summary = summarize(records, 1, 2, "results.jsonl")
    metric = summary["datasets"]["toy"]["modes"]["lexical_fusion"]
    assert metric["conditional_recall_at_k"] == 1.
    assert metric["query_coverage"] == metric["recall_at_k_all_states"] == 1 / len(states)
    assert summary["complete"]
    assert not summary["formal_speedup_evaluated"]
    assert not summary["accepted_prefix_length_evaluated"]
    json.dumps(summary, allow_nan=False)


def test_empty_query_summary_and_prefix_first_occurrences():
    assert first_prompt_positions([1, 2, 1, 3, 2]) == [0, 1, 3]
    empty = summarize([], 400, 8, "results.jsonl")
    assert not empty["complete"] and empty["num_records"] == 0
    json.dumps(empty, allow_nan=False)


def test_cached_topk_normalization_and_invalid_rows():
    assert normalize_rows([[1, [8, 9], [.3, .1]]], 2)[1][1] == pytest.approx([.75, .25])
    for rows in ([[1, [8, 8], [.3, .1]]], [[1, [8, 9], [0., 0.]]],
                 [[1, [8, 9], [float("nan"), .1]]], [[1, [8, 9], [-.1, .2]]]):
        with pytest.raises(ValueError, match="invalid saved"):
            normalize_rows(rows, 2)


def test_feature_alignment_validation():
    tokens, rows, positions, features = toy()
    with pytest.raises(ValueError, match="align"):
        replay_keys(tokens, 3, rows, positions[:-1], features, 2, -1)


def test_blocked_candidates_and_duplicate_source_rows():
    rows = {0: ([8, 9], [.8, .2])}
    assert fuse_rows([0, 0, 0], rows, 2, 8) == [9]
    assert fuse_rows([], rows, 2, -1) == []


def test_aliasing_subgroup_uses_past_labels_including_evicted_rows():
    tokens = [1, 2, 3] * 5
    positions = list(range(len(tokens)-1))
    rows = {p: ([20+p, 60+p], [.8, .2]) for p in positions}
    features = torch.ones(len(positions), 2)
    states = replay_keys(tokens, 3, rows, positions, features, 2, -1)
    assert not states[0]["historical_trigram_aliasing"]
    later = next(s for s in states if s["position"] == 11)
    assert later["historical_trigram_aliasing"]
    assert 2 not in later["pool_positions"]  # evicted evidence is not retrievable
