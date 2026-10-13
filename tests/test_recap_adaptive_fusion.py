import copy
import math
import random

import pytest
import torch

from evaluation.eval_recap_adaptive_fusion import replay
from method.sam_grounded.fusion_weighting import FusionRule, OFFLINE_RULES, concentration, source_weights
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model


def test_concentration_and_parameter_free_weighting():
    assert concentration([1.] * 8) == pytest.approx(1 / 8)
    assert concentration([1., 0.]) == 1.
    rows = [([1, 2], [.5, .5]), ([3, 4], [.99, .01])]
    assert source_weights(rows, 'confidence')[1] > source_weights(rows, 'confidence')[0]
    assert source_weights(rows, 'confidence_gate') == [0., 1.]
    assert source_weights(rows[::-1], 'confidence_gate') == [1., 0.]
    assert source_weights([rows[0], rows[0]], 'confidence_gate') == [1., 0.]
    assert source_weights(rows, 'prior_confidence')[0] == pytest.approx(.7 * .5)
    with pytest.raises(ValueError):
        concentration([-1., 2.])


def test_gate_and_normalization_do_not_modify_sources():
    rows = [([1, 2], [.5, .5]), ([3, 4], [.99, .01])]
    before = copy.deepcopy(rows)
    assert FusionRule('confidence_gate')(rows, 2) == rows[1]
    for mode in OFFLINE_RULES:
        ids, scores = FusionRule(mode)(rows, 2)
        assert len(ids) == len(set(ids)) == 2
        assert sum(scores) == pytest.approx(1.)
        assert all(math.isfinite(p) and p >= 0 for p in scores)
        assert FusionRule(mode)([], 8) == ([], [])
        assert FusionRule(mode)([rows[0]], 2) == rows[0]
    assert rows == before


def test_fixed_hook_preserves_production_tree_and_available_rank_weights():
    rng = random.Random(17)
    for missing in ((), (2,), (3,), (2, 3)):
        maps, scores = {}, {}
        for order in (1, 2, 3):
            maps[order], scores[order] = {}, {}
            if order in missing:
                continue
            for last in range(1, 11):
                key = last if order == 1 else tuple([1] * (order - 1) + [last])
                ids = rng.sample(range(1, 16), 8)
                probs = sorted([rng.random() for _ in ids], reverse=True)
                maps[order][key] = ids
                scores[order][key] = [p / sum(probs) for p in probs]
        kwargs = dict(root_token=1, transitions=torch.empty(0, dtype=torch.long),
            transition_valid=torch.empty(0, dtype=torch.bool), width=8, depth=10,
            node_budget=63, blocked_token_id=15, root_previous_token=1,
            root_previous_previous_token=1, host_transitions=maps[1],
            host_transition_scores=scores[1], host_context_transitions=maps[2],
            host_context_transition_scores=scores[2], host_trigram_transitions=maps[3],
            host_trigram_transition_scores=scores[3], context_candidate_mode='fusion', score_priority_layout=True)
        a = Model._build_tree(**kwargs)
        b = Model._build_tree(**kwargs, row_fusion=FusionRule('fixed'))
        assert a[0] == b[0] and a[3] == b[3]
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
        for mode in OFFLINE_RULES:
            output = Model._build_tree(**kwargs, row_fusion=FusionRule(mode))
            assert len(output[0]) - 1 == len(output[3]) <= 63
        with pytest.raises(ValueError, match='shared Python'):
            Model._build_tree(**kwargs, fast_score_priority_layout=True, row_fusion=FusionRule('fixed'))


def test_replay_uses_past_rows_only_and_equal_actual_tree_budgets():
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    record = dict(input_token_ids=tokens, prompt_length=3, blocked_token_id=-1,
        prediction_rows=[[p, [tokens[(p+1) % len(tokens)], 80], [.8, .2]] for p in range(len(tokens))])
    a = replay(record)
    changed = copy.deepcopy(record)
    for row in changed['prediction_rows']:
        if row[0] >= 3:
            row[1:] = [[98, 99], [.9, .1]]
    b = replay(changed)
    for mode in OFFLINE_RULES:
        assert a[0]['methods'][mode]['candidates'] == b[0]['methods'][mode]['candidates']
        assert a[0]['methods'][mode]['weights'] == b[0]['methods'][mode]['weights']
    assert all(all(p < state['position'] for p in state['source_positions']) for state in a)
    assert all(0 <= state['matched_node_budget'] <= 63 for state in a)
