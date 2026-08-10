import torch

from evaluation.eval_partial_layer_jit_ranker import (
    _candidate_scores,
    _consecutive_path_metrics,
    _layer_key,
    _normalized_layer_states,
    _optimistic_cost_model,
    _select_depth,
)


class _AddTen(torch.nn.Module):
    def forward(self, value):
        return value + 10


def test_intermediate_states_receive_final_norm_but_full_state_does_not():
    states = tuple(
        torch.tensor([[[float(depth), float(depth + 1)]]]) for depth in range(4)
    )

    result = _normalized_layer_states(
        states, [0, 2, 3], num_layers=3, final_norm=_AddTen()
    )

    assert torch.equal(result[_layer_key(0)], torch.tensor([10.0, 11.0]))
    assert torch.equal(result[_layer_key(2)], torch.tensor([12.0, 13.0]))
    assert torch.equal(result[_layer_key(3)], torch.tensor([3.0, 4.0]))


def test_candidate_projection_uses_only_selected_lm_head_rows():
    projection = torch.nn.Linear(2, 4, bias=False)
    projection.weight.data.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    )

    scores = _candidate_scores(
        torch.tensor([2.0, 1.0]), torch.tensor([3, 0]), projection
    )

    assert torch.equal(scores, torch.tensor([3.0, 2.0]))


def test_consecutive_path_requires_same_ranker_to_hit_both_steps():
    key = _layer_key(2)
    records = [
        {
            "sample_index": 0,
            "step_index": 0,
            "high_visual_state": True,
            "baseline_row_available": False,
            "pools": {"observed_visual": {"target_ranks": {key: 2}}},
        },
        {
            "sample_index": 0,
            "step_index": 1,
            "high_visual_state": False,
            "baseline_row_available": True,
            "pools": {"observed_visual": {"target_ranks": {key: 3}}},
        },
        {
            "sample_index": 0,
            "step_index": 2,
            "high_visual_state": True,
            "baseline_row_available": False,
            "pools": {"observed_visual": {"target_ranks": {key: 1}}},
        },
        {
            "sample_index": 0,
            "step_index": 3,
            "high_visual_state": False,
            "baseline_row_available": False,
            "pools": {"observed_visual": {"target_ranks": {key: 4}}},
        },
    ]

    metric = _consecutive_path_metrics(records, "observed_visual", key, 3)

    assert metric["num_states_with_next"] == 2
    assert metric["first_hits"] == 2
    assert metric["path_hits"] == 1
    assert metric["path_hit_rate"] == 0.5
    assert metric["second_given_first_rate"] == 0.5


def test_optimistic_recursive_cost_counts_root_and_each_first_branch():
    cost = _optimistic_cost_model(
        {"first_hit_rate": 0.4, "path_hit_rate": 0.1},
        depth=2,
        num_layers=28,
        branch_width=3,
    )

    assert cost["recursive_partial_tokens"] == 4
    assert cost["recursive_partial_full_forward_equivalents"] == 2 / 7
    assert cost["recursive_optimistic_saved_full_forward_equivalents"] == 0.5
    assert cost["recursive_optimistic_margin"] > 0
    assert cost["recursive_break_even"]


def test_depth_selection_requires_accuracy_specificity_and_break_even():
    depths = [1, 2, 4]
    metrics = {
        _layer_key(1): {"num_states": 100, "top3_hit_rate": 0.2},
        _layer_key(2): {"num_states": 100, "top3_hit_rate": 0.32},
        _layer_key(4): {"num_states": 100, "top3_hit_rate": 0.4},
    }
    paths = {
        _layer_key(1): {"first_hit_rate": 0.2, "path_hit_rate": 0.04},
        _layer_key(2): {"first_hit_rate": 0.32, "path_hit_rate": 0.06},
        _layer_key(4): {"first_hit_rate": 0.4, "path_hit_rate": 0.08},
    }
    specificity = {
        _layer_key(1): {"observed_minus_control_pp": 2.0},
        _layer_key(2): {"observed_minus_control_pp": 1.0},
        _layer_key(4): {"observed_minus_control_pp": 3.0},
    }

    selected, eligible, rows, costs, gates = _select_depth(
        metrics,
        paths,
        specificity,
        depths,
        num_layers=28,
        max_selection_depth=4,
        branch_width=3,
        min_gate_states=20,
        min_top3_hit_rate=0.25,
        min_path_hit_rate=0.05,
    )

    assert selected == _layer_key(2)
    assert eligible
    assert [row["config"] for row in rows] == [_layer_key(2)]
    assert costs[_layer_key(2)]["recursive_break_even"]
    assert not gates[_layer_key(4)]["optimistic_break_even_pass"]
