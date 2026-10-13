import copy

import torch

from evaluation.eval_recap_paper_controls import DATASETS, control_kwargs, make_stream, recovery_bin
from evaluation.eval_recap_path_coverage import build_paths, replay_paths
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model


def test_clean_initialization_and_past_only_queries():
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    predictions = {i: ([{1: 2, 2: 3, 3: 1, 4: 5}[token], 80], [0.8, 0.2]) for i, token in enumerate(tokens)}
    modes = ("U", "UC", "backoff", "uniform", "fusion")
    states = replay_paths(tokens, 3, predictions, 2, 4, 5, -1, modes, "none")
    assert states[0]["matched_node_budget"] == 0
    assert not any(value["available"] for value in states[0]["order_candidates"].values())
    assert all(not value["query_available"] for value in states[0]["modes"].values())
    altered = {p: ([99, 98], [0.8, 0.2]) if p >= 3 else row for p, row in predictions.items()}
    other = replay_paths(tokens, 3, altered, 2, 4, 5, -1, modes, "none")
    assert [s["matched_tree_sha256"] for s in states[0]["modes"].values()] == [s["matched_tree_sha256"] for s in other[0]["modes"].values()]
    rows = {1: {3: [4]}, 2: {(2, 3): [5]}, 3: {(1, 2, 3): [6]}}
    scores = {1: {3: [1.]}, 2: {(2, 3): [1.]}, 3: {(1, 2, 3): [1.]}}
    assert build_paths([1, 2, 3], 2, rows, scores, "U", 1, 1, 1, -1) == [[4]]
    assert build_paths([1, 2, 3], 2, rows, scores, "UC", 1, 1, 1, -1) == [[5]]
    assert build_paths([1, 2, 3], 2, rows, scores, "backoff", 1, 1, 1, -1) == [[6]]


def test_suffix_inclusive_budget_overlap_refill_and_no_future():
    args = dict(root_token=1, transitions=torch.empty(0, dtype=torch.long),
                transition_valid=torch.empty(0, dtype=torch.bool), width=2, depth=4,
                blocked_token_id=-1, host_transitions={1: [2, 8], 2: [3, 9], 3: [4, 10]},
                host_transition_scores={1: [.6, .4], 2: [.6, .4], 3: [.6, .4]}, score_priority_layout=True)
    flat, _, _, paths = Model._build_tree_budgeted(**args, node_budget=5, reserved_path_tokens=[2, 3])
    assert len(paths) == 5
    assert [2, 3] in Model._tree_path_token_ids(flat, paths)
    flat, _, _, paths = Model._build_tree_budgeted(**args, node_budget=2, reserved_path_tokens=[9, 8, 7, 6])
    assert len(paths) == 2 and Model._tree_path_token_ids(flat, paths) == [[9], [9, 8]]
    assert Model._build_tree_budgeted(**args, node_budget=0, reserved_path_tokens=[9])[0] == [1]
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    predictions = {i: ([2, 3], [.6, .4]) for i in range(len(tokens))}
    modes = ("backoff", "fusion", "backoff_suffix", "fusion_suffix")
    a = replay_paths(tokens, 3, predictions, 2, 4, 5, -1, modes)
    future_changed = tokens.copy()
    future_changed[4:] = [7] * (len(tokens) - 4)
    b = replay_paths(future_changed, 3, predictions, 2, 4, 5, -1, modes)
    assert [v["matched_tree_sha256"] for v in a[0]["modes"].values()] == [v["matched_tree_sha256"] for v in b[0]["modes"].values()]
    assert all(v["matched_nodes"] == state["matched_node_budget"] for state in a for v in state["modes"].values())


def test_stream_isolation_and_same_within_task_order():
    samples = [{"id": f"{d}:{i}", "dataset": d} for d in DATASETS for i in range(3)]
    original = copy.deepcopy(samples)
    blocked, alternating = (make_stream(samples, regime, 17) for regime in ("blocked", "alternating"))
    assert samples == original
    assert {s["id"] for s in blocked} == {s["id"] for s in alternating} == {s["id"] for s in samples}
    for dataset in DATASETS:
        assert [s["id"] for s in blocked if s["dataset"] == dataset] == [s["id"] for s in alternating if s["dataset"] == dataset]
    assert sum(s["task_switch"] for s in blocked) == 3
    assert all(s["task_switch"] for s in alternating[1:])
    assert [s["request_since_switch"] for s in blocked] == [0, 1, 2] * 4
    assert control_kwargs("scope", "carry")["recycling_cache_namespace"] != control_kwargs("scope", "task_reset")["recycling_cache_namespace"]
    assert control_kwargs("granularity", "U")["recycling_prompt_init"] == "none"
    assert control_kwargs("suffix", "suffix_only")["recycling_max_order"] == 0
    assert [recovery_bin(i) for i in (0, 1, 4, 5, 9, 10, 24, 25)] == ["1", "2-5", "2-5", "6-10", "6-10", "11-25", "11-25", "26+"]
