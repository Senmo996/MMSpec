import torch
from types import SimpleNamespace

from evaluation.eval_transition_kernel_ranker import (
    STATIC_CONFIG_KEY,
    _coverage_metrics,
    _path_metrics,
    summarize_records,
)
from method.sam_grounded.transition_kernel import (
    TransitionKernelBank,
    TransitionKernelConfig,
    build_config_grid,
    target_rank,
    valid_prompt_transition_positions,
)


def test_config_grid_is_deterministic_and_deduplicated():
    configs = build_config_grid(
        [0.0, 0.0, 1.0], [1], [0.1], [0.0, 0.5]
    )

    assert [config.key for config in configs] == [
        "tw0.00-k1-t0.10-vw0.00",
        "tw0.00-k1-t0.10-vw0.50",
        "tw1.00-k1-t0.10-vw0.00",
        "tw1.00-k1-t0.10-vw0.50",
    ]


def test_valid_prompt_positions_exclude_first_visual_and_special_tokens():
    token_ids = torch.tensor([9, 10, 11, 12, 13])
    visual_mask = torch.tensor([False, True, False, False, False])

    positions = valid_prompt_transition_positions(
        token_ids, visual_mask, excluded_token_ids=[12]
    )

    assert positions.tolist() == [2, 4]


def _toy_bank() -> TransitionKernelBank:
    return TransitionKernelBank(
        candidate_token_ids=torch.tensor([100, 101, 102]),
        source_token_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        source_context_vectors=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        source_candidate_logits=torch.tensor(
            [[9.0, 2.0, 1.0], [1.0, 2.0, 9.0]]
        ),
    )


def test_token_and_context_retrieval_select_different_transition_rows():
    bank = _toy_bank()
    token_config = TransitionKernelConfig(1.0, 1, 0.1, 0.0)
    context_config = TransitionKernelConfig(0.0, 1, 0.1, 0.0)
    query_token = torch.tensor([1.0, 0.0])
    query_context = torch.tensor([1.0, 0.0])

    token_scores = bank.score(
        query_token_vector=query_token,
        query_context_vector=query_context,
        config=token_config,
    )
    context_scores = bank.score(
        query_token_vector=query_token,
        query_context_vector=query_context,
        config=context_config,
    )

    assert target_rank(token_scores, bank.candidate_token_ids, 100) == 1
    assert target_rank(context_scores, bank.candidate_token_ids, 102) == 1


def test_visual_weight_one_preserves_static_candidate_order():
    bank = _toy_bank()
    config = TransitionKernelConfig(0.5, 2, 0.1, 1.0)

    scores = bank.score(
        query_token_vector=torch.tensor([0.0, 1.0]),
        query_context_vector=torch.tensor([1.0, 0.0]),
        config=config,
    )

    assert [target_rank(scores, bank.candidate_token_ids, token) for token in [100, 101, 102]] == [1, 2, 3]


def test_append_adds_verified_transition_and_grid_matches_single_score():
    bank = _toy_bank()
    bank.append(
        source_token_vector=torch.tensor([1.0, 1.0]),
        source_context_vector=torch.tensor([1.0, 1.0]),
        candidate_logits=torch.tensor([1.0, 10.0, 0.0]),
    )
    config = TransitionKernelConfig(0.5, 1, 0.05, 0.25)
    query = torch.tensor([1.0, 1.0])

    single = bank.score(
        query_token_vector=query,
        query_context_vector=query,
        config=config,
    )
    grid = bank.score_grid(
        query_token_vector=query,
        query_context_vector=query,
        configs=[config],
    )[config.key]

    assert bank.source_count == 3
    assert torch.allclose(single, grid)
    assert target_rank(single, bank.candidate_token_ids, 101) == 1
    assert target_rank(single, bank.candidate_token_ids, 999) is None


def test_from_prompt_aligns_token_context_and_next_token_rows():
    token_ids = torch.tensor([5, 6, 7, 8])
    token_embeddings = torch.eye(4)
    hidden_states = torch.flip(torch.eye(4), dims=[0])
    logits = torch.zeros(4, 20)
    logits[1, 10] = 8.0
    logits[2, 11] = 9.0
    logits[3, 12] = 10.0

    bank = TransitionKernelBank.from_prompt(
        candidate_token_ids=torch.tensor([10, 11, 12]),
        prompt_token_ids=token_ids,
        prompt_token_embeddings=token_embeddings,
        prompt_hidden_states=hidden_states,
        prompt_logits=logits,
        visual_mask=torch.tensor([False, False, False, False]),
        excluded_token_ids=[7],
    )

    assert bank.source_count == 2
    assert torch.equal(bank.source_candidate_scores.argmax(dim=-1), torch.tensor([0, 2]))


def _ranker_record(step, target, static_rank, kernel_rank, bridge=None):
    return {
        "sample_index": 0,
        "step_index": step,
        "target_token": target,
        "high_visual_state": True,
        "baseline_row_available": False,
        "bridge_candidates_before": bridge or [],
        "ranker_latency_ms": 0.1,
        "pools": {
            "observed_visual": {
                "target_ranks": {
                    STATIC_CONFIG_KEY: static_rank,
                    "tw0.50-k1-t0.10-vw0.50": kernel_rank,
                }
            }
        },
    }


def test_coverage_and_path_metrics_use_unconditional_state_denominators():
    records = [
        _ranker_record(0, 20, 5, 2, bridge=[30, 31, 32]),
        _ranker_record(1, 30, None, 1),
        _ranker_record(2, 40, None, None),
    ]

    coverage = _coverage_metrics(
        records, "observed_visual", "tw0.50-k1-t0.10-vw0.50"
    )
    path = _path_metrics(
        records, "observed_visual", "tw0.50-k1-t0.10-vw0.50", 3
    )

    assert coverage["num_states"] == 3
    assert coverage["pool_recall"] == 2 / 3
    assert coverage["top3_hit_rate"] == 2 / 3
    assert path["num_states_with_next"] == 2
    assert path["first_hits"] == 2
    assert path["path_hits"] == 1
    assert path["path_hit_rate"] == 0.5


def test_summary_selects_kernel_without_using_static_baseline():
    config = TransitionKernelConfig(0.5, 1, 0.1, 0.5)
    records = [
        _ranker_record(0, 20, 5, 2, bridge=[30, 31, 32]),
        _ranker_record(1, 30, None, 1),
    ]
    args = SimpleNamespace(
        topic_offset=0,
        max_new_token=8,
        candidate_pool_size=64,
        visual_threshold=0.55,
        candidate_row_budget=4,
        token_weights=[0.5],
        neighbors=[1],
        temperatures=[0.1],
        visual_weights=[0.5],
        default_latency_config=config.key,
        attn_implementation="sdpa",
        min_gate_states=2,
        min_top3_hit_rate=0.5,
        min_path_hit_rate=0.5,
    )

    summary = summarize_records(
        records,
        [{"inventory_construction_ms": 1.0}],
        [config],
        args,
    )

    selection = summary["within_split_selection"]
    assert selection["selected_config"] == config.key
    assert selection["selected_metrics"]["top3_hit_rate"] == 1.0
    assert selection["static_metrics"]["top3_hit_rate"] == 0.0
    assert summary["local_gate"]["decision"] == "go"
