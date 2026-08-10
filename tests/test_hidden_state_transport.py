import pytest
import torch

from evaluation.eval_hidden_state_transport_ranker import (
    _fuse_rankings,
    _paired_control_metrics,
    _recursive_path_metrics,
    _select_config,
    _select_recursive_policy,
)
from method.sam_grounded.hidden_state_transport import (
    HiddenStateTransportBank,
    HiddenStateTransportConfig,
    build_transport_config_grid,
)
from method.sam_grounded.transition_kernel import target_rank


def _bank() -> HiddenStateTransportBank:
    return HiddenStateTransportBank(
        candidate_token_ids=torch.tensor([100, 101, 102]),
        candidate_projection_weight=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        ),
        candidate_projection_bias=None,
        source_token_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        source_context_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        source_post_hidden=torch.tensor([[0.0, 3.0], [-3.0, 0.0]]),
        source_post_visual_mask=torch.tensor([False, True]),
    )


def test_transport_grid_keys_include_mode_scope_and_are_unique():
    configs = build_transport_config_grid(
        [0.5], [4], [0.1], ["delta_full", "post_state"],
        ["all_text", "post_visual_text"], [0.0, 0.5]
    )

    assert len(configs) == 8
    assert len({config.key for config in configs}) == 8
    assert configs[0].key == "tw0.50-k4-t0.10-mdf-sall-vw0.00"


def test_delta_transport_changes_parent_projection_using_retrieved_transition():
    bank = _bank()
    config = HiddenStateTransportConfig(
        token_weight=1.0,
        neighbors=1,
        temperature=0.1,
        transport_mode="delta_full",
        source_scope="all_text",
        visual_weight=0.0,
    )

    scores = bank.score(
        query_token_vector=torch.tensor([1.0, 0.0]),
        query_context_hidden=torch.tensor([1.0, 0.0]),
        config=config,
    )
    predicted = bank.transport_hidden(
        query_token_vector=torch.tensor([1.0, 0.0]),
        query_context_hidden=torch.tensor([1.0, 0.0]),
        config=config,
    )

    assert target_rank(scores, bank.candidate_token_ids, 101) == 1
    assert target_rank(
        bank.project_hidden(predicted), bank.candidate_token_ids, 101
    ) == 1


@pytest.mark.parametrize("transport_mode", ["delta_half", "delta_full", "post_state"])
def test_preprojected_candidate_logits_match_hidden_projection(transport_mode):
    bank = _bank()
    config = HiddenStateTransportConfig(
        token_weight=0.5,
        neighbors=2,
        temperature=0.2,
        transport_mode=transport_mode,
        source_scope="all_text",
        visual_weight=0.0,
    )
    query_token = torch.tensor([0.6, 0.8])
    query_context = torch.tensor([0.25, -0.75])
    query_logits = (
        bank.candidate_projection_weight @ query_context
    )

    reference = bank.score(
        query_token_vector=query_token,
        query_context_hidden=query_context,
        config=config,
    )
    preprojected = bank.score(
        query_token_vector=query_token,
        query_context_hidden=query_context,
        query_candidate_logits=query_logits,
        config=config,
    )

    assert torch.allclose(preprojected, reference, atol=1e-5, rtol=1e-5)


def test_post_visual_scope_excludes_pre_image_transition():
    bank = _bank()
    config = HiddenStateTransportConfig(
        token_weight=1.0,
        neighbors=1,
        temperature=0.1,
        transport_mode="post_state",
        source_scope="post_visual_text",
        visual_weight=0.0,
    )

    scores = bank.score(
        query_token_vector=torch.tensor([1.0, 0.0]),
        query_context_hidden=torch.tensor([1.0, 0.0]),
        config=config,
    )

    assert target_rank(scores, bank.candidate_token_ids, 102) == 1


def test_visual_weight_one_preserves_candidate_pool_order():
    bank = _bank()
    config = HiddenStateTransportConfig(
        0.5, 2, 0.1, "delta_half", "all_text", 1.0
    )

    scores = bank.score(
        query_token_vector=torch.tensor([0.0, 1.0]),
        query_context_hidden=torch.tensor([1.0, 0.0]),
        config=config,
    )

    assert [target_rank(scores, bank.candidate_token_ids, token) for token in [100, 101, 102]] == [1, 2, 3]


def test_append_adds_online_transition_to_post_visual_scope():
    bank = _bank()
    bank.append(
        source_token_vector=torch.tensor([1.0, 1.0]),
        source_context_hidden=torch.tensor([1.0, 1.0]),
        source_post_hidden=torch.tensor([0.0, 5.0]),
    )
    config = HiddenStateTransportConfig(
        1.0, 1, 0.05, "post_state", "post_visual_text", 0.0
    )
    query = torch.tensor([1.0, 1.0])

    single = bank.score(
        query_token_vector=query,
        query_context_hidden=query,
        config=config,
    )
    grid = bank.score_grid(
        query_token_vector=query,
        query_context_hidden=query,
        configs=[config],
    )[config.key]

    assert bank.source_count == 3
    assert bank.post_visual_source_count == 2
    assert torch.allclose(single, grid)
    assert target_rank(single, bank.candidate_token_ids, 101) == 1


def test_reserved_bank_appends_in_place_until_capacity_is_exhausted():
    bank = HiddenStateTransportBank(
        candidate_token_ids=torch.tensor([100, 101, 102]),
        candidate_projection_weight=torch.eye(3, 2),
        candidate_projection_bias=None,
        source_token_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        source_context_vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        source_post_hidden=torch.tensor([[0.0, 3.0], [-3.0, 0.0]]),
        source_post_visual_mask=torch.tensor([False, True]),
        reserve_capacity=4,
    )
    storage_pointer = bank.source_token_vectors.data_ptr()

    bank.append(
        source_token_vector=torch.tensor([1.0, 1.0]),
        source_context_hidden=torch.tensor([1.0, 1.0]),
        source_post_hidden=torch.tensor([0.0, 5.0]),
    )

    assert bank.source_count == 3
    assert bank.source_token_vectors.data_ptr() == storage_pointer
    assert torch.allclose(
        bank.source_token_vectors[2],
        torch.nn.functional.normalize(torch.tensor([1.0, 1.0]), dim=0),
    )


def test_from_prompt_marks_only_transitions_after_last_visual_position():
    token_ids = torch.tensor([5, 6, 7, 8, 9])
    embeddings = torch.eye(5)
    hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    projection = torch.arange(60, dtype=torch.float32).reshape(20, 3)

    bank = HiddenStateTransportBank.from_prompt(
        candidate_token_ids=torch.tensor([10, 11]),
        output_projection_weight=projection,
        output_projection_bias=None,
        prompt_token_ids=token_ids,
        prompt_token_embeddings=embeddings,
        prompt_hidden_states=hidden,
        visual_mask=torch.tensor([False, True, True, False, False]),
        excluded_token_ids=[],
    )

    assert bank.source_count == 2
    assert bank.post_visual_source_count == 2
    assert torch.equal(
        bank.candidate_projection_weight, projection.index_select(0, torch.tensor([10, 11]))
    )


def test_paired_specificity_uses_the_same_state_denominator():
    key = "transport"
    records = [
        {
            "high_visual_state": True,
            "baseline_row_available": False,
            "pools": {
                "observed_visual": {"target_ranks": {key: 2}},
                "mismatched_visual": {"target_ranks": {key: None}},
            },
        },
        {
            "high_visual_state": True,
            "baseline_row_available": False,
            "pools": {
                "observed_visual": {"target_ranks": {key: 8}},
                "mismatched_visual": {"target_ranks": {key: 1}},
            },
        },
        {
            "high_visual_state": True,
            "baseline_row_available": False,
            "pools": {
                "observed_visual": {"target_ranks": {key: 1}},
            },
        },
    ]

    paired = _paired_control_metrics(records, key)

    assert paired["num_paired_states"] == 2
    assert paired["observed_hits"] == 1
    assert paired["control_hits"] == 1
    assert paired["observed_minus_control_pp"] == 0.0
    assert paired["observed_only_hits"] == 1
    assert paired["control_only_hits"] == 1


def test_selection_requires_precision_path_and_positive_specificity():
    metrics = {
        "good": {"top3_hit_rate": 0.2, "top1_hit_rate": 0.1, "mrr": 0.15},
        "generic": {"top3_hit_rate": 0.3, "top1_hit_rate": 0.2, "mrr": 0.2},
    }
    paths = {
        "good": {"path_hit_rate": 0.06},
        "generic": {"path_hit_rate": 0.08},
    }
    specificity = {
        "good": {"observed_minus_control_pp": 1.0},
        "generic": {"observed_minus_control_pp": -2.0},
    }

    selected, has_eligible, eligible = _select_config(
        metrics,
        paths,
        specificity,
        ["good", "generic"],
        min_top3_hit_rate=0.15,
        min_path_hit_rate=0.05,
    )

    assert selected == "good"
    assert has_eligible
    assert eligible == ["good"]


def test_recursive_ranking_fusion_preserves_fixed_width_and_uniqueness():
    assert _fuse_rankings(
        [1, 2, 3], [2, 4, 5], primary_slots=2, width=3
    ) == [1, 2, 4]
    assert _fuse_rankings(
        [], [4, 5, 6], primary_slots=2, width=3
    ) == [4, 5, 6]


def test_recursive_path_metric_uses_branch_for_actual_first_target():
    config = "recursive"
    records = [
        {
            "sample_index": 0,
            "step_index": 0,
            "target_token": 20,
            "high_visual_state": True,
            "baseline_row_available": False,
            "recursive_paths": {
                config: {
                    "first_candidates": [20, 21, 22],
                    "branches": {"20": {"text_hst": [30, 31, 32]}},
                }
            },
        },
        {
            "sample_index": 0,
            "step_index": 1,
            "target_token": 30,
            "high_visual_state": True,
            "baseline_row_available": False,
            "recursive_paths": {
                config: {"first_candidates": [], "branches": {}}
            },
        },
    ]

    metric = _recursive_path_metrics(records, config, "text_hst")

    assert metric["num_states_with_next"] == 1
    assert metric["first_hit_rate"] == 1.0
    assert metric["path_hit_rate"] == 1.0


def test_recursive_selection_prefers_valid_path_action():
    metrics = {"a": {"top3_hit_rate": 0.2}, "b": {"top3_hit_rate": 0.3}}
    specificity = {
        "a": {"observed_minus_control_pp": 1.0},
        "b": {"observed_minus_control_pp": -1.0},
    }
    recursive = {
        "a": {
            "text_hst": {"path_hit_rate": 0.06},
            "visual_hst": {"path_hit_rate": 0.04},
        },
        "b": {"text_hst": {"path_hit_rate": 0.1}},
    }

    config, policy, eligible = _select_recursive_policy(
        primary_metrics=metrics,
        paired_specificity=specificity,
        recursive_metrics=recursive,
        config_keys=["a", "b"],
        min_top3_hit_rate=0.15,
        min_path_hit_rate=0.05,
    )

    assert (config, policy) == ("a", "text_hst")
    assert len(eligible) == 1
