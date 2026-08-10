from types import SimpleNamespace

import torch

from evaluation.eval_visual_lexical_inventory import summarize_records
from method.sam_grounded.counterfactual_probes import build_visual_probe_layout
from method.sam_grounded.visual_lexical_inventory import (
    build_visual_lexical_inventories,
    fuse_equal_budget_candidates,
    interleave_rankings,
    rank_scores,
)


def test_rank_scores_filters_special_and_out_of_tokenizer_vocabulary_ids():
    scores = torch.tensor([0.0, 5.0, 4.0, 9.0, 8.0])

    ranked = rank_scores(
        scores,
        3,
        excluded_token_ids=[1],
        valid_vocab_size=4,
    )

    assert ranked == [3, 2, 0]


def test_round_robin_ranking_is_unique_and_source_balanced():
    assert interleave_rankings([[1, 2, 3], [1, 4, 5]], 5) == [1, 2, 4, 3, 5]


def test_visual_region_contrast_suppresses_text_generic_tokens():
    logits = torch.zeros((6, 8), dtype=torch.float32)
    logits[torch.tensor([0, 5]), 0] = 10.0
    logits[torch.tensor([0, 5]), 3] = 5.0
    logits[torch.tensor([1, 2, 3, 4]), 0] = 10.0
    logits[torch.tensor([1, 2]), 1] = 9.0
    logits[torch.tensor([3, 4]), 2] = 8.0
    visual_mask = torch.tensor([False, True, True, True, True, False])
    layout = build_visual_probe_layout(
        visual_mask,
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        spatial_merge_size=1,
        num_regions=2,
    )

    inventories = build_visual_lexical_inventories(
        logits,
        visual_mask,
        layout,
        question_token_ids=[5, 6, 5],
        max_pool_size=4,
        excluded_token_ids=[7],
        valid_vocab_size=8,
    )

    assert inventories["visual_mean"][0] == 0
    assert inventories["visual_region_contrast"][:2] == [1, 2]
    assert inventories["visual_region_contrast_rr"][:2] == [1, 2]
    assert inventories["question_lexical"] == [5, 6]
    assert inventories["visual_question_combined"] == [1, 5, 2, 6]
    assert 7 not in inventories["visual_region_contrast"]


def test_equal_budget_fusion_anchors_baseline_and_fills_idle_rows():
    assert fuse_equal_budget_candidates(
        [10, 11, 12, 13], [20, 21, 22], budget=4, inventory_slots=2
    ) == [10, 11, 20, 21]
    assert fuse_equal_budget_candidates(
        [10, 11, 12, 13], [11, 20], budget=4, inventory_slots=2
    ) == [10, 11, 20, 12]
    assert fuse_equal_budget_candidates(
        [], [20, 21, 22, 23, 24], budget=4, inventory_slots=2
    ) == [20, 21, 22, 23]


def test_summary_gate_requires_image_specific_oracle_gain():
    states = [
        {
            "sample_index": 0,
            "target_token": 20,
            "baseline_candidates": [1, 2, 3, 4],
            "baseline_row_available": True,
            "high_visual_state": True,
        },
        {
            "sample_index": 1,
            "target_token": 30,
            "baseline_candidates": [],
            "baseline_row_available": False,
            "high_visual_state": True,
        },
    ]
    inventory_records = [
        {
            "sample_index": 0,
            "inventory_construction_ms": 1.0,
            "inventories": {"visual_region_contrast_rr": [20, 21, 22, 23]},
        },
        {
            "sample_index": 1,
            "inventory_construction_ms": 2.0,
            "inventories": {"visual_region_contrast_rr": [30, 31, 32, 33]},
        },
    ]
    args = SimpleNamespace(
        inventory_pool_sizes=[4],
        candidate_budgets=[4],
        inventory_slots=[2],
        primary_source="visual_region_contrast_rr",
        primary_inventory_pool_size=4,
        primary_candidate_budget=4,
        primary_inventory_slots=2,
        inventory_regions=2,
        visual_threshold=0.55,
        max_new_token=8,
        attn_implementation="sdpa",
        min_gate_states=2,
        min_oracle_gain=0.5,
        min_specificity_gain=0.5,
    )

    summary = summarize_records(states, inventory_records, args)

    observed = summary["gate"]["observed"]
    assert observed["baseline_coverage"] == 0.0
    assert observed["fixed_budget_coverage"] == 1.0
    assert observed["oracle_gain_pp"] == 100.0
    assert summary["gate"]["specificity_delta_pp"] == 100.0
    assert summary["gate"]["decision"] == "go"
