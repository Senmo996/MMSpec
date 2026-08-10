import torch

from method.sam_grounded.counterfactual_probes import (
    build_counterfactual_attention_mask,
    build_tree_counterfactual_attention_mask,
    build_visual_probe_layout,
    compose_candidate_view_logits,
    cover_candidates,
    full_view_candidates,
    make_repeated_position_ids,
    multiview_jsd,
    topk_union_size,
)


def test_visual_probe_layout_uses_merged_spatial_quadrants():
    visual_mask = torch.tensor(
        [False, True, True, True, True, True, True, True, True, False]
    )
    layout = build_visual_probe_layout(
        visual_mask,
        image_grid_thw=torch.tensor([[1, 4, 8]]),
        spatial_merge_size=2,
        num_regions=4,
    )

    assert layout.used_grid_metadata
    assert layout.visual_positions.tolist() == list(range(1, 9))
    assert layout.region_ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_visual_probe_layout_falls_back_on_inconsistent_metadata():
    layout = build_visual_probe_layout(
        torch.tensor([True, True, True, True, True]),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        spatial_merge_size=2,
        num_regions=2,
    )

    assert not layout.used_grid_metadata
    assert layout.region_ids.tolist() == [0, 0, 0, 1, 1]


def test_counterfactual_mask_drops_only_assigned_visual_region():
    visual_mask = torch.tensor(
        [False, True, True, True, True, True, True, True, True, False]
    )
    layout = build_visual_probe_layout(
        visual_mask,
        image_grid_thw=torch.tensor([[1, 4, 8]]),
        spatial_merge_size=2,
        num_regions=4,
    )
    mask = build_counterfactual_attention_mask(
        prefix_length=10,
        layout=layout,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    minimum = torch.finfo(torch.float32).min

    assert mask.shape == (1, 1, 5, 15)
    assert torch.all(mask[0, 0, 0, :10] == 0)
    assert torch.all(mask[0, 0, 1, torch.tensor([1, 2])] == minimum)
    assert torch.all(mask[0, 0, 1, torch.tensor([3, 4, 5, 6, 7, 8])] == 0)
    for query in range(5):
        assert mask[0, 0, query, 10 + query] == 0
        sibling_slots = [10 + index for index in range(5) if index != query]
        assert torch.all(mask[0, 0, query, sibling_slots] == minimum)


def test_tree_counterfactual_mask_keeps_probes_out_of_acceptance_tree():
    visual_mask = torch.tensor([False, True, True, True, True, False])
    layout = build_visual_probe_layout(
        visual_mask,
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        spatial_merge_size=1,
        num_regions=2,
    )
    tree = torch.tensor(
        [[[[True, False, False], [True, True, False], [True, False, True]]]]
    )
    mask = build_tree_counterfactual_attention_mask(
        prefix_length=6,
        tree_mask=tree,
        layout=layout,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    minimum = torch.finfo(torch.float32).min

    assert mask.shape == (1, 1, 5, 11)
    assert torch.all(mask[0, 0, :3, 9:] == minimum)
    assert mask[0, 0, 1, 6] == 0
    assert mask[0, 0, 1, 7] == 0
    assert mask[0, 0, 1, 8] == minimum
    assert mask[0, 0, 3, 9] == 0
    assert mask[0, 0, 3, 10] == minimum
    assert torch.all(mask[0, 0, :3, 1:5] == 0)


def test_cover_candidates_preserve_full_anchors_and_add_view_evidence():
    logits = torch.full((3, 10), -100.0)
    logits[0, torch.tensor([0, 1, 2, 3])] = torch.tensor([10.0, 9.0, 8.0, 7.0])
    logits[1, torch.tensor([5, 0, 1, 2])] = torch.tensor([10.0, 9.0, 8.0, 7.0])
    logits[2, torch.tensor([7, 0, 1, 3])] = torch.tensor([10.0, 9.0, 8.0, 7.0])

    assert full_view_candidates(logits[0], 4) == [0, 1, 2, 3]
    assert cover_candidates(logits, 4, anchor_fraction=0.5) == [0, 1, 5, 7]
    assert topk_union_size(logits, 1) == 3


def test_multiview_jsd_is_zero_for_identical_views_and_positive_otherwise():
    identical = torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
    divergent = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])

    assert abs(multiview_jsd(identical)) < 1e-7
    assert multiview_jsd(divergent) > 0.6


def test_evidence_logits_amplify_tokens_hurt_by_region_removal():
    full = torch.tensor([4.0, 3.0, 2.0])
    dropped = torch.tensor([[1.0, 3.0, 4.0]])

    masked = compose_candidate_view_logits(full, dropped, mode="masked")
    evidence = compose_candidate_view_logits(full, dropped, mode="evidence")
    both = compose_candidate_view_logits(full, dropped, mode="both")

    assert torch.equal(masked, torch.tensor([[4.0, 3.0, 2.0], [1.0, 3.0, 4.0]]))
    assert torch.equal(evidence, torch.tensor([[4.0, 3.0, 2.0], [7.0, 3.0, 0.0]]))
    assert both.shape == (3, 3)


def test_repeated_position_ids_share_one_logical_position():
    positions = make_repeated_position_ids(
        logical_position=12,
        query_length=5,
        rope_deltas=torch.tensor([[-3]]),
        device=torch.device("cpu"),
    )

    assert positions.shape == (3, 1, 5)
    assert torch.all(positions == 9)
