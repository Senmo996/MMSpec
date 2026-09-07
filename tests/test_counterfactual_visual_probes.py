import torch
from types import SimpleNamespace

from evaluation.selective_reuse_protocol import (
    PIXEL_PROBE_PROTOCOL,
    apply_teacher_forced_pixel_probes,
    build_pixel_probe_views,
    deterministic_cluster_split,
    raw_patch_region_ids,
)
from evaluation.selective_reuse_content_ablation_protocol import (
    CONTENT_ABLATION_PROBE_PROTOCOL,
    apply_teacher_forced_content_ablation_probes,
    build_whole_image_ablation_views,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (
    COUNTERFACTUAL_BANK_PROTOCOL,
    apply_teacher_forced_counterfactual_bank_probes,
    build_counterfactual_bank_views,
    build_matched_wrong_image_pairs,
)

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


def test_cluster_split_keeps_duplicate_images_together_and_is_reproducible():
    keys = ["image-a", "image-b", "image-a", "image-c", "image-d"]
    first = deterministic_cluster_split(
        keys, discovery_fraction=0.5, seed=17, stratum="toy"
    )
    second = deterministic_cluster_split(
        keys, discovery_fraction=0.5, seed=17, stratum="toy"
    )

    assert first == second
    assert first[0] == first[2]
    assert set(first) == {"discovery", "heldout"}


def test_raw_patch_regions_follow_qwen_premerge_flatten_order():
    region_ids = raw_patch_region_ids(
        torch.tensor([[1, 4, 8]]),
        spatial_merge_size=2,
        num_regions=4,
        device=torch.device("cpu"),
    )

    assert region_ids.numel() == 32
    assert torch.bincount(region_ids, minlength=4).tolist() == [8, 8, 8, 8]
    assert region_ids[:8].tolist() == [0] * 8
    assert region_ids[8:16].tolist() == [1] * 8
    assert region_ids[16:24].tolist() == [2] * 8
    assert region_ids[24:].tolist() == [3] * 8


def test_pixel_probe_views_replace_only_one_region_with_processor_mean():
    pixels = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1
    views = build_pixel_probe_views(
        pixels,
        torch.tensor([[1, 2, 2]]),
        spatial_merge_size=1,
        num_regions=4,
    )

    reshaped = views.pixel_values.reshape(5, 4, 2)
    assert torch.equal(reshaped[0], pixels)
    for region in range(4):
        assert torch.equal(reshaped[region + 1, region], torch.zeros(2))
        kept = [index for index in range(4) if index != region]
        assert torch.equal(reshaped[region + 1, kept], pixels[kept])


def test_whole_image_ablation_preserves_grid_and_removes_all_content():
    pixels = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1
    grid = torch.tensor([[1, 2, 2]])
    views = build_whole_image_ablation_views(pixels, grid)

    assert views.num_views == 2
    assert views.patches_per_view == 4
    assert torch.equal(views.pixel_values[:4], pixels)
    assert torch.equal(views.pixel_values[4:], torch.zeros_like(pixels))
    assert torch.equal(views.image_grid_thw, grid.repeat((2, 1)))


def test_counterfactual_bank_stacks_full_mean_and_wrong_views():
    full = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 1
    wrong = full + 20
    grid = torch.tensor([[1, 2, 2]])

    views = build_counterfactual_bank_views(full, wrong, grid, grid.clone())

    assert views.num_views == 3
    assert views.patches_per_view == 4
    reshaped = views.pixel_values.reshape(3, 4, 2)
    assert torch.equal(reshaped[0], full)
    assert torch.equal(reshaped[1], torch.zeros_like(full))
    assert torch.equal(reshaped[2], wrong)
    assert torch.equal(views.image_grid_thw, grid.repeat((3, 1)))


def test_wrong_image_pairs_are_deterministic_distinct_and_category_matched():
    rows = [
        {"_fixed_index": 0, "image_id": "a", "category": "x"},
        {"_fixed_index": 1, "image_id": "b", "category": "x"},
        {"_fixed_index": 2, "image_id": "c", "category": "y"},
    ]

    first = build_matched_wrong_image_pairs(rows, None, seed=7, name="toy")
    second = build_matched_wrong_image_pairs(rows, None, seed=7, name="toy")

    assert first == second
    assert first[0]["source_position"] == 1
    assert first[1]["source_position"] == 0
    assert all(
        pair["target_image_identity"] != pair["source_image_identity"]
        for pair in first
    )
    assert first[2]["used_category_fallback"] is True
    assert [pair["source_reused_within_run"] for pair in first] == [
        False,
        False,
        True,
    ]


def test_wrong_image_pairing_rejects_duplicate_rows_of_the_same_image():
    rows = [
        {"_fixed_index": 0, "image_id": "shared", "category": "x"},
        {"_fixed_index": 1, "image_id": "shared", "category": "x"},
        {"_fixed_index": 2, "image_id": "other", "category": "x"},
    ]

    pairs = build_matched_wrong_image_pairs(rows, None, seed=5, name="duplicates")

    assert pairs[0]["source_position"] == 2
    assert pairs[1]["source_position"] == 2
    assert pairs[2]["source_position"] in (0, 1)


def test_wrong_image_pairing_finds_one_to_one_derangement_when_it_exists():
    rows = [
        {"_fixed_index": index, "image_id": str(index), "category": "x"}
        for index in range(6)
    ]

    pairs = build_matched_wrong_image_pairs(
        rows, None, seed=19, name="derangement"
    )

    assert len({pair["source_position"] for pair in pairs}) == len(rows)
    assert not any(pair["source_reused_within_run"] for pair in pairs)
    assert all(
        pair["target_image_identity"] != pair["source_image_identity"]
        for pair in pairs
    )


class _MetadataOnlyDataset:
    column_names = ["_fixed_index", "category", "image"]

    def __init__(self):
        self._columns = {
            "_fixed_index": [0, 1, 2],
            "category": ["x", "x", "y"],
            "image": ["must-not-decode"] * 3,
        }

    def __len__(self):
        return 3

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise AssertionError("pairing decoded a full source row")
        return self._columns[key]


def test_wrong_image_pairing_reads_columnar_metadata_without_image_decode():
    targets = [
        {"_fixed_index": 0, "category": "x"},
        {"_fixed_index": 2, "category": "y"},
    ]
    manifest = [
        {"source_index": 0, "category": "x"},
        {"source_index": 2, "category": "y"},
    ]

    pairs = build_matched_wrong_image_pairs(
        targets,
        manifest,
        seed=11,
        name="columnar",
        source_rows=_MetadataOnlyDataset(),
    )

    assert pairs[0]["source_position"] == 1
    assert pairs[0]["used_category_fallback"] is False
    assert pairs[1]["source_position"] in (0, 1)
    assert pairs[1]["used_category_fallback"] is True


class _FakeVisualModel:
    def __init__(self):
        self.config = SimpleNamespace(
            image_token_id=99,
            vision_config=SimpleNamespace(spatial_merge_size=1),
        )
        self.rope_deltas = torch.tensor([[123]])

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        logits_to_keep,
        **_kwargs,
    ):
        batch = input_ids.shape[0]
        patches_per_view = pixel_values.shape[0] // batch
        view_pixels = pixel_values.reshape(batch, patches_per_view, -1)
        strength = view_pixels.sum(dim=(1, 2))
        states = int(logits_to_keep.numel())
        logits = torch.zeros(batch, states, 8)
        logits[:, :, 2] = strength[:, None]
        logits[:, :, 3] = -strength[:, None]
        self.rope_deltas = torch.tensor([[999]])
        return SimpleNamespace(logits=logits)


def test_teacher_forced_probe_annotates_same_trace_and_restores_rope_state():
    model = _FakeVisualModel()
    old_rope = model.rope_deltas
    prompt = torch.tensor([[99, 7]])
    output = torch.tensor([[99, 7, 2, 3]])
    trace = [
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 0,
            "selective_target_token_id": 2,
        },
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 1,
            "selective_target_token_id": 3,
        },
    ]
    metadata = apply_teacher_forced_pixel_probes(
        model,
        {
            "input_ids": prompt,
            "attention_mask": torch.ones_like(prompt),
            "pixel_values": torch.ones(4, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        output,
        prompt_length=2,
        trace=trace,
        num_regions=4,
        top_k=2,
        view_batch_size=5,
    )

    assert metadata["protocol"] == PIXEL_PROBE_PROTOCOL
    assert metadata["included_in_decode_timing"] is False
    assert metadata["num_states"] == 2
    assert model.rope_deltas is old_rope
    assert all(row["visual_probe_same_text_trajectory"] for row in trace)
    assert all(row["visual_probe_recomputed_vision_encoder"] for row in trace)
    assert all(row["visual_probe_jsd"] >= 0.0 for row in trace)
    assert trace[0]["visual_probe_full_top1_matches_target"] is True
    assert trace[1]["visual_probe_full_top1_matches_target"] is False


def test_content_ablation_probe_annotates_same_trace_and_restores_rope_state():
    model = _FakeVisualModel()
    old_rope = model.rope_deltas
    prompt = torch.tensor([[99, 7]])
    output = torch.tensor([[99, 7, 2, 3]])
    trace = [
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 0,
            "selective_target_token_id": 2,
        },
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 1,
            "selective_target_token_id": 3,
        },
    ]
    metadata = apply_teacher_forced_content_ablation_probes(
        model,
        {
            "input_ids": prompt,
            "attention_mask": torch.ones_like(prompt),
            "pixel_values": torch.ones(4, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        output,
        prompt_length=2,
        trace=trace,
        top_k=2,
        view_batch_size=2,
    )

    assert metadata["protocol"] == CONTENT_ABLATION_PROBE_PROTOCOL
    assert metadata["num_views"] == 2
    assert metadata["span_tokens"] == 2
    assert metadata["num_scored_tokens"] == 3
    assert metadata["included_in_decode_timing"] is False
    assert model.rope_deltas is old_rope
    assert all(row["visual_probe_ablation_scope"] == "whole_image_content" for row in trace)
    assert all(row["visual_probe_same_text_trajectory"] for row in trace)
    assert all(row["visual_probe_recomputed_vision_encoder"] for row in trace)
    assert trace[0]["visual_probe_max_target_logprob_drop"] > 0.0
    assert trace[0]["visual_probe_span2_num_tokens"] == 2
    assert trace[0]["visual_probe_span2_uses_future_tokens"] is True
    assert trace[0]["visual_probe_span2_same_text_trajectory"] is True
    assert trace[0]["visual_probe_span2_mean_target_drop_fraction"] > 0.0
    assert trace[1]["visual_probe_span2_num_tokens"] == 1
    assert trace[1]["visual_probe_span2_uses_future_tokens"] is False


def test_counterfactual_bank_probe_records_separate_margin_drops():
    model = _FakeVisualModel()
    old_rope = model.rope_deltas
    prompt = torch.tensor([[99, 7]])
    output = torch.tensor([[99, 7, 2, 3]])
    trace = [
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 0,
            "selective_target_token_id": 2,
            "selective_u_root_candidate_token_ids": [3, 4],
            "selective_gc_root_candidate_token_ids": [2, 5],
        },
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 1,
            "selective_target_token_id": 3,
        },
    ]
    full_inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "pixel_values": torch.ones(4, 2),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    wrong_inputs = {
        **full_inputs,
        "pixel_values": torch.full((4, 2), 2.0),
    }
    pair = {
        "source_image_identity": "image:b",
        "used_category_fallback": False,
    }

    metadata = apply_teacher_forced_counterfactual_bank_probes(
        model,
        full_inputs,
        wrong_inputs,
        output,
        prompt_length=2,
        trace=trace,
        wrong_image_pair=pair,
        view_batch_size=3,
    )

    assert metadata["protocol"] == COUNTERFACTUAL_BANK_PROTOCOL
    assert metadata["num_views"] == 3
    assert metadata["counterfactuals"] == [
        "mean_content",
        "matched_wrong_image",
    ]
    assert metadata["num_requested_token_scores"] == 4
    assert metadata["num_scored_tokens"] == 2
    assert metadata["included_in_decode_timing"] is False
    assert model.rope_deltas is old_rope
    assert trace[0]["visual_probe_mean_target_margin_drop"] > 0.0
    assert trace[0]["visual_probe_wrong_target_margin_drop"] < 0.0
    assert (
        trace[0]["visual_probe_span2_mean_mean_target_margin_drop"]
        != trace[0]["visual_probe_span2_wrong_mean_target_margin_drop"]
    )
    assert trace[0]["visual_probe_wrong_image_source_identity"] == "image:b"
    assert trace[0]["visual_probe_candidate_alignment_available"] is True
    assert trace[0]["visual_probe_candidate_alignment_budget"] == 2
    assert trace[0]["visual_probe_candidate_alignment_uses_target_outcome"] is False
    assert len(
        [
            trace[0][f"visual_probe_u_candidate_logmass_{view}"]
            for view in ("full", "mean", "wrong")
        ]
    ) == 3
    assert trace[0]["visual_probe_gc_minus_u_candidate_visual_support"] == (
        trace[0]["visual_probe_gc_candidate_consensus_support"]
        - trace[0]["visual_probe_u_candidate_consensus_support"]
    )
    assert metadata["candidate_set_visual_alignment"] is True
    assert metadata["candidate_set_visual_alignment_uses_target_outcome"] is False


def test_counterfactual_bank_scores_the_generated_gc_context_tokens():
    model = _FakeVisualModel()
    prompt = torch.tensor([[99, 7]])
    output = torch.tensor([[99, 7, 2, 3, 2, 3]])
    trace = [
        {
            "selective_reuse_diagnostics": True,
            "selective_output_position": 3,
            "selective_target_token_id": 3,
        }
    ]
    full_inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "pixel_values": torch.ones(4, 2),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    wrong_inputs = {
        **full_inputs,
        "pixel_values": torch.full((4, 2), 2.0),
    }

    metadata = apply_teacher_forced_counterfactual_bank_probes(
        model,
        full_inputs,
        wrong_inputs,
        output,
        prompt_length=2,
        trace=trace,
        wrong_image_pair={
            "source_image_identity": "image:b",
            "used_category_fallback": False,
        },
        view_batch_size=3,
    )

    assert metadata["context_lookback_tokens"] == 3
    assert metadata["num_context_scored_tokens"] == 3
    assert trace[0]["visual_probe_context3_num_tokens"] == 3
    assert trace[0]["visual_probe_context3_target_token_ids"] == [2, 3, 2]
    assert len(
        trace[0]["visual_probe_context3_mean_target_margin_drops"]
    ) == 3
    assert len(trace[0]["visual_probe_context3_wrong_top1_changed"]) == 3
