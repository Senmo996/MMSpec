"""Teacher-forced whole-image content ablation for selective-reuse analysis.

The existing spatial probe removes one image region at a time.  That is useful
for localized evidence, but it can underestimate visual reliance when evidence
is redundant across regions or depends on the whole scene.  This module keeps
the image grid, image-token count, and generated text trajectory fixed while
replacing every normalized pixel patch by the processor mean (zero after
normalization) before rerunning the vision encoder.

The probe is diagnostic only and must be run outside decode timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, MutableMapping, Sequence

import torch

from evaluation.selective_reuse_protocol import (
    _multiview_metrics,
    _selected_qwen_logits,
    _trace_positions,
)


CONTENT_ABLATION_PROBE_PROTOCOL = "teacher_forced_whole_image_mean_ablation_v1"
CONTENT_ABLATION_SPAN_TOKENS = 2


@dataclass(frozen=True)
class WholeImageAblationViews:
    """Full image followed by an equal-shape mean-image counterfactual."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    num_views: int
    patches_per_view: int


def build_whole_image_ablation_views(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> WholeImageAblationViews:
    """Return full and all-mean views without changing visual geometry."""

    if pixel_values.ndim != 2:
        raise ValueError("pixel_values must have shape [patches, patch_dimension]")
    if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[1]) != 3:
        raise ValueError("image_grid_thw must have shape [images, 3]")
    patches = int(pixel_values.shape[0])
    implied_patches = int(image_grid_thw.to(dtype=torch.long).prod(dim=1).sum())
    if implied_patches != patches:
        raise ValueError(
            "image_grid_thw implies "
            f"{implied_patches} patches, but pixel_values has {patches}"
        )
    return WholeImageAblationViews(
        pixel_values=torch.cat((pixel_values, torch.zeros_like(pixel_values)), dim=0),
        image_grid_thw=image_grid_thw.repeat((2, 1)),
        num_views=2,
        patches_per_view=patches,
    )


def _expanded_token_span_layout(
    selected: Sequence[MutableMapping],
    base_positions: torch.Tensor,
    sequence: torch.Tensor,
    *,
    span_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Expand each proposal root to exact adjacent generated-token positions."""

    if int(span_tokens) <= 0:
        raise ValueError("span_tokens must be positive")
    if len(selected) != int(base_positions.numel()):
        raise ValueError("selected records and base positions must have equal size")
    sequence_length = int(sequence.shape[1])
    positions = []
    target_ids = []
    spans = []
    for record, base_position_tensor in zip(selected, base_positions):
        base_position = int(base_position_tensor.item())
        available = sequence_length - (base_position + 1)
        count = min(int(span_tokens), available)
        if count <= 0:
            raise ValueError(
                "proposal state has no teacher-forced target token at "
                f"logit position {base_position}"
            )
        start = len(positions)
        for offset in range(count):
            logit_position = base_position + offset
            positions.append(logit_position)
            target_ids.append(int(sequence[0, logit_position + 1].item()))
        stop = len(positions)
        if target_ids[start] != int(record["selective_target_token_id"]):
            raise ValueError(
                "trace target does not match the fixed teacher-forced trajectory"
            )
        spans.append((start, stop))
    return (
        torch.tensor(positions, dtype=torch.long, device=sequence.device),
        torch.tensor(target_ids, dtype=torch.long, device=sequence.device),
        spans,
    )


@torch.inference_mode()
def apply_teacher_forced_content_ablation_probes(
    base_model,
    model_inputs: Mapping[str, torch.Tensor],
    output_ids: torch.Tensor,
    *,
    prompt_length: int,
    trace: Sequence[MutableMapping],
    top_k: int = 4,
    view_batch_size: int = 1,
    span_tokens: int = CONTENT_ABLATION_SPAN_TOKENS,
) -> dict:
    """Annotate trace states with a full-image versus mean-image contrast."""

    started = time.perf_counter()
    if "pixel_values" not in model_inputs or "image_grid_thw" not in model_inputs:
        raise ValueError("content-ablation probes require image inputs")
    if int(view_batch_size) <= 0:
        raise ValueError("view_batch_size must be positive")
    if output_ids.ndim != 2 or int(output_ids.shape[0]) != 1:
        raise ValueError("output_ids must have shape [1, sequence]")

    sequence = output_ids[:, :]
    selected, logit_positions = _trace_positions(
        trace, int(prompt_length), int(sequence.shape[1])
    )
    if not selected:
        return {
            "protocol": CONTENT_ABLATION_PROBE_PROTOCOL,
            "num_states": 0,
            "num_views": 2,
            "elapsed_seconds": time.perf_counter() - started,
            "included_in_decode_timing": False,
        }

    views = build_whole_image_ablation_views(
        model_inputs["pixel_values"], model_inputs["image_grid_thw"]
    )
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(sequence)
    else:
        generated_count = int(sequence.shape[1]) - int(attention_mask.shape[1])
        if generated_count < 0:
            raise ValueError("output sequence is shorter than the prompt mask")
        if generated_count:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, generated_count),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )

    logits_to_keep, target_ids, token_spans = _expanded_token_span_layout(
        selected,
        logit_positions,
        sequence,
        span_tokens=int(span_tokens),
    )
    logits_by_view = []
    grids_per_view = int(model_inputs["image_grid_thw"].shape[0])
    previous_rope_deltas = getattr(base_model, "rope_deltas", None)
    try:
        for view_start in range(0, views.num_views, int(view_batch_size)):
            view_stop = min(view_start + int(view_batch_size), views.num_views)
            batch_views = view_stop - view_start
            patch_start = view_start * views.patches_per_view
            patch_stop = view_stop * views.patches_per_view
            grid_start = view_start * grids_per_view
            grid_stop = view_stop * grids_per_view
            selected_logits = _selected_qwen_logits(
                base_model,
                input_ids=sequence.repeat((batch_views, 1)),
                attention_mask=attention_mask.repeat((batch_views, 1)),
                pixel_values=views.pixel_values[patch_start:patch_stop],
                image_grid_thw=views.image_grid_thw[grid_start:grid_stop],
                logit_positions=logits_to_keep,
            )
            logits_by_view.append(selected_logits.detach())
            del selected_logits
    finally:
        base_model.rope_deltas = previous_rope_deltas

    logits = torch.cat(logits_by_view, dim=0)
    metrics = _multiview_metrics(logits, target_ids, top_k=int(top_k))
    token_drop_fraction = metrics["max_target_logprob_drop"] / (
        metrics["max_target_logprob_drop"]
        + (-metrics["full_target_logprob"]).clamp_min(0.0)
        + 1e-8
    )
    for record, (span_start, span_stop) in zip(selected, token_spans):
        full_top1 = int(metrics["full_top1"][span_start].item())
        record.update(
            {
                "visual_probe_active": True,
                "visual_probe_protocol": CONTENT_ABLATION_PROBE_PROTOCOL,
                "visual_probe_ablation_scope": "whole_image_content",
                "visual_probe_jsd": float(metrics["jsd"][span_start].item()),
                "visual_probe_top1_disagreement_rate": float(
                    metrics["top1_disagreement"][span_start].item()
                ),
                "visual_probe_full_top1_token_id": full_top1,
                "visual_probe_full_top1_matches_target": bool(
                    full_top1 == int(record["selective_target_token_id"])
                ),
                "visual_probe_topk_union_size": int(
                    metrics["topk_union_sizes"][span_start]
                ),
                "visual_probe_max_target_logprob_drop": float(
                    metrics["max_target_logprob_drop"][span_start].item()
                ),
                "visual_probe_full_target_logprob": float(
                    metrics["full_target_logprob"][span_start].item()
                ),
                "visual_probe_span2_num_tokens": int(span_stop - span_start),
                "visual_probe_span2_mean_jsd": float(
                    metrics["jsd"][span_start:span_stop].mean().item()
                ),
                "visual_probe_span2_mean_target_logprob_drop": float(
                    metrics["max_target_logprob_drop"][span_start:span_stop]
                    .mean()
                    .item()
                ),
                "visual_probe_span2_mean_target_drop_fraction": float(
                    token_drop_fraction[span_start:span_stop].mean().item()
                ),
                "visual_probe_span2_same_text_trajectory": True,
                "visual_probe_span2_uses_future_tokens": bool(
                    span_stop - span_start > 1
                ),
                "visual_probe_num_regions": 1,
                "visual_probe_num_visual_tokens": int(
                    sequence.eq(base_model.config.image_token_id).sum().item()
                ),
                "visual_probe_num_raw_patches": int(views.patches_per_view),
                "visual_probe_used_grid_metadata": True,
                "visual_probe_same_text_trajectory": True,
                "visual_probe_recomputed_vision_encoder": True,
            }
        )

    elapsed = time.perf_counter() - started
    del logits, logits_by_view
    return {
        "protocol": CONTENT_ABLATION_PROBE_PROTOCOL,
        "num_states": len(selected),
        "num_scored_tokens": int(logits_to_keep.numel()),
        "span_tokens": int(span_tokens),
        "num_views": views.num_views,
        "num_raw_patches": views.patches_per_view,
        "elapsed_seconds": elapsed,
        "included_in_decode_timing": False,
    }
