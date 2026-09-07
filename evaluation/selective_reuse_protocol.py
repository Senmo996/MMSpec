"""Auditable protocol helpers for selective-reuse diagnostics.

This module deliberately separates two concerns from speculative decoding:

* the discovery/held-out assignment is a deterministic, image-cluster-level
  split that never depends on an observed model outcome; and
* visual sensitivity is measured by replacing spatial image patches *before*
  the vision encoder and then teacher-forcing the same generated text path.

The latter makes every counterfactual view share an identical textual prefix
while still recomputing the vision tower and all language-model states.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import List, Mapping, MutableMapping, Sequence, Tuple

import torch


PIXEL_PROBE_PROTOCOL = "teacher_forced_mean_patch_occlusion_v1"


def deterministic_cluster_split(
    cluster_keys: Sequence[str],
    *,
    discovery_fraction: float,
    seed: int,
    stratum: str,
) -> List[str]:
    """Assign whole clusters to discovery or held-out deterministically.

    The number of discovery clusters is fixed before any model output exists.
    Hash ranking, instead of input order, makes the assignment reproducible
    while keeping duplicated images in exactly one split.
    """

    if not 0.0 < float(discovery_fraction) < 1.0:
        raise ValueError("discovery_fraction must lie strictly between 0 and 1")
    normalized = [str(key) for key in cluster_keys]
    unique = sorted(set(normalized))
    if len(unique) < 2:
        raise ValueError("at least two image clusters are required for a split")

    def score(key: str) -> bytes:
        payload = f"{int(seed)}\0{stratum}\0{key}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    ranked = sorted(unique, key=lambda key: (score(key), key))
    discovery_count = int(round(len(ranked) * float(discovery_fraction)))
    discovery_count = min(max(discovery_count, 1), len(ranked) - 1)
    discovery = set(ranked[:discovery_count])
    return ["discovery" if key in discovery else "heldout" for key in normalized]


def _factor_grid(num_regions: int) -> Tuple[int, int]:
    if int(num_regions) <= 0:
        raise ValueError("num_regions must be positive")
    rows = int(math.sqrt(int(num_regions)))
    while rows > 1 and int(num_regions) % rows:
        rows -= 1
    return rows, int(num_regions) // rows


def raw_patch_region_ids(
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    num_regions: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Map Qwen's flattened pre-vision patches to spatial regions.

    Qwen's image processor flattens patches in
    ``(t, block_y, block_x, inner_y, inner_x)`` order.  We reconstruct that
    exact order here.  Unlike the legacy attention-mask diagnostic, this
    function has no approximate fallback: inconsistent metadata is an error.
    """

    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError("spatial_merge_size must be positive")
    rows, cols = _factor_grid(int(num_regions))
    grids = image_grid_thw.detach().to("cpu", dtype=torch.long).reshape(-1, 3)
    assignments: List[int] = []
    for temporal, height, width in grids.tolist():
        if temporal <= 0 or height <= 0 or width <= 0:
            raise ValueError("image_grid_thw contains a non-positive dimension")
        if height % merge or width % merge:
            raise ValueError("image grid is not divisible by spatial_merge_size")
        for _time in range(temporal):
            for block_y in range(height // merge):
                for block_x in range(width // merge):
                    for inner_y in range(merge):
                        y = block_y * merge + inner_y
                        region_y = min((y * rows) // height, rows - 1)
                        for inner_x in range(merge):
                            x = block_x * merge + inner_x
                            region_x = min((x * cols) // width, cols - 1)
                            assignments.append(region_y * cols + region_x)
    return torch.tensor(assignments, dtype=torch.long, device=device)


@dataclass(frozen=True)
class PixelProbeViews:
    """Full image plus one mean-patch-occluded view per spatial region."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    region_ids: torch.Tensor
    num_views: int
    num_regions: int
    patches_per_view: int


def build_pixel_probe_views(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    num_regions: int,
) -> PixelProbeViews:
    """Construct true pixel-patch counterfactuals for Qwen2.5-VL.

    ``pixel_values`` is already normalized by the processor, so zero-valued
    patch rows correspond to replacing those pixels by the processor mean.
    The returned tensor is flattened across views because Qwen's vision tower
    expects ``[total_patches, patch_dimension]``.
    """

    if pixel_values.ndim != 2:
        raise ValueError("pixel_values must have shape [patches, patch_dimension]")
    region_ids = raw_patch_region_ids(
        image_grid_thw,
        spatial_merge_size=spatial_merge_size,
        num_regions=num_regions,
        device=pixel_values.device,
    )
    patches = int(pixel_values.shape[0])
    if int(region_ids.numel()) != patches:
        raise ValueError(
            "image_grid_thw implies "
            f"{int(region_ids.numel())} patches, but pixel_values has {patches}"
        )
    views = [pixel_values]
    for region in range(int(num_regions)):
        masked = pixel_values.clone()
        masked[region_ids.eq(region)] = 0
        views.append(masked)
    num_views = int(num_regions) + 1
    return PixelProbeViews(
        pixel_values=torch.cat(views, dim=0),
        image_grid_thw=image_grid_thw.repeat((num_views, 1)),
        region_ids=region_ids,
        num_views=num_views,
        num_regions=int(num_regions),
        patches_per_view=patches,
    )


def _trace_positions(
    trace: Sequence[MutableMapping], prompt_length: int, sequence_length: int
) -> Tuple[List[MutableMapping], torch.Tensor]:
    selected = []
    positions = []
    for record in trace:
        if not record.get("selective_reuse_diagnostics"):
            continue
        offset = record.get("selective_output_position")
        if offset is None or record.get("selective_target_token_id") is None:
            continue
        # The hidden state immediately before generated token ``offset``
        # predicts that token under causal teacher forcing.
        position = int(prompt_length) + int(offset) - 1
        if position < 0 or position >= int(sequence_length):
            raise ValueError(
                f"invalid teacher-forced logit position {position} for "
                f"sequence length {sequence_length}"
            )
        selected.append(record)
        positions.append(position)
    return selected, torch.tensor(positions, dtype=torch.long)


def _multiview_metrics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int,
) -> Mapping[str, torch.Tensor | List[int]]:
    """Compute per-state visual metrics from ``[views, states, vocab]`` logits."""

    if logits.ndim != 3 or int(logits.shape[0]) < 2:
        raise ValueError("logits must have shape [views>=2, states, vocab]")
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    mean_log_prob = torch.logsumexp(log_probs, dim=0) - math.log(
        int(logits.shape[0])
    )
    jsd = (
        log_probs.exp() * (log_probs - mean_log_prob.unsqueeze(0))
    ).sum(dim=-1).mean(dim=0).clamp_min(0.0)
    top1 = torch.argmax(logits, dim=-1)
    disagreement = top1[1:].ne(top1[:1]).float().mean(dim=0)

    k = min(max(int(top_k), 1), int(logits.shape[-1]))
    top_ids = torch.topk(logits, k=k, dim=-1).indices.detach().cpu()
    union_sizes = [
        len(set(top_ids[:, state_index, :].reshape(-1).tolist()))
        for state_index in range(int(logits.shape[1]))
    ]

    state_indices = torch.arange(
        int(logits.shape[1]), device=logits.device, dtype=torch.long
    )
    target_ids = target_ids.to(device=logits.device, dtype=torch.long)
    target_log_probs = log_probs[:, state_indices, target_ids]
    target_drop = (
        target_log_probs[0] - target_log_probs[1:].amin(dim=0)
    ).clamp_min(0.0)
    return {
        "jsd": jsd,
        "top1_disagreement": disagreement,
        "full_top1": top1[0],
        "topk_union_sizes": union_sizes,
        "full_target_logprob": target_log_probs[0],
        "max_target_logprob_drop": target_drop,
    }


def _selected_qwen_logits(
    base_model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    logit_positions: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen and project only the requested hidden-state rows.

    The repository carries a KV-enabled Qwen fork whose top-level ``forward``
    predates ``logits_to_keep``.  Calling the vision tower and decoder modules
    directly keeps this diagnostic compatible with both that fork and recent
    Transformers, and avoids materializing full-sequence vocabulary logits.
    """

    required = ("get_input_embeddings", "visual", "model", "lm_head")
    if not all(hasattr(base_model, name) for name in required):
        # Lightweight test doubles and future compatible models can use the
        # public optimized interface.
        output = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
            logits_to_keep=logit_positions,
        )
        return output.logits

    inputs_embeds = base_model.get_input_embeddings()(input_ids)
    image_embeds = base_model.visual(
        pixel_values.to(dtype=base_model.visual.dtype),
        grid_thw=image_grid_thw,
    )
    image_mask = input_ids.eq(base_model.config.image_token_id)
    if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
        raise ValueError(
            "Image features and image tokens do not match in teacher-forced "
            f"probe: tokens={int(image_mask.sum().item())}, "
            f"features={int(image_embeds.shape[0])}"
        )
    image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask, image_embeds.to(inputs_embeds.dtype)
    )
    position_ids, _rope_deltas = base_model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    is_project_kv_fork = base_model.__class__.__module__.startswith("method.")
    probe_cache = None
    probe_cache_storage = None
    if is_project_kv_fork:
        if int(input_ids.shape[0]) != 1:
            raise ValueError(
                "the project's Qwen KV cache supports one visual probe view "
                "per forward; set --visual-probe-batch-size 1"
            )
        from method.vispec.kv_cache import initialize_past_key_values

        probe_cache, probe_cache_storage, _probe_cache_lengths = (
            initialize_past_key_values(base_model)
        )
    output = base_model.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=probe_cache,
        inputs_embeds=inputs_embeds,
        use_cache=is_project_kv_fork,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    selected_hidden = output[0].index_select(
        1, logit_positions.to(output[0].device)
    )
    logits = base_model.lm_head(selected_hidden)
    del (
        output,
        selected_hidden,
        inputs_embeds,
        image_embeds,
        probe_cache,
        probe_cache_storage,
    )
    return logits


@torch.inference_mode()
def apply_teacher_forced_pixel_probes(
    base_model,
    model_inputs: Mapping[str, torch.Tensor],
    output_ids: torch.Tensor,
    *,
    prompt_length: int,
    trace: Sequence[MutableMapping],
    num_regions: int = 4,
    top_k: int = 4,
    view_batch_size: int = 1,
) -> dict:
    """Annotate a trace using true visual counterfactuals on one text path.

    Diagnostic forwards happen after timed decoding.  They must therefore not
    be interpreted as part of the method's throughput measurement.
    """

    started = time.perf_counter()
    if "pixel_values" not in model_inputs or "image_grid_thw" not in model_inputs:
        raise ValueError("teacher-forced visual probes require image inputs")
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
            "protocol": PIXEL_PROBE_PROTOCOL,
            "num_states": 0,
            "num_views": int(num_regions) + 1,
            "elapsed_seconds": time.perf_counter() - started,
            "included_in_decode_timing": False,
        }

    spatial_merge_size = int(
        base_model.config.vision_config.spatial_merge_size
    )
    views = build_pixel_probe_views(
        model_inputs["pixel_values"],
        model_inputs["image_grid_thw"],
        spatial_merge_size=spatial_merge_size,
        num_regions=int(num_regions),
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
                [
                    attention_mask,
                    torch.ones(
                        (1, generated_count),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

    logits_to_keep = logit_positions.to(sequence.device)
    logits_by_view = []
    previous_rope_deltas = getattr(base_model, "rope_deltas", None)
    try:
        for view_start in range(0, views.num_views, int(view_batch_size)):
            view_stop = min(
                view_start + int(view_batch_size), views.num_views
            )
            batch_views = view_stop - view_start
            patch_start = view_start * views.patches_per_view
            patch_stop = view_stop * views.patches_per_view
            grid_start = view_start * int(model_inputs["image_grid_thw"].shape[0])
            grid_stop = view_stop * int(model_inputs["image_grid_thw"].shape[0])
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
        # The Hugging Face Qwen implementation caches mRoPE deltas on the
        # module.  A diagnostic forward must not perturb later decoding.
        base_model.rope_deltas = previous_rope_deltas

    logits = torch.cat(logits_by_view, dim=0)
    target_ids = torch.tensor(
        [int(record["selective_target_token_id"]) for record in selected],
        dtype=torch.long,
        device=logits.device,
    )
    metrics = _multiview_metrics(logits, target_ids, top_k=int(top_k))
    for state_index, record in enumerate(selected):
        record.update(
            {
                "visual_probe_active": True,
                "visual_probe_protocol": PIXEL_PROBE_PROTOCOL,
                "visual_probe_jsd": float(metrics["jsd"][state_index].item()),
                "visual_probe_top1_disagreement_rate": float(
                    metrics["top1_disagreement"][state_index].item()
                ),
                "visual_probe_full_top1_token_id": int(
                    metrics["full_top1"][state_index].item()
                ),
                "visual_probe_full_top1_matches_target": bool(
                    int(metrics["full_top1"][state_index].item())
                    == int(record["selective_target_token_id"])
                ),
                "visual_probe_topk_union_size": int(
                    metrics["topk_union_sizes"][state_index]
                ),
                "visual_probe_max_target_logprob_drop": float(
                    metrics["max_target_logprob_drop"][state_index].item()
                ),
                "visual_probe_full_target_logprob": float(
                    metrics["full_target_logprob"][state_index].item()
                ),
                "visual_probe_num_regions": int(num_regions),
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
        "protocol": PIXEL_PROBE_PROTOCOL,
        "num_states": len(selected),
        "num_views": views.num_views,
        "num_raw_patches": views.patches_per_view,
        "elapsed_seconds": elapsed,
        "included_in_decode_timing": False,
    }
