"""Teacher-forced visual probe with two complementary counterfactuals.

The original image is compared with (1) an equal-shape mean-content image and
(2) a distinct, preferably same-category benchmark image resized to the target
image.  All views use the exact same text trajectory.  The paired perturbations
separate visual reliance that is robust across interventions from artifacts of
one out-of-distribution null image.

This probe is diagnostic only and runs outside decode timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Mapping, MutableMapping, Sequence

from PIL import Image
import torch

from evaluation.selective_reuse_content_ablation_protocol import (
    _expanded_token_span_layout,
)
from evaluation.selective_reuse_protocol import (
    _selected_qwen_logits,
    _trace_positions,
)


COUNTERFACTUAL_BANK_PROTOCOL = (
    "teacher_forced_mean_and_matched_wrong_image_v1"
)
COUNTERFACTUAL_BANK_SPAN_TOKENS = 2
COUNTERFACTUAL_CONTEXT_TOKENS = 3
PAIRING_METADATA_KEYS = (
    "source_index",
    "_fixed_index",
    "image_id",
    "image_path",
    "imgname",
    "id",
    "question_id",
    "category",
    "type",
    "topic",
    "dataset_name",
)


@dataclass(frozen=True)
class CounterfactualBankViews:
    """Full, mean-content, and matched-wrong-image pixel views."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    num_views: int
    patches_per_view: int


def _category(row: Mapping, manifest_row: Mapping | None = None) -> str:
    manifest_row = manifest_row or {}
    for key in ("category", "type", "topic", "dataset_name"):
        value = manifest_row.get(key, row.get(key))
        if value not in (None, "", "default", "unknown"):
            return str(value)
    return "default"


def _image_identity(
    row: Mapping,
    manifest_row: Mapping | None = None,
    *,
    allow_raw_image: bool = True,
) -> str:
    manifest_row = manifest_row or {}
    for key in ("image_id", "image_path", "imgname"):
        value = manifest_row.get(key, row.get(key))
        if value not in (None, ""):
            return f"{key}:{value}"
    if allow_raw_image:
        value = row.get("image")
        if isinstance(value, str):
            return "image:" + hashlib.sha1(value.encode("utf-8")).hexdigest()
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return "bytes:" + hashlib.sha1(value["bytes"]).hexdigest()
            if value.get("path"):
                return f"path:{value['path']}"
    for key in ("_fixed_index", "source_index", "id", "question_id"):
        value = manifest_row.get(key, row.get(key))
        if value not in (None, ""):
            return f"{key}:{value}"
    raise ValueError("cannot derive a deterministic image identity")


def _pairing_metadata_rows(rows: Sequence[Mapping]) -> Sequence[Mapping]:
    """Read dataset metadata columns without decoding every image payload."""

    column_names = getattr(rows, "column_names", None)
    if column_names is None:
        return rows
    keys = [key for key in PAIRING_METADATA_KEYS if key in column_names]
    if not keys:
        return rows
    try:
        columns = {key: rows[key] for key in keys}
    except (TypeError, KeyError, IndexError):
        # The repository's lightweight JsonListDataset advertises
        # ``column_names`` but intentionally supports integer indexing only.
        return rows
    return [
        {key: values[index] for key, values in columns.items()}
        for index in range(len(rows))
    ]


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def build_matched_wrong_image_pairs(
    rows: Sequence[Mapping],
    manifest: Sequence[Mapping] | None,
    *,
    seed: int,
    name: str,
    source_rows: Sequence[Mapping] | None = None,
    source_manifest: Sequence[Mapping] | None = None,
) -> list[dict]:
    """Pair every target with a distinct same-category image when possible."""

    source_rows = rows if source_rows is None else source_rows
    source_manifest = (
        manifest
        if source_manifest is None and source_rows is rows
        else source_manifest
    )
    if len(source_rows) < 2:
        raise ValueError(f"{name} needs at least two rows for wrong-image pairing")
    if manifest is not None and len(manifest) != len(rows):
        raise ValueError("manifest and rows must have equal length")
    if source_manifest is not None and len(source_manifest) != len(source_rows):
        raise ValueError("source manifest and source rows must have equal length")
    rng = random.Random(_stable_seed(seed, name))
    source_metadata = _pairing_metadata_rows(source_rows)
    allow_raw_image_identity = source_metadata is source_rows
    target_categories = [
        _category(row, manifest[index] if manifest is not None else None)
        for index, row in enumerate(rows)
    ]
    target_identities = [
        _image_identity(
            row,
            manifest[index] if manifest is not None else None,
            allow_raw_image=allow_raw_image_identity,
        )
        for index, row in enumerate(rows)
    ]
    source_categories = [
        _category(
            row,
            source_manifest[index] if source_manifest is not None else None,
        )
        for index, row in enumerate(source_metadata)
    ]
    source_identities = [
        _image_identity(
            row,
            source_manifest[index] if source_manifest is not None else None,
            allow_raw_image=allow_raw_image_identity,
        )
        for index, row in enumerate(source_metadata)
    ]
    candidate_lists: list[list[int]] = []
    category_fallback_flags: list[bool] = []
    for target in range(len(rows)):
        candidates = [
            index
            for index in range(len(source_rows))
            if source_identities[index] != target_identities[target]
            and target_categories[target] != "default"
            and source_categories[index] == target_categories[target]
        ]
        used_category_fallback = False
        if not candidates:
            used_category_fallback = True
            candidates = [
                index
                for index in range(len(source_rows))
                if source_identities[index] != target_identities[target]
            ]
        if not candidates:
            raise ValueError(f"{name}:{target} has no distinct source image")
        rng.shuffle(candidates)
        candidate_lists.append(candidates)
        category_fallback_flags.append(used_category_fallback)

    # Maximum bipartite matching prevents accidental control reuse when a
    # valid one-to-one assignment exists (a greedy derangement can get stuck
    # on its final target even with enough source images).
    source_to_target: dict[int, int] = {}
    target_to_source: dict[int, int] = {}

    def assign_unique(target: int, visited: set[int]) -> bool:
        for source in candidate_lists[target]:
            if source in visited:
                continue
            visited.add(source)
            previous_target = source_to_target.get(source)
            if previous_target is None or assign_unique(previous_target, visited):
                source_to_target[source] = target
                target_to_source[target] = source
                return True
        return False

    for target in sorted(
        range(len(rows)), key=lambda index: (len(candidate_lists[index]), index)
    ):
        assign_unique(target, set())

    pairs = []
    for target in range(len(rows)):
        source_reused = target not in target_to_source
        source = target_to_source.get(target)
        if source is None:
            candidates = candidate_lists[target]
            source = candidates[rng.randrange(len(candidates))]
        pairs.append(
            {
                "target_position": int(target),
                "source_position": int(source),
                "category": target_categories[target],
                "target_image_identity": target_identities[target],
                "source_image_identity": source_identities[source],
                "used_category_fallback": bool(
                    category_fallback_flags[target]
                ),
                "source_reused_within_run": bool(source_reused),
            }
        )
    return pairs


def resize_wrong_image(wrong_image, target_image):
    """Match target geometry before processor-side resizing and tokenization."""

    if not isinstance(wrong_image, Image.Image):
        return wrong_image
    wrong_image = wrong_image.convert("RGB")
    if isinstance(target_image, Image.Image) and wrong_image.size != target_image.size:
        wrong_image = wrong_image.resize(
            target_image.size, resample=Image.Resampling.BICUBIC
        )
    return wrong_image


def build_counterfactual_bank_views(
    full_pixel_values: torch.Tensor,
    wrong_pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    wrong_image_grid_thw: torch.Tensor,
) -> CounterfactualBankViews:
    """Stack full, mean, and wrong views while preserving visual geometry."""

    if full_pixel_values.ndim != 2 or wrong_pixel_values.ndim != 2:
        raise ValueError("pixel values must have shape [patches, patch_dimension]")
    if full_pixel_values.shape != wrong_pixel_values.shape:
        raise ValueError("full and wrong images must produce equal patch tensors")
    if image_grid_thw.ndim != 2 or int(image_grid_thw.shape[1]) != 3:
        raise ValueError("image_grid_thw must have shape [images, 3]")
    if not torch.equal(image_grid_thw, wrong_image_grid_thw):
        raise ValueError("full and wrong images must have identical grid metadata")
    patches = int(full_pixel_values.shape[0])
    implied_patches = int(image_grid_thw.to(dtype=torch.long).prod(dim=1).sum())
    if implied_patches != patches:
        raise ValueError(
            "image_grid_thw implies "
            f"{implied_patches} patches, but pixel_values has {patches}"
        )
    return CounterfactualBankViews(
        pixel_values=torch.cat(
            (
                full_pixel_values,
                torch.zeros_like(full_pixel_values),
                wrong_pixel_values.to(
                    device=full_pixel_values.device,
                    dtype=full_pixel_values.dtype,
                ),
            ),
            dim=0,
        ),
        image_grid_thw=image_grid_thw.repeat((3, 1)),
        num_views=3,
        patches_per_view=patches,
    )


def _pairwise_visual_metrics(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return full-vs-each-view JSD, target-logprob, and target-margin loss."""

    if logits.ndim != 3 or int(logits.shape[0]) != 3:
        raise ValueError("counterfactual bank logits must be [3, states, vocab]")
    float_logits = logits.float()
    log_probs = torch.log_softmax(float_logits, dim=-1)
    state_indices = torch.arange(
        int(logits.shape[1]), device=logits.device, dtype=torch.long
    )
    target_ids = target_ids.to(device=logits.device, dtype=torch.long)
    target_log_probs = log_probs[:, state_indices, target_ids]
    target_logits = float_logits[:, state_indices, target_ids]
    top2_values, top2_ids = torch.topk(float_logits, k=2, dim=-1)
    best_other = torch.where(
        top2_ids[..., 0].eq(target_ids.unsqueeze(0)),
        top2_values[..., 1],
        top2_values[..., 0],
    )
    margins = target_logits - best_other

    pairwise_jsd = []
    for view_index in (1, 2):
        pair = log_probs[[0, view_index]]
        mean_log_prob = torch.logsumexp(pair, dim=0) - math.log(2.0)
        pairwise_jsd.append(
            (
                pair.exp() * (pair - mean_log_prob.unsqueeze(0))
            ).sum(dim=-1).mean(dim=0).clamp_min(0.0)
        )
    top1 = torch.argmax(float_logits, dim=-1)
    return {
        "full_target_logprob": target_log_probs[0],
        "target_logprob_drop": target_log_probs[0:1] - target_log_probs[1:],
        "target_margin_drop": margins[0:1] - margins[1:],
        "pairwise_jsd": torch.stack(pairwise_jsd, dim=0),
        "full_top1": top1[0],
        "top1_changed": top1[1:].ne(top1[0:1]),
        # Retain only the compact partition term.  Candidate-set masses can
        # then be gathered from the already-computed logits without keeping a
        # second full-vocabulary log-probability tensor alive.
        "log_partition": torch.logsumexp(float_logits, dim=-1),
    }


def _candidate_set_log_mass(
    logits: torch.Tensor,
    log_partition: torch.Tensor,
    *,
    state_index: int,
    candidate_ids: Sequence[int],
) -> torch.Tensor:
    """Return log probability mass of one fixed candidate set in each view."""

    if logits.ndim != 3:
        raise ValueError("candidate logits must have shape [views, states, vocab]")
    if log_partition.shape != logits.shape[:2]:
        raise ValueError("log_partition must have shape [views, states]")
    if not 0 <= int(state_index) < int(logits.shape[1]):
        raise ValueError("candidate state index is out of range")
    ids = [int(token_id) for token_id in candidate_ids]
    if not ids:
        raise ValueError("candidate set must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate set must contain unique token IDs")
    if min(ids) < 0 or max(ids) >= int(logits.shape[2]):
        raise ValueError("candidate token ID is outside the model vocabulary")
    index = torch.tensor(ids, dtype=torch.long, device=logits.device)
    selected = logits[:, int(state_index), :].index_select(-1, index).float()
    return torch.logsumexp(selected, dim=-1) - log_partition[:, int(state_index)]


def _candidate_alignment_metrics(
    logits: torch.Tensor,
    log_partition: torch.Tensor,
    *,
    state_index: int,
    u_candidate_ids: Sequence[int],
    gc_candidate_ids: Sequence[int],
) -> dict[str, object]:
    """Score source-relative visual evidence without consulting outcomes."""

    budget = min(len(u_candidate_ids), len(gc_candidate_ids), 8)
    if budget <= 0:
        return {
            "available": False,
            "budget": 0,
            "invalid_reason": "one_or_both_candidate_sets_empty",
        }
    u_ids = [int(token) for token in u_candidate_ids[:budget]]
    gc_ids = [int(token) for token in gc_candidate_ids[:budget]]
    if len(set(u_ids)) != budget or len(set(gc_ids)) != budget:
        return {
            "available": False,
            "budget": int(budget),
            "invalid_reason": "duplicate_candidate_id",
        }
    try:
        u_mass = _candidate_set_log_mass(
            logits,
            log_partition,
            state_index=state_index,
            candidate_ids=u_ids,
        )
        gc_mass = _candidate_set_log_mass(
            logits,
            log_partition,
            state_index=state_index,
            candidate_ids=gc_ids,
        )
    except ValueError as error:
        return {
            "available": False,
            "budget": int(budget),
            "invalid_reason": str(error),
        }

    # Views are fixed as true image, mean-content image, matched wrong image.
    u_drops = u_mass[0:1] - u_mass[1:]
    gc_drops = gc_mass[0:1] - gc_mass[1:]
    u_support = float(u_drops.min().item())
    gc_support = float(gc_drops.min().item())
    return {
        "available": True,
        "budget": int(budget),
        "invalid_reason": None,
        "u_token_ids": u_ids,
        "gc_token_ids": gc_ids,
        "u_log_mass": [float(value) for value in u_mass.tolist()],
        "gc_log_mass": [float(value) for value in gc_mass.tolist()],
        "u_counterfactual_drops": [float(value) for value in u_drops.tolist()],
        "gc_counterfactual_drops": [float(value) for value in gc_drops.tolist()],
        "u_consensus_support": u_support,
        "gc_consensus_support": gc_support,
        "gc_minus_u_consensus_support": gc_support - u_support,
    }


def _expanded_context_layout(
    selected: Sequence[MutableMapping],
    base_positions: torch.Tensor,
    sequence: torch.Tensor,
    *,
    context_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Select logits that originally predicted each generated context token."""

    if int(context_tokens) <= 0:
        raise ValueError("context_tokens must be positive")
    if len(selected) != int(base_positions.numel()):
        raise ValueError("selected records and base positions must have equal size")
    positions: list[int] = []
    target_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for record, base_position_tensor in zip(selected, base_positions):
        base_position = int(base_position_tensor.item())
        output_position = int(record["selective_output_position"])
        count = min(max(output_position, 0), int(context_tokens))
        start = len(positions)
        for logit_position in range(base_position - count, base_position):
            if logit_position < 0 or logit_position + 1 >= int(sequence.shape[1]):
                raise ValueError("invalid generated-context logit position")
            positions.append(logit_position)
            target_ids.append(int(sequence[0, logit_position + 1].item()))
        spans.append((start, len(positions)))
    return (
        torch.tensor(positions, dtype=torch.long, device=sequence.device),
        torch.tensor(target_ids, dtype=torch.long, device=sequence.device),
        spans,
    )


@torch.inference_mode()
def apply_teacher_forced_counterfactual_bank_probes(
    base_model,
    model_inputs: Mapping[str, torch.Tensor],
    wrong_model_inputs: Mapping[str, torch.Tensor],
    output_ids: torch.Tensor,
    *,
    prompt_length: int,
    trace: Sequence[MutableMapping],
    wrong_image_pair: Mapping,
    view_batch_size: int = 1,
    span_tokens: int = COUNTERFACTUAL_BANK_SPAN_TOKENS,
    context_tokens: int = COUNTERFACTUAL_CONTEXT_TOKENS,
) -> dict:
    """Annotate trace states with mean-image and matched-wrong-image contrasts."""

    started = time.perf_counter()
    required = ("input_ids", "pixel_values", "image_grid_thw")
    for key in required:
        if key not in model_inputs or key not in wrong_model_inputs:
            raise ValueError(f"counterfactual bank requires {key}")
    if int(view_batch_size) <= 0:
        raise ValueError("view_batch_size must be positive")
    if output_ids.ndim != 2 or int(output_ids.shape[0]) != 1:
        raise ValueError("output_ids must have shape [1, sequence]")
    if not torch.equal(model_inputs["input_ids"], wrong_model_inputs["input_ids"]):
        raise ValueError("wrong-image processing changed the text token sequence")

    sequence = output_ids[:, :]
    selected, base_positions = _trace_positions(
        trace, int(prompt_length), int(sequence.shape[1])
    )
    if not selected:
        return {
            "protocol": COUNTERFACTUAL_BANK_PROTOCOL,
            "num_states": 0,
            "num_views": 3,
            "elapsed_seconds": time.perf_counter() - started,
            "included_in_decode_timing": False,
            "wrong_image_pair": dict(wrong_image_pair),
        }

    views = build_counterfactual_bank_views(
        model_inputs["pixel_values"],
        wrong_model_inputs["pixel_values"],
        model_inputs["image_grid_thw"],
        wrong_model_inputs["image_grid_thw"],
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

    future_positions, future_target_ids, token_spans = _expanded_token_span_layout(
        selected,
        base_positions,
        sequence,
        span_tokens=int(span_tokens),
    )
    context_positions, context_target_ids, context_spans = _expanded_context_layout(
        selected,
        base_positions,
        sequence,
        context_tokens=int(context_tokens),
    )
    requested_positions = torch.cat((future_positions, context_positions), dim=0)
    logits_to_keep, inverse_positions = torch.unique(
        requested_positions, sorted=True, return_inverse=True
    )
    target_ids = sequence[0].index_select(0, logits_to_keep + 1)
    token_indices = [
        inverse_positions[start:stop] for start, stop in token_spans
    ]
    context_index_offset = int(future_positions.numel())
    context_indices = [
        inverse_positions[
            context_index_offset + start : context_index_offset + stop
        ]
        for start, stop in context_spans
    ]
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
    metrics = _pairwise_visual_metrics(logits, target_ids)
    positive_logprob_drop = metrics["target_logprob_drop"].clamp_min(0.0)
    max_logprob_drop = positive_logprob_drop.amax(dim=0)
    token_drop_fraction = max_logprob_drop / (
        max_logprob_drop
        + (-metrics["full_target_logprob"]).clamp_min(0.0)
        + 1e-8
    )
    counterfactual_names = ("mean", "wrong")
    for record, span_index, context_index in zip(
        selected, token_indices, context_indices
    ):
        current_index = int(span_index[0].item())
        full_top1 = int(metrics["full_top1"][current_index].item())
        candidate_alignment = _candidate_alignment_metrics(
            logits,
            metrics["log_partition"],
            state_index=current_index,
            u_candidate_ids=record.get(
                "selective_u_root_candidate_token_ids", []
            ),
            gc_candidate_ids=record.get(
                "selective_gc_root_candidate_token_ids", []
            ),
        )
        context_count = int(context_index.numel())
        context_targets = target_ids.index_select(0, context_index)
        context_full_top1 = metrics["full_top1"].index_select(0, context_index)
        update = {
            "visual_probe_active": True,
            "visual_probe_protocol": COUNTERFACTUAL_BANK_PROTOCOL,
            "visual_probe_ablation_scope": "mean_and_matched_wrong_image",
            "visual_probe_jsd": float(
                metrics["pairwise_jsd"][:, current_index].mean().item()
            ),
            "visual_probe_top1_disagreement_rate": float(
                metrics["top1_changed"][:, current_index].float().mean().item()
            ),
            "visual_probe_full_top1_token_id": full_top1,
            "visual_probe_full_top1_matches_target": bool(
                full_top1 == int(record["selective_target_token_id"])
            ),
            "visual_probe_max_target_logprob_drop": float(
                max_logprob_drop[current_index].item()
            ),
            "visual_probe_full_target_logprob": float(
                metrics["full_target_logprob"][current_index].item()
            ),
            "visual_probe_span2_num_tokens": int(span_index.numel()),
            "visual_probe_span2_mean_jsd": float(
                metrics["pairwise_jsd"]
                .index_select(1, span_index)
                .mean()
                .item()
            ),
            "visual_probe_span2_mean_target_logprob_drop": float(
                max_logprob_drop.index_select(0, span_index).mean().item()
            ),
            "visual_probe_span2_mean_target_drop_fraction": float(
                token_drop_fraction.index_select(0, span_index).mean().item()
            ),
            "visual_probe_span2_same_text_trajectory": True,
            "visual_probe_span2_uses_future_tokens": bool(
                span_index.numel() > 1
            ),
            "visual_probe_context3_num_tokens": context_count,
            "visual_probe_context3_target_token_ids": [
                int(token) for token in context_targets.tolist()
            ],
            "visual_probe_context3_full_top1_token_ids": [
                int(token) for token in context_full_top1.tolist()
            ],
            "visual_probe_context3_full_top1_match_rate": (
                float(
                    context_full_top1.eq(context_targets).float().mean().item()
                )
                if context_count
                else None
            ),
            "visual_probe_context3_same_text_trajectory": True,
            "visual_probe_num_regions": 2,
            "visual_probe_num_visual_tokens": int(
                sequence.eq(base_model.config.image_token_id).sum().item()
            ),
            "visual_probe_num_raw_patches": int(views.patches_per_view),
            "visual_probe_num_views": 3,
            "visual_probe_used_grid_metadata": True,
            "visual_probe_same_text_trajectory": True,
            "visual_probe_recomputed_vision_encoder": True,
            "visual_probe_wrong_image_source_identity": str(
                wrong_image_pair["source_image_identity"]
            ),
            "visual_probe_wrong_image_used_category_fallback": bool(
                wrong_image_pair.get("used_category_fallback", False)
            ),
            "visual_probe_candidate_alignment_available": bool(
                candidate_alignment["available"]
            ),
            "visual_probe_candidate_alignment_budget": int(
                candidate_alignment["budget"]
            ),
            "visual_probe_candidate_alignment_invalid_reason": (
                candidate_alignment["invalid_reason"]
            ),
            "visual_probe_candidate_alignment_uses_target_outcome": False,
        }
        if candidate_alignment["available"]:
            view_names = ("full", "mean", "wrong")
            for source in ("u", "gc"):
                for name, value in zip(
                    view_names, candidate_alignment[f"{source}_log_mass"]
                ):
                    update[
                        f"visual_probe_{source}_candidate_logmass_{name}"
                    ] = float(value)
                for name, value in zip(
                    ("mean", "wrong"),
                    candidate_alignment[f"{source}_counterfactual_drops"],
                ):
                    update[
                        f"visual_probe_{source}_candidate_mass_drop_{name}"
                    ] = float(value)
                update[
                    f"visual_probe_{source}_candidate_consensus_support"
                ] = float(candidate_alignment[f"{source}_consensus_support"])
            update["visual_probe_gc_minus_u_candidate_visual_support"] = float(
                candidate_alignment["gc_minus_u_consensus_support"]
            )
        for view_index, name in enumerate(counterfactual_names):
            context_margin_drops = metrics["target_margin_drop"][
                view_index
            ].index_select(0, context_index)
            context_top1_changed = metrics["top1_changed"][view_index].index_select(
                0, context_index
            )
            span_margin_drops = metrics["target_margin_drop"][
                view_index
            ].index_select(0, span_index)
            span_logprob_drops = metrics["target_logprob_drop"][
                view_index
            ].index_select(0, span_index)
            span_jsd = metrics["pairwise_jsd"][view_index].index_select(
                0, span_index
            )
            span_top1_changed = metrics["top1_changed"][view_index].index_select(
                0, span_index
            )
            update.update(
                {
                    f"visual_probe_{name}_target_logprob_drop": float(
                        metrics["target_logprob_drop"][view_index, current_index].item()
                    ),
                    f"visual_probe_{name}_target_margin_drop": float(
                        metrics["target_margin_drop"][view_index, current_index].item()
                    ),
                    f"visual_probe_{name}_jsd": float(
                        metrics["pairwise_jsd"][view_index, current_index].item()
                    ),
                    f"visual_probe_{name}_top1_changed": bool(
                        metrics["top1_changed"][view_index, current_index].item()
                    ),
                    f"visual_probe_span2_{name}_mean_target_logprob_drop": float(
                        span_logprob_drops.mean().item()
                    ),
                    f"visual_probe_span2_{name}_mean_target_margin_drop": float(
                        span_margin_drops.mean().item()
                    ),
                    f"visual_probe_span2_{name}_mean_jsd": float(
                        span_jsd.mean().item()
                    ),
                    f"visual_probe_span2_{name}_top1_change_rate": float(
                        span_top1_changed.float().mean().item()
                    ),
                    f"visual_probe_context3_{name}_target_margin_drops": [
                        float(value) for value in context_margin_drops.tolist()
                    ],
                    f"visual_probe_context3_{name}_top1_changed": [
                        bool(value) for value in context_top1_changed.tolist()
                    ],
                    f"visual_probe_context3_{name}_max_target_margin_drop": (
                        float(context_margin_drops.max().item())
                        if context_count
                        else None
                    ),
                    f"visual_probe_context3_{name}_top1_change_rate": (
                        float(context_top1_changed.float().mean().item())
                        if context_count
                        else None
                    ),
                }
            )
        record.update(update)

    elapsed = time.perf_counter() - started
    del logits, logits_by_view
    return {
        "protocol": COUNTERFACTUAL_BANK_PROTOCOL,
        "num_states": len(selected),
        "num_scored_tokens": int(logits_to_keep.numel()),
        "num_requested_token_scores": int(requested_positions.numel()),
        "span_tokens": int(span_tokens),
        "context_lookback_tokens": int(context_tokens),
        "num_context_scored_tokens": int(context_positions.numel()),
        "num_views": views.num_views,
        "counterfactuals": ["mean_content", "matched_wrong_image"],
        "candidate_set_visual_alignment": True,
        "candidate_set_visual_alignment_uses_target_outcome": False,
        "num_raw_patches": views.patches_per_view,
        "elapsed_seconds": elapsed,
        "included_in_decode_timing": False,
        "wrong_image_pair": dict(wrong_image_pair),
    }
