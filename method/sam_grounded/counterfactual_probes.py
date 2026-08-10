"""Utilities for training-free counterfactual visual evidence probes.

The probe batch is represented along the query-length dimension rather than
the batch dimension.  Every query has the same token and logical position,
but a different row in a pre-inverted 4-D attention mask.  Query zero is the
unmasked target view; the remaining queries each remove one spatial region
from the visual prefix.  Probe queries only attend to themselves among the new
KV slots, so they cannot exchange information with one another.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class VisualProbeLayout:
    """Spatial assignment used to construct counterfactual visual views."""

    visual_positions: torch.Tensor
    region_ids: torch.Tensor
    num_regions: int
    used_grid_metadata: bool

    @property
    def num_visual_tokens(self) -> int:
        return int(self.visual_positions.numel())


def _factor_grid(num_regions: int) -> Tuple[int, int]:
    """Return a near-square rows x columns factorization."""

    if num_regions <= 0:
        raise ValueError("num_regions must be positive")
    rows = int(math.sqrt(num_regions))
    while rows > 1 and num_regions % rows:
        rows -= 1
    return rows, num_regions // rows


def _fallback_region_ids(
    num_visual_tokens: int, num_regions: int, device: torch.device
) -> torch.Tensor:
    if num_visual_tokens == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    # Contiguous bands are preferable to modulo stripes when metadata is not
    # available because Qwen visual tokens are emitted in raster order.
    ids = torch.arange(num_visual_tokens, device=device, dtype=torch.long)
    return torch.div(ids * num_regions, num_visual_tokens, rounding_mode="floor").clamp(
        max=num_regions - 1
    )


def build_visual_probe_layout(
    visual_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
    num_regions: int,
) -> VisualProbeLayout:
    """Map Qwen visual tokens to a spatial grid of counterfactual regions.

    ``image_grid_thw`` describes the pre-merge visual grid.  Qwen emits
    ``t * (h / merge) * (w / merge)`` language-model visual tokens per image.
    When that metadata is absent or inconsistent, a deterministic raster-band
    fallback keeps the diagnostic usable without pretending the mapping is
    exact.
    """

    if visual_mask.ndim != 1:
        raise ValueError("visual_mask must be one-dimensional")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    if num_regions <= 0:
        raise ValueError("num_regions must be positive")

    visual_positions = torch.nonzero(visual_mask, as_tuple=True)[0]
    device = visual_positions.device
    fallback = _fallback_region_ids(
        int(visual_positions.numel()), num_regions, device
    )
    if image_grid_thw is None or visual_positions.numel() == 0:
        return VisualProbeLayout(
            visual_positions, fallback, num_regions, used_grid_metadata=False
        )

    rows, cols = _factor_grid(num_regions)
    assignments: List[int] = []
    try:
        grids = image_grid_thw.detach().to("cpu", dtype=torch.long).reshape(-1, 3)
        for temporal, height, width in grids.tolist():
            if height % spatial_merge_size or width % spatial_merge_size:
                raise ValueError("visual grid is not divisible by spatial merge size")
            merged_h = height // spatial_merge_size
            merged_w = width // spatial_merge_size
            if temporal <= 0 or merged_h <= 0 or merged_w <= 0:
                raise ValueError("invalid visual grid")
            for _time in range(temporal):
                for y in range(merged_h):
                    row = min((y * rows) // merged_h, rows - 1)
                    for x in range(merged_w):
                        col = min((x * cols) // merged_w, cols - 1)
                        assignments.append(row * cols + col)
    except (TypeError, ValueError, RuntimeError):
        assignments = []

    if len(assignments) != int(visual_positions.numel()):
        return VisualProbeLayout(
            visual_positions, fallback, num_regions, used_grid_metadata=False
        )
    region_ids = torch.tensor(assignments, dtype=torch.long, device=device)
    return VisualProbeLayout(
        visual_positions, region_ids, num_regions, used_grid_metadata=True
    )


def build_counterfactual_attention_mask(
    prefix_length: int,
    layout: VisualProbeLayout,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build an additive mask for one full query plus one drop-region query.

    The returned shape is ``[1, 1, 1 + R, prefix_length + 1 + R]``.  Row zero
    sees the complete prefix.  Row ``r + 1`` cannot see visual region ``r``.
    Every row sees its own new KV slot and no sibling probe slot.
    """

    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    if not dtype.is_floating_point:
        raise ValueError("attention mask dtype must be floating point")
    if layout.visual_positions.numel() and int(layout.visual_positions.max()) >= prefix_length:
        raise ValueError("visual position lies outside the cached prefix")

    query_length = layout.num_regions + 1
    key_length = prefix_length + query_length
    min_value = torch.finfo(dtype).min
    mask = torch.full(
        (1, 1, query_length, key_length),
        min_value,
        dtype=dtype,
        device=device,
    )
    mask[:, :, :, :prefix_length] = 0
    diagonal = prefix_length + torch.arange(query_length, device=device)
    mask[0, 0, torch.arange(query_length, device=device), diagonal] = 0

    visual_positions = layout.visual_positions.to(device)
    region_ids = layout.region_ids.to(device)
    for region in range(layout.num_regions):
        dropped = visual_positions[region_ids.eq(region)]
        if dropped.numel():
            mask[0, 0, region + 1, dropped] = min_value
    return mask


def build_tree_counterfactual_attention_mask(
    prefix_length: int,
    tree_mask: torch.Tensor,
    layout: VisualProbeLayout,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Append draft-only counterfactual root probes to a verifier tree mask.

    The ordinary tree occupies the first ``T`` query slots.  The following
    ``R`` slots duplicate the root token and logical position, with probe ``r``
    dropping visual region ``r``.  Tree queries keep their original ancestor
    visibility; probes see only their own new KV slot.  No tree path can attend
    to a probe, and callers must exclude probe indices from path retrieval.
    """

    if tree_mask.ndim != 4 or tree_mask.shape[:2] != (1, 1):
        raise ValueError("tree_mask must have shape [1, 1, tree_len, tree_len]")
    tree_length = int(tree_mask.shape[-1])
    if tree_mask.shape[-2] != tree_length or tree_length <= 0:
        raise ValueError("tree_mask must be non-empty and square")
    if prefix_length < 0:
        raise ValueError("prefix_length must be non-negative")
    if layout.visual_positions.numel() and int(layout.visual_positions.max()) >= prefix_length:
        raise ValueError("visual position lies outside the cached prefix")

    query_length = tree_length + layout.num_regions
    key_length = prefix_length + query_length
    min_value = torch.finfo(dtype).min
    mask = torch.full(
        (1, 1, query_length, key_length),
        min_value,
        dtype=dtype,
        device=device,
    )
    mask[:, :, :, :prefix_length] = 0

    allowed_tree = tree_mask[0, 0].to(device=device, dtype=torch.bool)
    tree_block = mask[0, 0, :tree_length, prefix_length : prefix_length + tree_length]
    tree_block[allowed_tree] = 0

    visual_positions = layout.visual_positions.to(device)
    region_ids = layout.region_ids.to(device)
    for region in range(layout.num_regions):
        query_index = tree_length + region
        mask[0, 0, query_index, prefix_length + query_index] = 0
        dropped = visual_positions[region_ids.eq(region)]
        if dropped.numel():
            mask[0, 0, query_index, dropped] = min_value
    return mask


def full_view_candidates(logits: torch.Tensor, budget: int) -> List[int]:
    """Return the conventional Token Recycling top-k row."""

    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    budget = min(max(int(budget), 0), int(logits.numel()))
    if budget == 0:
        return []
    return [int(token) for token in logits.topk(budget).indices.tolist()]


def cover_candidates(
    view_logits: torch.Tensor,
    budget: int,
    anchor_fraction: float = 0.5,
) -> List[int]:
    """Fuse full and counterfactual ranks under an exact candidate budget.

    A fixed fraction of the row is anchored to the full visual view.  Remaining
    slots are filled round-robin from counterfactual views, which makes the
    hypothesis test deliberately about *new visual evidence* rather than an
    unconstrained score ensemble.  If views agree, the full-view tail fills the
    unused slots.
    """

    if view_logits.ndim != 2 or view_logits.shape[0] < 1:
        raise ValueError("view_logits must have shape [num_views, vocab_size]")
    if not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError("anchor_fraction must lie in [0, 1]")
    budget = min(max(int(budget), 0), int(view_logits.shape[-1]))
    if budget == 0:
        return []

    anchor_count = min(budget, max(1, int(math.ceil(budget * anchor_fraction))))
    search_width = min(
        int(view_logits.shape[-1]), max(budget * 8, budget + 8)
    )
    ranks = view_logits.topk(search_width, dim=-1).indices.detach().to("cpu")
    selected = [int(token) for token in ranks[0, :anchor_count].tolist()]
    seen = set(selected)

    # Each rank round gives every counterfactual view one chance to contribute.
    for rank in range(search_width):
        for view in range(1, int(ranks.shape[0])):
            token = int(ranks[view, rank].item())
            if token in seen:
                continue
            selected.append(token)
            seen.add(token)
            if len(selected) == budget:
                return selected

    for token_tensor in ranks[0]:
        token = int(token_tensor.item())
        if token not in seen:
            selected.append(token)
            seen.add(token)
            if len(selected) == budget:
                break
    return selected


def compose_candidate_view_logits(
    full_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    mode: str = "masked",
    evidence_scale: float = 1.0,
) -> torch.Tensor:
    """Turn counterfactual outputs into candidate-producing view logits.

    ``masked`` directly recycles predictions from each drop-region view.
    ``evidence`` uses the counterfactual effect ``full - dropped`` to amplify
    tokens causally supported by the removed region.  ``both`` exposes both
    families to the same fixed-budget rank fusion.  All three modes use the
    same target forward; only the deterministic post-processing differs.
    """

    if full_logits.ndim != 1:
        raise ValueError("full_logits must be one-dimensional")
    if counterfactual_logits.ndim != 2:
        raise ValueError("counterfactual_logits must be two-dimensional")
    if counterfactual_logits.shape[-1] != full_logits.numel():
        raise ValueError("full and counterfactual vocabulary sizes differ")
    if mode not in ("masked", "evidence", "both"):
        raise ValueError("mode must be one of: masked, evidence, both")
    if evidence_scale < 0:
        raise ValueError("evidence_scale must be non-negative")

    full = full_logits.unsqueeze(0)
    evidence = full + float(evidence_scale) * (full - counterfactual_logits)
    if mode == "masked":
        return torch.cat([full, counterfactual_logits], dim=0)
    if mode == "evidence":
        return torch.cat([full, evidence], dim=0)
    return torch.cat([full, counterfactual_logits, evidence], dim=0)


def multiview_jsd(view_logits: torch.Tensor) -> float:
    """Jensen-Shannon divergence across full and counterfactual views."""

    if view_logits.ndim != 2 or view_logits.shape[0] < 1:
        raise ValueError("view_logits must have shape [num_views, vocab_size]")
    log_probs = torch.log_softmax(view_logits.float(), dim=-1)
    probs = log_probs.exp()
    mean_probs = probs.mean(dim=0)
    log_mean = mean_probs.clamp_min(torch.finfo(mean_probs.dtype).tiny).log()
    jsd = (probs * (log_probs - log_mean)).sum(dim=-1).mean()
    return float(jsd.item())


def topk_union_size(view_logits: torch.Tensor, k: int) -> int:
    """Number of unique tokens in the per-view top-k union."""

    if view_logits.ndim != 2:
        raise ValueError("view_logits must be two-dimensional")
    k = min(max(int(k), 0), int(view_logits.shape[-1]))
    if k == 0:
        return 0
    return int(torch.unique(view_logits.topk(k, dim=-1).indices).numel())


def make_repeated_position_ids(
    logical_position: int,
    query_length: int,
    rope_deltas: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Create Qwen M-RoPE IDs for same-position query replicas."""

    if logical_position < 0 or query_length <= 0:
        raise ValueError("invalid logical position or query length")
    positions = torch.full(
        (1, query_length), logical_position, dtype=torch.long, device=device
    )
    if rope_deltas is not None:
        delta = rope_deltas.to(device=device, dtype=torch.long).reshape(-1, 1)
        if delta.shape[0] != 1:
            raise ValueError("counterfactual probes currently require batch size 1")
        positions = positions + delta
    return positions.unsqueeze(0).expand(3, -1, -1)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Small dependency-free Pearson correlation helper."""

    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denom.item()) == 0.0:
        return None
    return float((x * y).sum().div(denom).item())
