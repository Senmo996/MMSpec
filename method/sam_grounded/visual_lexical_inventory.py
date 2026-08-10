"""Prompt-native visual lexical inventories for training-free drafting.

The helpers in this module deliberately operate on logits already produced by
the multimodal target prefill.  They do not introduce a learned head or an
auxiliary model.  The resulting inventory is sample-specific but independent
of the generated history, so it can supply candidates for token roots that
have never appeared in the prompt or an earlier decode step.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .counterfactual_probes import VisualProbeLayout


def unique_tokens(tokens: Iterable[int]) -> List[int]:
    """Return integer token IDs in first-occurrence order."""

    result: List[int] = []
    seen = set()
    for raw_token in tokens:
        token = int(raw_token)
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def rank_scores(
    scores: torch.Tensor,
    limit: int,
    *,
    excluded_token_ids: Sequence[int] = (),
    valid_vocab_size: Optional[int] = None,
) -> List[int]:
    """Rank a vocabulary score vector after deterministic token filtering."""

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    limit = max(int(limit), 0)
    if limit == 0 or scores.numel() == 0:
        return []
    vocab_size = int(scores.numel())
    valid_vocab_size = (
        vocab_size
        if valid_vocab_size is None
        else min(max(int(valid_vocab_size), 0), vocab_size)
    )
    if valid_vocab_size == 0:
        return []

    filtered = scores[:valid_vocab_size].float().clone()
    excluded = sorted(
        {
            int(token)
            for token in excluded_token_ids
            if 0 <= int(token) < valid_vocab_size
        }
    )
    if excluded:
        excluded_tensor = torch.tensor(
            excluded, dtype=torch.long, device=filtered.device
        )
        filtered[excluded_tensor] = -torch.inf
    finite_count = int(torch.isfinite(filtered).sum().item())
    keep = min(limit, finite_count)
    if keep == 0:
        return []
    return [int(token) for token in filtered.topk(keep).indices.tolist()]


def interleave_rankings(rankings: Sequence[Sequence[int]], limit: int) -> List[int]:
    """Round-robin multiple ranked lists while removing duplicates."""

    limit = max(int(limit), 0)
    if limit == 0:
        return []
    rows = [list(map(int, row)) for row in rankings if row]
    if not rows:
        return []
    selected: List[int] = []
    seen = set()
    max_length = max(len(row) for row in rows)
    for rank in range(max_length):
        for row in rows:
            if rank >= len(row):
                continue
            token = row[rank]
            if token in seen:
                continue
            selected.append(token)
            seen.add(token)
            if len(selected) == limit:
                return selected
    return selected


def _region_mean_logits(
    prompt_logits: torch.Tensor, layout: VisualProbeLayout
) -> torch.Tensor:
    rows = []
    visual_positions = layout.visual_positions.to(prompt_logits.device)
    region_ids = layout.region_ids.to(prompt_logits.device)
    for region in range(layout.num_regions):
        positions = visual_positions[region_ids.eq(region)]
        if positions.numel() == 0:
            continue
        rows.append(
            prompt_logits.index_select(0, positions).mean(dim=0, dtype=torch.float32)
        )
    if not rows:
        return torch.empty(
            (0, prompt_logits.shape[-1]),
            dtype=torch.float32,
            device=prompt_logits.device,
        )
    return torch.stack(rows, dim=0)


def build_visual_lexical_inventories(
    prompt_logits: torch.Tensor,
    visual_mask: torch.Tensor,
    layout: VisualProbeLayout,
    question_token_ids: Sequence[int],
    max_pool_size: int,
    *,
    excluded_token_ids: Sequence[int] = (),
    valid_vocab_size: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Build image, text-control, and combined prompt-native inventories.

    ``visual_region_contrast`` is the primary image-conditioned source.  It
    averages the target prefill logits within spatial image regions, subtracts
    the mean non-visual prompt logits, and takes a max over regions.  The
    subtraction suppresses tokens that are generally likely under the system
    and question text, while the regional max avoids washing out small objects.

    The other sources are explicit ablations and controls rather than hidden
    ingredients of the primary method.
    """

    if prompt_logits.ndim != 2:
        raise ValueError("prompt_logits must have shape [sequence, vocabulary]")
    if visual_mask.ndim != 1 or visual_mask.numel() != prompt_logits.shape[0]:
        raise ValueError("visual_mask must match the prompt sequence length")
    max_pool_size = max(int(max_pool_size), 0)
    if max_pool_size == 0:
        return {
            "visual_mean": [],
            "visual_max": [],
            "visual_region_raw": [],
            "visual_region_contrast": [],
            "visual_region_contrast_rr": [],
            "text_mean_control": [],
            "question_lexical": [],
            "visual_question_combined": [],
        }

    visual_positions = torch.nonzero(visual_mask, as_tuple=True)[0].to(
        prompt_logits.device
    )
    text_positions = torch.nonzero(~visual_mask, as_tuple=True)[0].to(
        prompt_logits.device
    )
    if visual_positions.numel() == 0:
        raise ValueError("at least one visual prompt token is required")
    if text_positions.numel() == 0:
        raise ValueError("at least one non-visual prompt token is required")

    visual_logits = prompt_logits.index_select(0, visual_positions)
    visual_mean_scores = visual_logits.mean(dim=0, dtype=torch.float32)
    visual_max_scores = visual_logits.max(dim=0).values.float()
    text_mean_scores = prompt_logits.index_select(0, text_positions).mean(
        dim=0, dtype=torch.float32
    )
    region_rows = _region_mean_logits(prompt_logits, layout)
    if region_rows.shape[0] == 0:
        region_rows = visual_mean_scores.unsqueeze(0)
    contrast_rows = region_rows - text_mean_scores.unsqueeze(0)

    rank_kwargs = {
        "excluded_token_ids": excluded_token_ids,
        "valid_vocab_size": valid_vocab_size,
    }
    visual_mean = rank_scores(
        visual_mean_scores, max_pool_size, **rank_kwargs
    )
    visual_max = rank_scores(visual_max_scores, max_pool_size, **rank_kwargs)
    visual_region_raw = rank_scores(
        region_rows.max(dim=0).values, max_pool_size, **rank_kwargs
    )
    visual_region_contrast = rank_scores(
        contrast_rows.max(dim=0).values, max_pool_size, **rank_kwargs
    )
    contrast_rankings = [
        rank_scores(row, max_pool_size, **rank_kwargs) for row in contrast_rows
    ]
    visual_region_contrast_rr = interleave_rankings(
        contrast_rankings, max_pool_size
    )
    text_mean_control = rank_scores(
        text_mean_scores, max_pool_size, **rank_kwargs
    )

    excluded_set = {int(token) for token in excluded_token_ids}
    vocabulary_limit = (
        int(prompt_logits.shape[-1])
        if valid_vocab_size is None
        else min(int(valid_vocab_size), int(prompt_logits.shape[-1]))
    )
    # Recent question tokens are ranked first.  This is a strong text-only
    # lexical control for answers that copy an entity, label, option, or number.
    question_lexical = unique_tokens(
        token
        for token in reversed([int(token) for token in question_token_ids])
        if 0 <= token < vocabulary_limit and token not in excluded_set
    )[:max_pool_size]
    visual_question_combined = interleave_rankings(
        [visual_region_contrast_rr, question_lexical], max_pool_size
    )

    return {
        "visual_mean": visual_mean,
        "visual_max": visual_max,
        "visual_region_raw": visual_region_raw,
        "visual_region_contrast": visual_region_contrast,
        "visual_region_contrast_rr": visual_region_contrast_rr,
        "text_mean_control": text_mean_control,
        "question_lexical": question_lexical,
        "visual_question_combined": visual_question_combined,
    }


def fuse_equal_budget_candidates(
    baseline_candidates: Sequence[int],
    inventory_candidates: Sequence[int],
    budget: int,
    inventory_slots: int,
) -> List[int]:
    """Inject inventory tokens without increasing the draft width.

    When a recycled row exists, ``inventory_slots`` tail positions are replaced
    by inventory candidates and the remaining positions stay anchored to the
    recycled row.  When no row exists, the otherwise-idle full budget is filled
    from the history-independent inventory.
    """

    budget = max(int(budget), 0)
    inventory_slots = min(max(int(inventory_slots), 0), budget)
    if budget == 0:
        return []
    baseline = unique_tokens(baseline_candidates)
    inventory = unique_tokens(inventory_candidates)
    if not baseline:
        return inventory[:budget]

    anchor_count = budget - inventory_slots
    selected = baseline[:anchor_count]
    seen = set(selected)
    for token in inventory:
        if token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) == budget:
            return selected
    for token in baseline[anchor_count:]:
        if token in seen:
            continue
        selected.append(token)
        seen.add(token)
        if len(selected) == budget:
            break
    return selected
