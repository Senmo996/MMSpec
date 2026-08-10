"""Training-free prompt-local transition-kernel candidate reranking.

The kernel approximates the missing transition row for an unseen root token by
retrieving prompt/history transitions with a similar source token and/or source
context.  It only scores a bounded candidate pool and does not run an extra
Transformer forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransitionKernelConfig:
    """A single transition-kernel policy."""

    token_weight: float
    neighbors: int
    temperature: float
    visual_weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.token_weight) <= 1.0:
            raise ValueError("token_weight must be in [0, 1]")
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= float(self.visual_weight) <= 1.0:
            raise ValueError("visual_weight must be in [0, 1]")

    @property
    def key(self) -> str:
        return (
            f"tw{self.token_weight:.2f}-k{self.neighbors}-"
            f"t{self.temperature:.2f}-vw{self.visual_weight:.2f}"
        )


def build_config_grid(
    token_weights: Iterable[float],
    neighbors: Iterable[int],
    temperatures: Iterable[float],
    visual_weights: Iterable[float],
) -> list[TransitionKernelConfig]:
    """Build a deterministic de-duplicated configuration grid."""

    configs = []
    seen = set()
    for token_weight in token_weights:
        for neighbor_count in neighbors:
            for temperature in temperatures:
                for visual_weight in visual_weights:
                    config = TransitionKernelConfig(
                        token_weight=float(token_weight),
                        neighbors=int(neighbor_count),
                        temperature=float(temperature),
                        visual_weight=float(visual_weight),
                    )
                    if config.key not in seen:
                        configs.append(config)
                        seen.add(config.key)
    return configs


def _normalize_vectors(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), p=2, dim=-1, eps=1e-6)


def standardize_scores(values: torch.Tensor) -> torch.Tensor:
    """Standardize the final dimension while keeping constant rows finite."""

    values = values.float()
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


def rank_prior(
    candidate_count: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return a standardized log-rank prior preserving the pool order."""

    if int(candidate_count) <= 0:
        raise ValueError("candidate_count must be positive")
    ranks = torch.arange(
        1, int(candidate_count) + 1, dtype=torch.float32, device=device
    )
    return standardize_scores(-torch.log(ranks))


def valid_prompt_transition_positions(
    prompt_token_ids: torch.Tensor,
    visual_mask: torch.Tensor,
    excluded_token_ids: Sequence[int],
) -> torch.Tensor:
    """Select text-token transitions with a preceding contextual state."""

    prompt_token_ids = prompt_token_ids.reshape(-1)
    visual_mask = visual_mask.reshape(-1).bool()
    if prompt_token_ids.shape[0] != visual_mask.shape[0]:
        raise ValueError("prompt_token_ids and visual_mask must have equal length")
    if prompt_token_ids.numel() < 2:
        return torch.empty(0, dtype=torch.long, device=prompt_token_ids.device)

    positions = torch.arange(
        1, prompt_token_ids.shape[0], dtype=torch.long, device=prompt_token_ids.device
    )
    keep = ~visual_mask.index_select(0, positions)
    if excluded_token_ids:
        excluded = torch.tensor(
            sorted(set(int(token) for token in excluded_token_ids)),
            dtype=prompt_token_ids.dtype,
            device=prompt_token_ids.device,
        )
        source_tokens = prompt_token_ids.index_select(0, positions)
        keep &= ~torch.isin(source_tokens, excluded)
    return positions[keep]


class TransitionKernelBank:
    """Candidate-specific transition rows indexed by token and context."""

    def __init__(
        self,
        *,
        candidate_token_ids: torch.Tensor,
        source_token_vectors: torch.Tensor,
        source_context_vectors: torch.Tensor,
        source_candidate_logits: torch.Tensor,
        candidate_prior: Optional[torch.Tensor] = None,
    ) -> None:
        candidate_token_ids = candidate_token_ids.reshape(-1).long()
        if candidate_token_ids.numel() == 0:
            raise ValueError("candidate_token_ids must not be empty")
        if torch.unique(candidate_token_ids).numel() != candidate_token_ids.numel():
            raise ValueError("candidate_token_ids must be unique")
        source_count = int(source_token_vectors.shape[0])
        if source_context_vectors.shape[0] != source_count:
            raise ValueError("source token/context counts differ")
        if source_candidate_logits.shape != (
            source_count,
            candidate_token_ids.numel(),
        ):
            raise ValueError("source_candidate_logits has incompatible shape")
        if source_count == 0:
            raise ValueError("at least one source transition is required")

        self.candidate_token_ids = candidate_token_ids
        self.source_token_vectors = _normalize_vectors(source_token_vectors)
        self.source_context_vectors = _normalize_vectors(source_context_vectors)
        self.source_candidate_scores = standardize_scores(source_candidate_logits)
        if candidate_prior is None:
            candidate_prior = rank_prior(
                candidate_token_ids.numel(), device=candidate_token_ids.device
            )
        if candidate_prior.reshape(-1).shape[0] != candidate_token_ids.numel():
            raise ValueError("candidate_prior has incompatible shape")
        self.candidate_prior = standardize_scores(candidate_prior.reshape(-1))

    @classmethod
    def from_prompt(
        cls,
        *,
        candidate_token_ids: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        prompt_token_embeddings: torch.Tensor,
        prompt_hidden_states: torch.Tensor,
        prompt_logits: torch.Tensor,
        visual_mask: torch.Tensor,
        excluded_token_ids: Sequence[int],
        candidate_prior: Optional[torch.Tensor] = None,
    ) -> "TransitionKernelBank":
        """Construct a bank using target-model outputs already made at prefill."""

        prompt_token_ids = prompt_token_ids.reshape(-1)
        prompt_token_embeddings = prompt_token_embeddings.reshape(
            prompt_token_ids.shape[0], -1
        )
        prompt_hidden_states = prompt_hidden_states.reshape(
            prompt_token_ids.shape[0], -1
        )
        if prompt_logits.shape[0] != prompt_token_ids.shape[0]:
            raise ValueError("prompt logits and tokens must have equal length")
        positions = valid_prompt_transition_positions(
            prompt_token_ids, visual_mask, excluded_token_ids
        )
        if positions.numel() == 0:
            raise ValueError("prompt has no valid transition positions")

        candidate_token_ids = candidate_token_ids.reshape(-1).long()
        row_positions = positions[:, None].expand(-1, candidate_token_ids.numel())
        candidate_positions = candidate_token_ids[None, :].expand(
            positions.numel(), -1
        )
        source_candidate_logits = prompt_logits[row_positions, candidate_positions]
        return cls(
            candidate_token_ids=candidate_token_ids,
            source_token_vectors=prompt_token_embeddings.index_select(0, positions),
            source_context_vectors=prompt_hidden_states.index_select(
                0, positions - 1
            ),
            source_candidate_logits=source_candidate_logits,
            candidate_prior=candidate_prior,
        )

    @property
    def source_count(self) -> int:
        return int(self.source_token_vectors.shape[0])

    def append(
        self,
        *,
        source_token_vector: torch.Tensor,
        source_context_vector: torch.Tensor,
        candidate_logits: torch.Tensor,
    ) -> None:
        """Append a verified online transition for later decoding steps."""

        token_vector = _normalize_vectors(source_token_vector.reshape(1, -1))
        context_vector = _normalize_vectors(source_context_vector.reshape(1, -1))
        row = candidate_logits.reshape(1, -1)
        if row.shape[1] != self.candidate_token_ids.numel():
            raise ValueError("candidate_logits has incompatible shape")
        self.source_token_vectors = torch.cat(
            [self.source_token_vectors, token_vector], dim=0
        )
        self.source_context_vectors = torch.cat(
            [self.source_context_vectors, context_vector], dim=0
        )
        self.source_candidate_scores = torch.cat(
            [self.source_candidate_scores, standardize_scores(row)], dim=0
        )

    def retrieval_similarities(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_vector: torch.Tensor,
        token_weight: float,
    ) -> torch.Tensor:
        query_token = _normalize_vectors(query_token_vector.reshape(1, -1))[0]
        query_context = _normalize_vectors(query_context_vector.reshape(1, -1))[0]
        token_similarity = self.source_token_vectors @ query_token
        context_similarity = self.source_context_vectors @ query_context
        weight = float(token_weight)
        return weight * token_similarity + (1.0 - weight) * context_similarity

    def score(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_vector: torch.Tensor,
        config: TransitionKernelConfig,
    ) -> torch.Tensor:
        similarities = self.retrieval_similarities(
            query_token_vector=query_token_vector,
            query_context_vector=query_context_vector,
            token_weight=config.token_weight,
        )
        neighbor_count = min(int(config.neighbors), similarities.numel())
        neighbor_scores, neighbor_indices = similarities.topk(neighbor_count)
        weights = torch.softmax(neighbor_scores / float(config.temperature), dim=0)
        transition_score = weights @ self.source_candidate_scores.index_select(
            0, neighbor_indices
        )
        transition_score = standardize_scores(transition_score)
        visual_weight = float(config.visual_weight)
        return (
            (1.0 - visual_weight) * transition_score
            + visual_weight * self.candidate_prior
        )

    def score_grid(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_vector: torch.Tensor,
        configs: Sequence[TransitionKernelConfig],
    ) -> Dict[str, torch.Tensor]:
        """Score a small diagnostic grid, reusing retrieval similarities."""

        similarity_cache: Dict[float, torch.Tensor] = {}
        transition_cache: Dict[tuple[float, int, float], torch.Tensor] = {}
        results: Dict[str, torch.Tensor] = {}
        for config in configs:
            token_weight = float(config.token_weight)
            if token_weight not in similarity_cache:
                similarity_cache[token_weight] = self.retrieval_similarities(
                    query_token_vector=query_token_vector,
                    query_context_vector=query_context_vector,
                    token_weight=token_weight,
                )
            similarities = similarity_cache[token_weight]
            neighbor_count = min(int(config.neighbors), similarities.numel())
            transition_key = (
                token_weight,
                neighbor_count,
                float(config.temperature),
            )
            if transition_key not in transition_cache:
                neighbor_scores, neighbor_indices = similarities.topk(neighbor_count)
                weights = torch.softmax(
                    neighbor_scores / float(config.temperature), dim=0
                )
                transition_score = weights @ self.source_candidate_scores.index_select(
                    0, neighbor_indices
                )
                transition_cache[transition_key] = standardize_scores(
                    transition_score
                )
            visual_weight = float(config.visual_weight)
            results[config.key] = (
                (1.0 - visual_weight) * transition_cache[transition_key]
                + visual_weight * self.candidate_prior
            )
        return results


def target_rank(
    scores: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    target_token_id: int,
) -> Optional[int]:
    """Return a deterministic one-indexed rank, or ``None`` outside the pool."""

    scores = scores.reshape(-1)
    candidates = candidate_token_ids.reshape(-1)
    matches = torch.nonzero(candidates == int(target_token_id), as_tuple=False)
    if matches.numel() == 0:
        return None
    target_index = int(matches[0, 0].item())
    target_score = scores[target_index]
    indices = torch.arange(scores.numel(), device=scores.device)
    better = scores > target_score
    tied_before = (scores == target_score) & (indices < target_index)
    return 1 + int((better | tied_before).sum().item())
