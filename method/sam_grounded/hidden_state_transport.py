"""Prompt-local hidden-state transport for training-free multimodal drafting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from method.sam_grounded.transition_kernel import (
    rank_prior,
    standardize_scores,
    valid_prompt_transition_positions,
)


TRANSPORT_MODES = ("delta_half", "delta_full", "post_state")
SOURCE_SCOPES = ("all_text", "post_visual_text")


@dataclass(frozen=True)
class HiddenStateTransportConfig:
    token_weight: float
    neighbors: int
    temperature: float
    transport_mode: str
    source_scope: str
    visual_weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.token_weight) <= 1.0:
            raise ValueError("token_weight must be in [0, 1]")
        if int(self.neighbors) <= 0:
            raise ValueError("neighbors must be positive")
        if float(self.temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if self.transport_mode not in TRANSPORT_MODES:
            raise ValueError(f"unknown transport_mode: {self.transport_mode}")
        if self.source_scope not in SOURCE_SCOPES:
            raise ValueError(f"unknown source_scope: {self.source_scope}")
        if not 0.0 <= float(self.visual_weight) <= 1.0:
            raise ValueError("visual_weight must be in [0, 1]")

    @property
    def key(self) -> str:
        mode = {
            "delta_half": "dh",
            "delta_full": "df",
            "post_state": "ps",
        }[self.transport_mode]
        scope = {"all_text": "all", "post_visual_text": "postvis"}[
            self.source_scope
        ]
        return (
            f"tw{self.token_weight:.2f}-k{self.neighbors}-"
            f"t{self.temperature:.2f}-m{mode}-s{scope}-"
            f"vw{self.visual_weight:.2f}"
        )


def build_transport_config_grid(
    token_weights: Iterable[float],
    neighbors: Iterable[int],
    temperatures: Iterable[float],
    transport_modes: Iterable[str],
    source_scopes: Iterable[str],
    visual_weights: Iterable[float],
) -> list[HiddenStateTransportConfig]:
    configs = []
    seen = set()
    for token_weight in token_weights:
        for neighbor_count in neighbors:
            for temperature in temperatures:
                for transport_mode in transport_modes:
                    for source_scope in source_scopes:
                        for visual_weight in visual_weights:
                            config = HiddenStateTransportConfig(
                                token_weight=float(token_weight),
                                neighbors=int(neighbor_count),
                                temperature=float(temperature),
                                transport_mode=str(transport_mode),
                                source_scope=str(source_scope),
                                visual_weight=float(visual_weight),
                            )
                            if config.key not in seen:
                                configs.append(config)
                                seen.add(config.key)
    return configs


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), p=2, dim=-1, eps=1e-6)


class HiddenStateTransportBank:
    """Prompt/history state transitions plus a bounded output projection."""

    def __init__(
        self,
        *,
        candidate_token_ids: torch.Tensor,
        candidate_projection_weight: torch.Tensor,
        candidate_projection_bias: Optional[torch.Tensor],
        source_token_vectors: torch.Tensor,
        source_context_vectors: torch.Tensor,
        source_post_hidden: torch.Tensor,
        source_post_visual_mask: torch.Tensor,
        candidate_prior: Optional[torch.Tensor] = None,
        reserve_capacity: Optional[int] = None,
    ) -> None:
        candidate_token_ids = candidate_token_ids.reshape(-1).long()
        if candidate_token_ids.numel() == 0:
            raise ValueError("candidate_token_ids must not be empty")
        if torch.unique(candidate_token_ids).numel() != candidate_token_ids.numel():
            raise ValueError("candidate_token_ids must be unique")
        source_count = int(source_token_vectors.shape[0])
        if source_count == 0:
            raise ValueError("at least one source transition is required")
        if source_context_vectors.shape[0] != source_count:
            raise ValueError("source token/context counts differ")
        if source_post_hidden.shape[0] != source_count:
            raise ValueError("source post-hidden count differs")
        if source_post_visual_mask.reshape(-1).shape[0] != source_count:
            raise ValueError("source scope mask count differs")
        hidden_size = int(source_context_vectors.shape[-1])
        if candidate_projection_weight.shape != (
            candidate_token_ids.numel(),
            hidden_size,
        ):
            raise ValueError("candidate projection has incompatible shape")

        self.candidate_token_ids = candidate_token_ids
        self.candidate_projection_weight = candidate_projection_weight.float()
        self.candidate_projection_bias = (
            candidate_projection_bias.reshape(-1).float()
            if candidate_projection_bias is not None
            else None
        )
        if (
            self.candidate_projection_bias is not None
            and self.candidate_projection_bias.numel()
            != candidate_token_ids.numel()
        ):
            raise ValueError("candidate projection bias has incompatible shape")
        normalized_tokens = _normalize(source_token_vectors)
        normalized_contexts = _normalize(source_context_vectors)
        context_hidden = source_context_vectors.float()
        post_hidden = source_post_hidden.float()
        deltas = post_hidden - context_hidden
        projection_transpose = self.candidate_projection_weight.t()
        delta_candidate_logits = deltas @ projection_transpose
        post_candidate_logits = post_hidden @ projection_transpose
        if self.candidate_projection_bias is not None:
            post_candidate_logits = (
                post_candidate_logits + self.candidate_projection_bias
            )
        post_visual_mask = source_post_visual_mask.reshape(-1).bool()
        self._source_count = source_count
        self._source_capacity = max(
            source_count,
            int(reserve_capacity) if reserve_capacity is not None else source_count,
        )
        self._source_indices = torch.arange(
            self._source_capacity,
            dtype=torch.long,
            device=source_token_vectors.device,
        )

        def reserve_rows(values: torch.Tensor) -> torch.Tensor:
            if self._source_capacity == source_count:
                return values
            storage = torch.empty(
                (self._source_capacity, *values.shape[1:]),
                dtype=values.dtype,
                device=values.device,
            )
            storage[:source_count].copy_(values)
            return storage

        self.source_token_vectors = reserve_rows(normalized_tokens)
        self.source_context_vectors = reserve_rows(normalized_contexts)
        self.source_context_hidden = reserve_rows(context_hidden)
        self.source_post_hidden = reserve_rows(post_hidden)
        self.source_deltas = reserve_rows(deltas)
        self.source_delta_candidate_logits = reserve_rows(
            delta_candidate_logits
        )
        self.source_post_candidate_logits = reserve_rows(
            post_candidate_logits
        )
        self.source_post_visual_mask = reserve_rows(post_visual_mask)
        if candidate_prior is None:
            candidate_prior = rank_prior(
                candidate_token_ids.numel(), device=candidate_token_ids.device
            )
        self.candidate_prior = standardize_scores(candidate_prior.reshape(-1))

    @classmethod
    def from_prompt(
        cls,
        *,
        candidate_token_ids: torch.Tensor,
        output_projection_weight: torch.Tensor,
        output_projection_bias: Optional[torch.Tensor],
        prompt_token_ids: torch.Tensor,
        prompt_token_embeddings: torch.Tensor,
        prompt_hidden_states: torch.Tensor,
        visual_mask: torch.Tensor,
        excluded_token_ids: Sequence[int],
        candidate_prior: Optional[torch.Tensor] = None,
        additional_capacity: int = 0,
    ) -> "HiddenStateTransportBank":
        prompt_token_ids = prompt_token_ids.reshape(-1)
        prompt_token_embeddings = prompt_token_embeddings.reshape(
            prompt_token_ids.shape[0], -1
        )
        prompt_hidden_states = prompt_hidden_states.reshape(
            prompt_token_ids.shape[0], -1
        )
        positions = valid_prompt_transition_positions(
            prompt_token_ids, visual_mask, excluded_token_ids
        )
        if positions.numel() == 0:
            raise ValueError("prompt has no valid transition positions")
        candidate_token_ids = candidate_token_ids.reshape(-1).long()
        projection = output_projection_weight.index_select(
            0, candidate_token_ids
        )
        bias = (
            output_projection_bias.index_select(0, candidate_token_ids)
            if output_projection_bias is not None
            else None
        )
        visual_positions = torch.nonzero(
            visual_mask.reshape(-1).bool(), as_tuple=False
        ).reshape(-1)
        last_visual_position = (
            int(visual_positions[-1].item()) if visual_positions.numel() else -1
        )
        return cls(
            candidate_token_ids=candidate_token_ids,
            candidate_projection_weight=projection,
            candidate_projection_bias=bias,
            source_token_vectors=prompt_token_embeddings.index_select(0, positions),
            source_context_vectors=prompt_hidden_states.index_select(
                0, positions - 1
            ),
            source_post_hidden=prompt_hidden_states.index_select(0, positions),
            source_post_visual_mask=positions > last_visual_position,
            candidate_prior=candidate_prior,
            reserve_capacity=int(positions.numel())
            + max(int(additional_capacity), 0),
        )

    @property
    def source_count(self) -> int:
        return int(self._source_count)

    @property
    def post_visual_source_count(self) -> int:
        return int(
            self.source_post_visual_mask[: self.source_count].sum().item()
        )

    def _ensure_source_capacity(self, required: int) -> None:
        if required <= self._source_capacity:
            return
        new_capacity = max(
            required,
            self._source_capacity + max(self._source_capacity // 2, 32),
        )

        def grow(values: torch.Tensor) -> torch.Tensor:
            storage = torch.empty(
                (new_capacity, *values.shape[1:]),
                dtype=values.dtype,
                device=values.device,
            )
            storage[: self.source_count].copy_(values[: self.source_count])
            return storage

        self.source_token_vectors = grow(self.source_token_vectors)
        self.source_context_vectors = grow(self.source_context_vectors)
        self.source_context_hidden = grow(self.source_context_hidden)
        self.source_post_hidden = grow(self.source_post_hidden)
        self.source_deltas = grow(self.source_deltas)
        self.source_delta_candidate_logits = grow(
            self.source_delta_candidate_logits
        )
        self.source_post_candidate_logits = grow(
            self.source_post_candidate_logits
        )
        self.source_post_visual_mask = grow(self.source_post_visual_mask)
        self._source_indices = torch.arange(
            new_capacity,
            dtype=torch.long,
            device=self.source_token_vectors.device,
        )
        self._source_capacity = new_capacity

    def append(
        self,
        *,
        source_token_vector: torch.Tensor,
        source_context_hidden: torch.Tensor,
        source_post_hidden: torch.Tensor,
    ) -> None:
        token_vector = _normalize(source_token_vector.reshape(1, -1))[0]
        context_hidden = source_context_hidden.reshape(-1).float()
        post_hidden = source_post_hidden.reshape(-1).float()
        write_index = self.source_count
        self._ensure_source_capacity(write_index + 1)
        self.source_token_vectors[write_index].copy_(token_vector)
        self.source_context_vectors[write_index].copy_(
            _normalize(context_hidden.reshape(1, -1))[0]
        )
        self.source_context_hidden[write_index].copy_(context_hidden)
        self.source_post_hidden[write_index].copy_(post_hidden)
        delta = post_hidden - context_hidden
        self.source_deltas[write_index].copy_(delta)
        self.source_delta_candidate_logits[write_index].copy_(
            self.candidate_projection_weight @ delta
        )
        post_candidate_logits = self.candidate_projection_weight @ post_hidden
        if self.candidate_projection_bias is not None:
            post_candidate_logits = (
                post_candidate_logits + self.candidate_projection_bias
            )
        self.source_post_candidate_logits[write_index].copy_(
            post_candidate_logits
        )
        self.source_post_visual_mask[write_index] = True
        self._source_count += 1

    def _scope_indices(self, source_scope: str) -> torch.Tensor:
        if source_scope == "all_text":
            return self._source_indices[: self.source_count]
        indices = torch.nonzero(
            self.source_post_visual_mask[: self.source_count], as_tuple=False
        ).reshape(-1)
        if indices.numel() == 0:
            raise ValueError("post_visual_text scope has no source transitions")
        return indices

    def retrieval_similarities(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_hidden: torch.Tensor,
        token_weight: float,
        source_scope: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_indices = self._scope_indices(source_scope)
        query_token = _normalize(query_token_vector.reshape(1, -1))[0]
        query_context = _normalize(query_context_hidden.reshape(1, -1))[0]
        if source_scope == "all_text":
            source_tokens = self.source_token_vectors[: self.source_count]
            source_contexts = self.source_context_vectors[: self.source_count]
        else:
            source_tokens = self.source_token_vectors.index_select(
                0, source_indices
            )
            source_contexts = self.source_context_vectors.index_select(
                0, source_indices
            )
        token_similarity = source_tokens @ query_token
        context_similarity = source_contexts @ query_context
        weight = float(token_weight)
        similarities = (
            weight * token_similarity + (1.0 - weight) * context_similarity
        )
        return similarities, source_indices

    def project_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        scores = self.candidate_projection_weight @ hidden_state.reshape(-1).float()
        if self.candidate_projection_bias is not None:
            scores = scores + self.candidate_projection_bias
        return scores

    def project_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return standardize_scores(self.project_logits(hidden_state))

    def _transported_hidden(
        self,
        *,
        query_context_hidden: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor,
        transport_mode: str,
    ) -> torch.Tensor:
        parent = query_context_hidden.reshape(-1).float()
        if transport_mode == "post_state":
            return neighbor_weights @ self.source_post_hidden.index_select(
                0, neighbor_indices
            )
        delta = neighbor_weights @ self.source_deltas.index_select(
            0, neighbor_indices
        )
        scale = 0.5 if transport_mode == "delta_half" else 1.0
        return parent + scale * delta

    def score(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_hidden: torch.Tensor,
        config: HiddenStateTransportConfig,
        query_candidate_logits: Optional[torch.Tensor] = None,
        rank_only: bool = False,
    ) -> torch.Tensor:
        if query_candidate_logits is None:
            predicted_hidden = self.transport_hidden(
                query_token_vector=query_token_vector,
                query_context_hidden=query_context_hidden,
                config=config,
            )
            transported_logits = self.project_logits(predicted_hidden)
        else:
            neighbor_indices, neighbor_weights = self._retrieve_neighbors(
                query_token_vector=query_token_vector,
                query_context_hidden=query_context_hidden,
                config=config,
            )
            transported_logits = self._transported_candidate_logits(
                query_candidate_logits=query_candidate_logits,
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                transport_mode=config.transport_mode,
            )
        visual_weight = float(config.visual_weight)
        if rank_only and visual_weight == 0.0:
            # Standardization is monotonic and therefore cannot change top-k.
            # Avoid its extra reduction kernels on the latency-critical path.
            return transported_logits
        transported_scores = standardize_scores(transported_logits)
        return (
            (1.0 - visual_weight) * transported_scores
            + visual_weight * self.candidate_prior
        )

    def transport_hidden(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_hidden: torch.Tensor,
        config: HiddenStateTransportConfig,
    ) -> torch.Tensor:
        """Predict the post-token hidden state without projecting candidates."""

        neighbor_indices, weights = self._retrieve_neighbors(
            query_token_vector=query_token_vector,
            query_context_hidden=query_context_hidden,
            config=config,
        )
        return self._transported_hidden(
            query_context_hidden=query_context_hidden,
            neighbor_indices=neighbor_indices,
            neighbor_weights=weights,
            transport_mode=config.transport_mode,
        )

    def _retrieve_neighbors(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_hidden: torch.Tensor,
        config: HiddenStateTransportConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        similarities, source_indices = self.retrieval_similarities(
            query_token_vector=query_token_vector,
            query_context_hidden=query_context_hidden,
            token_weight=config.token_weight,
            source_scope=config.source_scope,
        )
        neighbor_count = min(int(config.neighbors), similarities.numel())
        neighbor_scores, local_indices = similarities.topk(neighbor_count)
        return (
            source_indices.index_select(0, local_indices),
            torch.softmax(
                neighbor_scores / float(config.temperature), dim=0
            ),
        )

    def _transported_candidate_logits(
        self,
        *,
        query_candidate_logits: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor,
        transport_mode: str,
    ) -> torch.Tensor:
        candidate_count = int(self.candidate_token_ids.numel())
        parent_logits = query_candidate_logits.reshape(-1).float()
        if parent_logits.numel() != candidate_count:
            raise ValueError(
                "query_candidate_logits must match the candidate pool size"
            )
        if transport_mode == "post_state":
            return neighbor_weights @ self.source_post_candidate_logits.index_select(
                0, neighbor_indices
            )
        delta_logits = (
            neighbor_weights
            @ self.source_delta_candidate_logits.index_select(
                0, neighbor_indices
            )
        )
        scale = 0.5 if transport_mode == "delta_half" else 1.0
        return parent_logits + scale * delta_logits

    def score_grid(
        self,
        *,
        query_token_vector: torch.Tensor,
        query_context_hidden: torch.Tensor,
        configs: Sequence[HiddenStateTransportConfig],
    ) -> Dict[str, torch.Tensor]:
        retrieval_cache: Dict[tuple[str, float], tuple[torch.Tensor, torch.Tensor]] = {}
        neighbor_cache: Dict[
            tuple[str, float, int, float], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        projection_cache: Dict[tuple[str, float, int, float, str], torch.Tensor] = {}
        results = {}
        for config in configs:
            retrieval_key = (config.source_scope, float(config.token_weight))
            if retrieval_key not in retrieval_cache:
                retrieval_cache[retrieval_key] = self.retrieval_similarities(
                    query_token_vector=query_token_vector,
                    query_context_hidden=query_context_hidden,
                    token_weight=config.token_weight,
                    source_scope=config.source_scope,
                )
            similarities, source_indices = retrieval_cache[retrieval_key]
            neighbor_count = min(int(config.neighbors), similarities.numel())
            neighbor_key = (
                config.source_scope,
                float(config.token_weight),
                neighbor_count,
                float(config.temperature),
            )
            if neighbor_key not in neighbor_cache:
                neighbor_scores, local_indices = similarities.topk(neighbor_count)
                neighbor_cache[neighbor_key] = (
                    source_indices.index_select(0, local_indices),
                    torch.softmax(
                        neighbor_scores / float(config.temperature), dim=0
                    ),
                )
            projection_key = (*neighbor_key, config.transport_mode)
            if projection_key not in projection_cache:
                neighbor_indices, weights = neighbor_cache[neighbor_key]
                predicted_hidden = self._transported_hidden(
                    query_context_hidden=query_context_hidden,
                    neighbor_indices=neighbor_indices,
                    neighbor_weights=weights,
                    transport_mode=config.transport_mode,
                )
                projection_cache[projection_key] = self.project_hidden(
                    predicted_hidden
                )
            visual_weight = float(config.visual_weight)
            results[config.key] = (
                (1.0 - visual_weight) * projection_cache[projection_key]
                + visual_weight * self.candidate_prior
            )
        return results
