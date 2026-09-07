"""Grounded broad/shallow versus narrow/deep token-recycling trees."""

import heapq

from typing import List, Optional, Tuple

import torch

from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids, _has_stop_token

from .controller import (
    GroundedDraftController,
    VisualGroundingCalibrator,
    confidence_from_logits,
)
from .counterfactual_probes import (
    build_counterfactual_attention_mask,
    build_tree_counterfactual_attention_mask,
    build_visual_probe_layout,
    cover_candidates,
    multiview_jsd,
    topk_union_size,
)
from .hidden_state_transport import (
    HiddenStateTransportBank,
    HiddenStateTransportConfig,
    SOURCE_SCOPES,
    TRANSPORT_MODES,
)
from .spec_model import (
    SUPPORTED_MULTIMODAL_ARCHITECTURES,
    SpecModel as _GroundedSamSpecModel,
    resolve_generation_max_length,
    resolve_image_token_id,
)
from .visual_lexical_inventory import fuse_equal_budget_candidates, rank_scores


SELECTIVE_REUSE_PROBE_MODES = frozenset(
    {
        "packed-attention",
        "teacher-forced-pixel",
        "teacher-forced-content-ablation",
        "teacher-forced-counterfactual-bank",
    }
)


TRIGRAM_PLUS4_NODE_BUDGETS = {
    f"context-score-trigram-deeper-wide-plus4-node{budget}": budget
    for budget in (23, 31, 39, 47, 55)
}

PERSISTENT_FUSION_NODE_BUDGETS = {
    f"context-score-trigram-fusion-persistent-node{budget}-deepest-wide-plus4": budget
    for budget in (39, 47, 55, 63, 79, 95)
}

PERSISTENT_COMMITTED_POLICIES = {
    "context-score-trigram-fusion-persistent-committed-deepest-wide-plus4"
}

PERSISTENT_STABLE_NODE_BUDGETS = {
    f"context-score-trigram-fusion-persistent-stable-node{budget}-deepest-wide-plus4": budget
    for budget in (55, 63)
}

PERSISTENT_DEPTH_NODE_CONFIGS = {
    f"context-score-trigram-fusion-persistent-depth{depth}-node{budget}-wide-plus4": (
        budget,
        depth,
    )
    for depth, budget in (
        (7, 63),
        (8, 55),
        (8, 63),
        (10, 47),
        (10, 55),
        (10, 63),
        (10, 79),
        (10, 95),
    )
}

PERSISTENT_ADAPTIVE_DEPTH_POLICIES = {
    "context-score-trigram-fusion-persistent-adaptive95-depth10-wide-plus4"
}

PERSISTENT_OPTIMIZED_DEPTH_CONFIGS = {
    "context-score-trigram-fusion-persistent-contextcal-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-shadow-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-contextnodes-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-contextnodes-hotpath-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-contextnodes-hotpath-cpp-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-contextnodes95-hotpath-cpp-"
    "depth10-node95-wide-plus4": (95, 10),
    "context-score-trigram-fusion-persistent-hotpath-cpp-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-hotpath-cpp-"
    "depth10-node79-wide-plus4": (79, 10),
    "context-score-trigram-fusion-persistent-hotpath-cpp-"
    "depth10-node95-wide-plus4": (95, 10),
    "context-score-trigram-fusion-persistent-empirical-hotpath-cpp-"
    "depth14-node63-wide-plus4": (63, 14),
    "context-score-trigram-fusion-persistent-suffix4-hotpath-cpp-"
    "depth10-node63-wide-plus4": (63, 10),
    "context-score-trigram-fusion-persistent-suffix4-visualcache-"
    "hotpath-cpp-depth10-node63-wide-plus4": (63, 10),
}

MATCHED_BUDGET_CONTROL_CONFIGS = {
    "fixed-depth10-node63-wide8": (8, 10, 63),
    "score-prior-depth10-node63-wide8": (8, 10, 63),
    "context-score-trigram-strict-depth10-node63-wide8": (8, 10, 63),
    "context-score-trigram-fusion-uniform-depth10-node63-wide8": (
        8,
        10,
        63,
    ),
    "context-score-trigram-fusion-depth10-node63-wide8": (8, 10, 63),
    "context-score-trigram-fusion-persistent-depth10-node63-wide8": (
        8,
        10,
        63,
    ),
}
MATCHED_BUDGET_CONTROL_CONFIGS.update(
    {
        f"{allocator}-depth10-node{budget}-wide8": (8, 10, budget)
        for allocator in ("fixed", "score-prior")
        for budget in (31, 47, 63, 79, 95)
    }
)

CONTEXT_PLUS4_SPECIAL_POLICIES = frozenset(
    TRIGRAM_PLUS4_NODE_BUDGETS
) | frozenset(PERSISTENT_FUSION_NODE_BUDGETS) | frozenset(
    PERSISTENT_COMMITTED_POLICIES
) | frozenset(PERSISTENT_STABLE_NODE_BUDGETS)
CONTEXT_PLUS4_SPECIAL_POLICIES |= frozenset(
    PERSISTENT_DEPTH_NODE_CONFIGS
)
CONTEXT_PLUS4_SPECIAL_POLICIES |= frozenset(
    PERSISTENT_ADAPTIVE_DEPTH_POLICIES
)
CONTEXT_PLUS4_SPECIAL_POLICIES |= frozenset(
    PERSISTENT_OPTIMIZED_DEPTH_CONFIGS
)
CONTEXT_PLUS4_SPECIAL_POLICIES |= frozenset(
    MATCHED_BUDGET_CONTROL_CONFIGS
)

GLOBAL_BACKOFF_POLICIES = {
    "context-score-trigram-fusion-global7-deepest-wide-plus4": (7, 1),
    "context-score-trigram-fusion-global15-deepest-wide-plus4": (15, 2),
    "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4": (
        15,
        2,
    ),
    "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4": (
        15,
        2,
    ),
    "context-score-trigram-fusion-bank-global15-deepest-wide-plus4": (15, 2),
}

PERSISTENT_NGRAM_ROW_LIMIT = 65536


class TreeRecyclingSpecModel(_GroundedSamSpecModel):
    """Full-tree token recycling with multimodal width/depth allocation."""

    def reset_persistent_recycling_cache(self) -> None:
        """Clear request-spanning proposal state after evaluator warmup."""

        persistent_caches = getattr(
            self, "_gwtr_persistent_unigram_caches", None
        )
        if persistent_caches is not None:
            persistent_caches.clear()
        persistent_banks = getattr(self, "_gwtr_persistent_unigram_banks", None)
        if persistent_banks is not None:
            persistent_banks.clear()
        self._gwtr_visual_feature_cache = None

    @staticmethod
    def _select_prompt_transition_rows(
        prompt_values,
        cached_tokens=None,
    ):
        """Return first occurrences, optionally excluding persistent hits."""

        cached = cached_tokens if cached_tokens is not None else ()
        seen = set()
        indices = []
        values = []
        for prompt_index, token in enumerate(prompt_values):
            token = int(token)
            if token in seen or token in cached:
                continue
            seen.add(token)
            indices.append(prompt_index)
            values.append(token)
        return indices, values

    @staticmethod
    def _layerwise_verification_diagnostics(
        packed_output,
        single_output,
        packed_index: int = 0,
        single_index: int = 0,
    ) -> dict:
        """Compare packed and q_len=1 states with one compact host transfer."""

        packed_layers = getattr(packed_output, "hidden_states", None)
        single_layers = getattr(single_output, "hidden_states", None)
        if not packed_layers or not single_layers:
            raise ValueError(
                "layer diagnostics require output_hidden_states=True"
            )
        if len(packed_layers) != len(single_layers):
            raise ValueError("packed and single outputs have different layer counts")

        packed = torch.stack(
            [layer[0, packed_index] for layer in packed_layers]
        ).float()
        single = torch.stack(
            [layer[0, single_index] for layer in single_layers]
        ).float()
        difference = packed - single
        absolute = difference.abs()
        packed_norm = torch.linalg.vector_norm(packed, dim=-1)
        single_norm = torch.linalg.vector_norm(single, dim=-1)
        cosine = (packed * single).sum(dim=-1) / (
            packed_norm * single_norm
        ).clamp_min(1e-12)
        layer_metrics = torch.stack(
            [
                absolute.amax(dim=-1),
                absolute.mean(dim=-1),
                difference.square().mean(dim=-1).sqrt(),
                difference.square().mean(dim=-1).sqrt()
                / single.square().mean(dim=-1).sqrt().clamp_min(1e-12),
                cosine,
            ],
            dim=-1,
        ).cpu().tolist()
        layers = [
            {
                "layer": layer_index,
                "max_abs": float(values[0]),
                "mean_abs": float(values[1]),
                "rms": float(values[2]),
                "relative_rms": float(values[3]),
                "cosine": float(values[4]),
            }
            for layer_index, values in enumerate(layer_metrics)
        ]

        packed_logits = packed_output.logits[0, packed_index].float()
        single_logits = single_output.logits[0, single_index].float()
        logit_difference = packed_logits - single_logits
        packed_top_values, packed_top_ids = torch.topk(packed_logits, k=2)
        single_top_values, single_top_ids = torch.topk(single_logits, k=2)
        logit_metrics = torch.stack(
            [
                logit_difference.abs().amax(),
                logit_difference.abs().mean(),
                logit_difference.square().mean().sqrt(),
            ]
        ).cpu().tolist()
        top_values = torch.cat(
            [packed_top_values, single_top_values]
        ).cpu().tolist()
        top_ids = torch.cat([packed_top_ids, single_top_ids]).cpu().tolist()
        first_different_layer = next(
            (row["layer"] for row in layers if row["max_abs"] > 0.0),
            None,
        )
        return {
            "num_hidden_states": len(layers),
            "first_different_layer": first_different_layer,
            "layers": layers,
            "logits": {
                "max_abs": float(logit_metrics[0]),
                "mean_abs": float(logit_metrics[1]),
                "rms": float(logit_metrics[2]),
                "packed_top_ids": [int(top_ids[0]), int(top_ids[1])],
                "single_top_ids": [int(top_ids[2]), int(top_ids[3])],
                "packed_top_values": [
                    float(top_values[0]),
                    float(top_values[1]),
                ],
                "single_top_values": [
                    float(top_values[2]),
                    float(top_values[3]),
                ],
                "same_argmax": bool(top_ids[0] == top_ids[2]),
            },
        }

    def _score_state(
        self,
        output,
        query_index: int,
        calibrator: Optional[VisualGroundingCalibrator],
        grounding_layer: int,
        confidence_margin_scale: float,
        need_grounding: bool = True,
        need_confidence: bool = True,
    ) -> Tuple[float, float]:
        query_index = max(0, min(int(query_index), output.logits.shape[1] - 1))
        hidden = (
            self._layer_hidden(output, grounding_layer)
            if need_grounding
            else None
        )
        score = (
            calibrator.score(hidden[0, query_index])
            if need_grounding and calibrator is not None and hidden is not None
            else 0.0
        )
        confidence = (
            confidence_from_logits(
                output.logits[0, query_index], confidence_margin_scale
            )
            if need_confidence
            else 1.0
        )
        return score, confidence

    @torch.no_grad()
    def _target_only_from_prefill(
        self,
        *,
        input_ids: torch.Tensor,
        prefill_output,
        past_key_values,
        current_length_data: torch.Tensor,
        prompt_length: int,
        max_new_tokens: int,
        max_length: int,
        stop_token_ids,
        log: bool,
        return_acceptance_len: bool,
        return_decode_time: bool,
        return_policy_trace: bool,
    ):
        """Continue a completed multimodal prefill with plain greedy KV decode.

        The former ``target`` policy still built recycling rows and empty trees
        on every token.  This path is the fair non-speculative baseline: one
        argmax and one cached target forward per generated token.
        """

        acceptance_lengths: List[int] = []
        trace = []
        generated = 0
        idx = -1
        if max_new_tokens > 0 and input_ids.shape[1] < max_length:
            next_id = torch.argmax(prefill_output.logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)
            generated = 1
            idx = 0
            current_length_data.fill_(input_ids.shape[1] - 1)

        while generated < max_new_tokens and input_ids.shape[1] < max_length:
            if _has_stop_token(
                input_ids[0, prompt_length:], stop_token_ids
            ):
                break
            remaining = int(max_new_tokens - generated)
            output = self.base_model(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            next_id = torch.argmax(output.logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)
            generated += 1
            idx += 1
            current_length_data.fill_(input_ids.shape[1] - 1)
            acceptance_lengths.append(0)
            if return_policy_trace:
                trace.append(
                    {
                        "iteration": len(trace),
                        "budget": 0,
                        "risk": 1.0,
                        "grounding_score": 0.0,
                        "confidence": 1.0,
                        "acceptance_ema": 1.0,
                        "raw_draft_len": 0,
                        "used_draft_len": 0,
                        "verified_tree_nodes": 0,
                        "remaining_tokens": remaining,
                        "accept_len": 0,
                        "accept_ratio": 0.0,
                        "next_grounding_score": 0.0,
                        "next_confidence": 1.0,
                        "acceptance_ema_after": 1.0,
                    }
                )

        generated_ids = input_ids[0, prompt_length:]
        keep_tokens = min(int(generated_ids.numel()), int(max_new_tokens))
        for token_index, token_id in enumerate(
            generated_ids[:keep_tokens].tolist()
        ):
            if int(token_id) in stop_token_ids:
                keep_tokens = token_index + 1
                break
        input_ids = input_ids[:, : prompt_length + keep_tokens]
        outputs = (input_ids,)
        if log:
            outputs += (keep_tokens, max(idx, 0))
        if return_acceptance_len:
            outputs += (acceptance_lengths,)
        if return_decode_time:
            outputs += (0.0,)
        if return_policy_trace:
            outputs += (trace,)
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def _tree_shape(
        policy: str,
        grounding_score: float,
        confidence: float,
        risk: float,
        visual_threshold: float,
        confidence_threshold: float,
        fixed_width: int,
        fixed_depth: int,
        broad_width: int,
        shallow_depth: int,
    ) -> Tuple[int, int]:
        if policy == "target":
            return 0, 0
        if policy in (
            "broad",
            "modal-broad",
            "grounded-backoff",
            "grounded-backoff-reverse",
            "visual-lexical-backoff",
            "visual-hst-backoff",
            "visual-hst-backoff-gated",
        ):
            return broad_width, shallow_depth
        if policy in PERSISTENT_DEPTH_NODE_CONFIGS:
            return (
                broad_width + 4,
                PERSISTENT_DEPTH_NODE_CONFIGS[policy][1],
            )
        if policy in PERSISTENT_ADAPTIVE_DEPTH_POLICIES:
            return broad_width + 4, 10
        if policy in PERSISTENT_OPTIMIZED_DEPTH_CONFIGS:
            return (
                broad_width + 4,
                PERSISTENT_OPTIMIZED_DEPTH_CONFIGS[policy][1],
            )
        if policy in MATCHED_BUDGET_CONTROL_CONFIGS:
            width, depth, _ = MATCHED_BUDGET_CONTROL_CONFIGS[policy]
            return width, depth
        context_width_augmentation = {
            "context-score-prior-deeper-wide-plus4": 4,
            "context-score-trigram-deeper-wide-plus4": 4,
            "context-score-trigram-residual2-deeper-wide-plus4": 4,
            "context-score-trigram-residual2-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-deeper-wide-plus4": 4,
            "context-score-trigram-fusion-deepest-wide-plus4": 4,
            "context-score-trigram-fusion55-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-adaptive-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-calibrated-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-bank-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-bank-global15-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-global7-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-global15-deepest-wide-plus4": 4,
            "context-score-trigram-deepest-wide-plus4": 4,
            "context-score-trigram-deeper-wide-plus6": 6,
            "context-score-trigram-deeper-wide-plus8": 8,
            "context-score-fourgram-deeper-wide-plus4": 4,
            "context-score-fourgram-deepest-wide-plus4": 4,
            "context-score-prior-deeper-wide-plus5": 5,
            "context-score-prior-deeper-wide-plus6": 6,
            "context-score-prior-deeper-wide-plus8": 8,
        }.get(policy)
        if policy in CONTEXT_PLUS4_SPECIAL_POLICIES:
            context_width_augmentation = 4
        if context_width_augmentation is not None:
            depth_augmentation = (
                3 if "-deepest-wide-" in policy else 2
            )
            return (
                broad_width + context_width_augmentation,
                shallow_depth + depth_augmentation,
            )
        if policy in (
            "wide-plus2",
            "rank-prior-wide-plus2",
            "rank-prior-deep-wide-plus2",
            "rank-prior-deeper-wide-plus2",
            "rank-prior-deepest-wide-plus2",
            "score-prior-deep-wide-plus2",
            "score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2",
            "context-score-calibrated-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2-node55",
            "context-score-prior-deeper-wide-plus2-node47",
            "context-score-adaptive-safe-deeper-wide-plus2",
            "score-prior-deepest-wide-plus2",
            "score-prior-maxdeep-wide-plus2",
            "score-adaptive-safe-deeper-wide-plus2",
            "score-adaptive-deeper-wide-plus2",
        ):
            if policy in (
                "rank-prior-deep-wide-plus2",
                "score-prior-deep-wide-plus2",
            ):
                return broad_width + 2, shallow_depth + 1
            if policy in (
                "rank-prior-deeper-wide-plus2",
                "score-prior-deeper-wide-plus2",
                "context-score-prior-deeper-wide-plus2",
                "context-score-calibrated-deeper-wide-plus2",
                "context-score-prior-deeper-wide-plus2-node55",
                "context-score-prior-deeper-wide-plus2-node47",
                "context-score-adaptive-safe-deeper-wide-plus2",
                "score-adaptive-safe-deeper-wide-plus2",
                "score-adaptive-deeper-wide-plus2",
            ):
                return broad_width + 2, shallow_depth + 2
            if policy == "rank-prior-deepest-wide-plus2":
                return broad_width + 2, shallow_depth + 3
            if policy == "score-prior-deepest-wide-plus2":
                return broad_width + 2, shallow_depth + 3
            if policy == "score-prior-maxdeep-wide-plus2":
                return broad_width + 2, shallow_depth + 7
            return broad_width + 2, shallow_depth
        if policy in (
            "visual-wide-plus2",
            "visual-wide-plus2-reverse",
            "visual-rootwide-plus2",
            "visual-rootwide-plus2-hst-backoff",
            "visual-wide-plus2-vli-backoff",
            "visual-wide-plus2-hst-backoff",
            "visual-wide-plus2-hst-backoff-gated",
        ):
            augment = grounding_score >= visual_threshold
            if policy == "visual-wide-plus2-reverse":
                augment = not augment
            return broad_width + (2 if augment else 0), shallow_depth
        if policy in ("grounded-residual", "grounded-residual-reverse"):
            augment = grounding_score >= visual_threshold
            if policy == "grounded-residual-reverse":
                augment = not augment
            return broad_width + (2 if augment else 0), shallow_depth
        if policy == "narrow":
            return fixed_width, shallow_depth
        if policy == "spine":
            return broad_width, fixed_depth
        if policy in ("hybrid", "modal-hybrid"):
            return broad_width, fixed_depth
        if policy == "visual-hybrid":
            if grounding_score >= visual_threshold:
                return broad_width, shallow_depth
            return broad_width, fixed_depth
        if policy in ("grounded-hybrid", "grounded-hybrid-reverse"):
            use_broad_root = grounding_score >= visual_threshold
            if policy == "grounded-hybrid-reverse":
                use_broad_root = not use_broad_root
            return (
                broad_width if use_broad_root else fixed_width,
                fixed_depth,
            )
        if policy == "short":
            return 1, min(2, fixed_depth)
        if policy in ("visual-hard", "modal-hard"):
            if grounding_score >= visual_threshold:
                return broad_width, shallow_depth
            return fixed_width, fixed_depth
        if policy == "visual-anchor":
            if (
                grounding_score >= visual_threshold
                and confidence <= confidence_threshold
            ):
                return broad_width, shallow_depth
            return fixed_width, fixed_depth
        if policy == "visual-reverse":
            if grounding_score < visual_threshold:
                return broad_width, shallow_depth
            return fixed_width, fixed_depth
        if policy == "visual-width":
            if grounding_score >= visual_threshold:
                return broad_width, shallow_depth
            return fixed_width, shallow_depth
        if policy == "visual-width-reverse":
            if grounding_score < visual_threshold:
                return broad_width, shallow_depth
            return fixed_width, shallow_depth
        if policy in ("visual-spine", "visual-spine-reverse", "modal-spine"):
            use_spine = grounding_score >= visual_threshold
            if policy == "visual-spine-reverse":
                use_spine = not use_spine
            if use_spine:
                return broad_width, fixed_depth
            return fixed_width, fixed_depth
        if policy in ("visual-soft", "visual-accept"):
            width = int(round(fixed_width + (broad_width - fixed_width) * risk))
            depth = int(round(fixed_depth + (shallow_depth - fixed_depth) * risk))
            return max(width, 1), max(depth, 1)
        return fixed_width, fixed_depth

    @staticmethod
    def _effective_tree_node_budget(
        policy: str,
        configured_budget: int,
        root_transition_top_probability: Optional[float],
        root_transition_context_order: int = 0,
    ) -> int:
        """Choose a verifier batch budget without changing tree ranking."""

        budget = int(configured_budget)
        if policy in TRIGRAM_PLUS4_NODE_BUDGETS:
            return min(budget, TRIGRAM_PLUS4_NODE_BUDGETS[policy])
        if policy in PERSISTENT_FUSION_NODE_BUDGETS:
            return min(budget, PERSISTENT_FUSION_NODE_BUDGETS[policy])
        if policy in PERSISTENT_STABLE_NODE_BUDGETS:
            return min(budget, PERSISTENT_STABLE_NODE_BUDGETS[policy])
        if policy in PERSISTENT_DEPTH_NODE_CONFIGS:
            return min(budget, PERSISTENT_DEPTH_NODE_CONFIGS[policy][0])
        if policy in PERSISTENT_ADAPTIVE_DEPTH_POLICIES:
            if (
                root_transition_top_probability is not None
                and 0.4 <= root_transition_top_probability < 0.7
            ):
                return min(budget, 95)
            return min(budget, 63)
        if "-contextnodes95-" in policy:
            context_order = int(root_transition_context_order)
            selected_budget = 47 if context_order <= 1 else (
                63 if context_order == 2 else 95
            )
            return min(budget, selected_budget)
        if "-contextnodes-" in policy:
            return min(
                budget,
                47 if int(root_transition_context_order) <= 1 else 63,
            )
        if policy in PERSISTENT_OPTIMIZED_DEPTH_CONFIGS:
            return min(
                budget, PERSISTENT_OPTIMIZED_DEPTH_CONFIGS[policy][0]
            )
        if policy in MATCHED_BUDGET_CONTROL_CONFIGS:
            return min(budget, MATCHED_BUDGET_CONTROL_CONFIGS[policy][2])
        if policy == "context-score-prior-deeper-wide-plus2-node55":
            return min(budget, 55)
        if policy == "context-score-prior-deeper-wide-plus2-node47":
            return min(budget, 47)
        if (
            policy == "context-score-adaptive-safe-deeper-wide-plus2"
            and root_transition_top_probability is not None
            and root_transition_top_probability >= 0.85
        ):
            return min(budget, 47)
        if (
            policy == "score-adaptive-safe-deeper-wide-plus2"
            and root_transition_top_probability is not None
            and root_transition_top_probability >= 0.85
        ):
            return min(budget, 47)
        if (
            policy == "score-adaptive-deeper-wide-plus2"
            and root_transition_top_probability is not None
        ):
            if root_transition_top_probability >= 0.90:
                return min(budget, 31)
            if root_transition_top_probability >= 0.75:
                return min(budget, 47)
        return budget

    @staticmethod
    def _score_priority_hit_masses(
        policy: str,
        root_transition_context_order: int,
    ):
        """Calibrate depth allocation while leaving candidate scores intact."""

        if policy == (
            "context-score-trigram-fusion-calibrated-"
            "deepest-wide-plus4"
        ):
            return (0.70, 0.67, 0.59, 0.49, 0.38, 0.39)
        if policy == "context-score-calibrated-deeper-wide-plus2":
            return (0.66, 0.65, 0.62, 0.54, 0.38, 0.36)
        context_calibrated = policy == (
            "context-score-trigram-fusion-persistent-contextcal-"
            "depth10-node63-wide-plus4"
        )
        empirical_deep = policy == (
            "context-score-trigram-fusion-persistent-empirical-hotpath-cpp-"
            "depth14-node63-wide-plus4"
        )
        if not context_calibrated and not empirical_deep:
            return None

        context_order = int(root_transition_context_order)
        if context_order >= 3:
            masses = (
                0.78,
                0.76,
                0.68,
                0.62,
                0.56,
                0.50,
                0.44,
                0.38,
                0.34,
                0.30,
            )
        elif context_order == 2:
            masses = (
                0.74,
                0.72,
                0.64,
                0.57,
                0.51,
                0.45,
                0.40,
                0.35,
                0.31,
                0.28,
            )
        else:
            masses = (
                0.68,
                0.69,
                0.60,
                0.54,
                0.48,
                0.43,
                0.38,
                0.34,
                0.30,
                0.27,
            )
        if empirical_deep:
            return (*masses, 0.25, 0.23, 0.21, 0.19)
        return masses

    @staticmethod
    def _merge_persistent_transition_row(
        old_candidates,
        old_scores,
        old_count: int,
        new_candidates,
        new_scores,
        limit: int,
    ):
        """Maintain a bounded running-average proposal row across requests."""

        effective_old_count = min(max(int(old_count), 0), 7)
        fused_scores = {}
        first_seen = {}
        for weight, candidates, scores in (
            (effective_old_count, old_candidates or (), old_scores or ()),
            (1, new_candidates or (), new_scores or ()),
        ):
            for candidate, score in zip(candidates, scores):
                candidate = int(candidate)
                fused_scores[candidate] = fused_scores.get(candidate, 0.0) + (
                    float(weight) * float(score)
                )
                first_seen.setdefault(candidate, len(first_seen))
        selected = sorted(
            fused_scores,
            key=lambda candidate: (
                -fused_scores[candidate],
                first_seen[candidate],
            ),
        )[: max(int(limit), 1)]
        total = sum(fused_scores[candidate] for candidate in selected) or 1.0
        return (
            selected,
            [fused_scores[candidate] / total for candidate in selected],
            min(effective_old_count + 1, 8),
        )

    @staticmethod
    def _store_bounded_transition_row(
        rows,
        score_rows,
        key,
        candidates,
        scores,
        limit: int,
    ) -> None:
        """Update an insertion-ordered row table with bounded LRU-like size."""

        if key in rows:
            rows.pop(key)
            if score_rows is not None:
                score_rows.pop(key, None)
        elif len(rows) >= max(int(limit), 1):
            oldest_key = next(iter(rows))
            rows.pop(oldest_key)
            if score_rows is not None:
                score_rows.pop(oldest_key, None)
        rows[key] = candidates
        if score_rows is not None and scores is not None:
            score_rows[key] = scores

    @staticmethod
    def _tree_branch_width(
        policy: str,
        grounding_score: float,
        visual_threshold: float,
        root_width: int,
        fixed_width: int,
    ) -> int:
        """Allocate visual ambiguity to the root without multiplying it at depth two.

        The wide-root policies retain the six-way first-hop coverage that was
        beneficial for grounded states, but use the narrow configured width for
        each child.  Low-grounding states retain the ordinary broad tree.
        """

        if (
            policy
            in ("visual-rootwide-plus2", "visual-rootwide-plus2-hst-backoff")
            and grounding_score >= visual_threshold
        ):
            return min(max(int(fixed_width), 1), max(int(root_width), 0))
        if policy == "context-score-prior-deeper-wide-plus6":
            return min(8, max(int(root_width), 0))
        if policy == "context-score-prior-deeper-wide-plus5":
            return min(8, max(int(root_width), 0))
        if policy == "context-score-prior-deeper-wide-plus8":
            return min(6, max(int(root_width), 0))
        return max(int(root_width), 0)

    @staticmethod
    def _visual_lexical_backoff_active(
        policy: str,
        grounding_score: float,
        visual_threshold: float,
        root_row_valid: bool,
    ) -> bool:
        """Route prompt-native visual candidates only to unseen grounded roots."""

        return bool(
            policy
            in (
                "visual-lexical-backoff",
                "visual-hst-backoff",
                "visual-hst-backoff-gated",
                "visual-wide-plus2-hst-backoff",
                "visual-wide-plus2-hst-backoff-gated",
                "visual-rootwide-plus2-hst-backoff",
                "visual-wide-plus2-vli-backoff",
            )
            and not root_row_valid
            and grounding_score >= visual_threshold
        )

    @staticmethod
    def _transition_candidates(
        transitions: torch.Tensor,
        transition_valid: torch.Tensor,
        parent_token: int,
        primary_bin: int,
        limit: int,
        fallback_bin: Optional[int] = None,
        transition_counts: Optional[torch.Tensor] = None,
        max_conditional_weight: float = 0.5,
        preserve_fallback_limit: int = 0,
        min_conditional_count: int = 2,
    ) -> List[int]:
        """Fuse a conditional row with a reliable global fallback.

        The conditional contribution grows with the number of observations but
        is capped so an immature modality-specific row cannot erase the global
        Token Recycling candidates. Reciprocal-rank fusion rewards agreement
        between the two rows while allowing conditional residual candidates to
        replace only the global tail.
        """
        limit = max(int(limit), 0)
        if limit == 0:
            return []

        primary_bin = int(primary_bin)
        primary_valid = bool(
            transition_valid[primary_bin, parent_token].item()
        )
        if not primary_valid and fallback_bin is None:
            available = torch.nonzero(
                transition_valid[:, parent_token], as_tuple=True
            )[0]
            if available.numel() == 0:
                return []
            primary_bin = int(available[0].item())
            primary_valid = True

        fallback_valid = (
            fallback_bin is not None
            and bool(transition_valid[int(fallback_bin), parent_token].item())
        )
        if not primary_valid:
            if not fallback_valid:
                return []
            fallback_limit = (
                min(limit, int(preserve_fallback_limit))
                if preserve_fallback_limit > 0
                else limit
            )
            return [
                int(token)
                for token in transitions[
                    int(fallback_bin), parent_token, :fallback_limit
                ].tolist()
            ]
        if (
            fallback_bin is None
            or int(fallback_bin) == primary_bin
            or not fallback_valid
        ):
            return [
                int(token)
                for token in transitions[primary_bin, parent_token, :limit].tolist()
            ]

        conditional_count = 1.0
        if transition_counts is not None:
            conditional_count = float(
                transition_counts[primary_bin, parent_token].item()
            )
        if preserve_fallback_limit > 0:
            preserved = [
                int(token)
                for token in transitions[
                    int(fallback_bin),
                    parent_token,
                    : min(limit, int(preserve_fallback_limit)),
                ].tolist()
            ]
            if conditional_count < max(int(min_conditional_count), 1):
                return preserved
            seen = set(preserved)
            for token in transitions[primary_bin, parent_token].tolist():
                if len(preserved) >= limit:
                    break
                token = int(token)
                if token in seen:
                    continue
                preserved.append(token)
                seen.add(token)
            return preserved

        conditional_weight = min(
            max(float(max_conditional_weight), 0.0),
            conditional_count / (conditional_count + 2.0),
        )
        global_weight = 1.0 - conditional_weight
        scores = {}
        first_seen = {}
        seen_index = 0
        for row_bin, weight in (
            (int(fallback_bin), global_weight),
            (primary_bin, conditional_weight),
        ):
            for rank, token in enumerate(
                transitions[row_bin, parent_token].tolist(), start=1
            ):
                token = int(token)
                scores[token] = scores.get(token, 0.0) + weight / rank
                if token not in first_seen:
                    first_seen[token] = seen_index
                    seen_index += 1
        ranked = sorted(scores, key=lambda token: (-scores[token], first_seen[token]))
        return ranked[:limit]

    @staticmethod
    def _build_tree(
        root_token: int,
        transitions: torch.Tensor,
        transition_valid: torch.Tensor,
        width: int,
        depth: int,
        node_budget: int,
        blocked_token_id: Optional[int],
        branch_width: Optional[int] = None,
        transition_bin: int = 0,
        fallback_transition_bin: Optional[int] = None,
        transition_counts: Optional[torch.Tensor] = None,
        max_conditional_weight: float = 0.5,
        preserve_fallback_limit: int = 0,
        min_conditional_count: int = 2,
        topology_cache=None,
        host_transitions=None,
        host_transition_scores=None,
        root_previous_token=None,
        root_previous_previous_token=None,
        root_previous_previous_previous_token=None,
        host_context_transitions=None,
        host_context_transition_scores=None,
        host_trigram_transitions=None,
        host_trigram_transition_scores=None,
        host_fourgram_transitions=None,
        host_fourgram_transition_scores=None,
        host_persistent_transitions=None,
        host_persistent_transition_scores=None,
        host_global_candidates=None,
        host_global_candidate_scores=None,
        context_candidate_mode: str = "strict",
        metadata_out=None,
        priority_layout: bool = False,
        score_priority_layout: bool = False,
        score_hit_masses=None,
        copy_candidate_rows: bool = True,
        fast_score_priority_layout: bool = False,
        reserved_path_tokens=None,
    ):
        # Node indices in the returned flat sequence start at one; zero is the
        # already-generated root token that has not yet entered the KV cache.
        nodes = []

        def candidate_distribution_for(
            parent_token,
            level_width,
            previous_token=None,
            previous_previous_token=None,
            previous_previous_previous_token=None,
        ):
            def normalized(candidate_row, score_row):
                candidate_count = min(
                    len(candidate_row), int(level_width)
                )
                selected_candidates = (
                    list(candidate_row[:candidate_count])
                    if copy_candidate_rows
                    else (
                        candidate_row
                        if candidate_count == len(candidate_row)
                        else candidate_row[:candidate_count]
                    )
                )
                selected_scores = (
                    list(score_row[:candidate_count])
                    if copy_candidate_rows
                    else (
                        score_row
                        if len(score_row) == candidate_count
                        else score_row[:candidate_count]
                    )
                )
                if len(selected_scores) < len(selected_candidates):
                    reciprocal = [
                        1.0 / rank
                        for rank in range(
                            1, len(selected_candidates) + 1
                        )
                    ]
                    total = sum(reciprocal) or 1.0
                    selected_scores = [
                        value / total for value in reciprocal
                    ]
                return selected_candidates, selected_scores

            fourgram_key = (
                (
                    int(previous_previous_previous_token),
                    int(previous_previous_token),
                    int(previous_token),
                    int(parent_token),
                )
                if previous_previous_previous_token is not None
                and previous_previous_token is not None
                and previous_token is not None
                else None
            )
            rows = []
            if (
                host_fourgram_transitions is not None
                and fourgram_key in host_fourgram_transitions
            ):
                rows.append(
                    normalized(
                        host_fourgram_transitions[fourgram_key],
                        host_fourgram_transition_scores.get(fourgram_key, ())
                        if host_fourgram_transition_scores is not None
                        else (),
                    )
                )
            trigram_key = (
                (
                    int(previous_previous_token),
                    int(previous_token),
                    int(parent_token),
                )
                if previous_previous_token is not None
                and previous_token is not None
                else None
            )
            if (
                host_trigram_transitions is not None
                and trigram_key in host_trigram_transitions
            ):
                rows.append(
                    normalized(
                        host_trigram_transitions[trigram_key],
                        host_trigram_transition_scores.get(trigram_key, ())
                        if host_trigram_transition_scores is not None
                        else (),
                    )
                )
            context_key = (
                (int(previous_token), int(parent_token))
                if previous_token is not None
                else None
            )
            if (
                host_context_transitions is not None
                and context_key in host_context_transitions
            ):
                rows.append(
                    normalized(
                        host_context_transitions[context_key],
                        host_context_transition_scores.get(context_key, ())
                        if host_context_transition_scores is not None
                        else (),
                    )
                )
            if host_transitions is not None:
                if parent_token in host_transitions:
                    rows.append(
                        normalized(
                            host_transitions[parent_token],
                            host_transition_scores.get(parent_token, ())
                            if host_transition_scores is not None
                            else (),
                        )
                    )
                if (
                    host_persistent_transitions is not None
                    and parent_token in host_persistent_transitions
                ):
                    rows.append(
                        normalized(
                            host_persistent_transitions[parent_token],
                            host_persistent_transition_scores.get(
                                parent_token, ()
                            )
                            if host_persistent_transition_scores is not None
                            else (),
                        )
                    )
                if not rows:
                    if host_global_candidates:
                        return normalized(
                            host_global_candidates,
                            host_global_candidate_scores or (),
                        )
                    return [], []
                if context_candidate_mode == "strict" or len(rows) == 1:
                    return rows[0]
                if context_candidate_mode == "residual2":
                    primary_candidates, primary_scores = rows[0]
                    keep = max(int(level_width) - 2, 1)
                    selected = list(primary_candidates[:keep])
                    selected_scores = list(primary_scores[:keep])
                    seen = set(selected)
                    for fallback_candidates, fallback_scores in rows[1:]:
                        for candidate, score in zip(
                            fallback_candidates, fallback_scores
                        ):
                            candidate = int(candidate)
                            if candidate in seen:
                                continue
                            selected.append(candidate)
                            selected_scores.append(0.5 * float(score))
                            seen.add(candidate)
                            if len(selected) >= level_width:
                                break
                        if len(selected) >= level_width:
                            break
                    for candidate, score in zip(
                        primary_candidates[keep:], primary_scores[keep:]
                    ):
                        if len(selected) >= level_width:
                            break
                        candidate = int(candidate)
                        if candidate in seen:
                            continue
                        selected.append(candidate)
                        selected_scores.append(float(score))
                        seen.add(candidate)
                    total = sum(selected_scores) or 1.0
                    return selected, [score / total for score in selected_scores]
                if context_candidate_mode.startswith("fusion"):
                    if context_candidate_mode == "fusion_uniform":
                        source_weights = (1.0, 1.0, 1.0, 1.0)
                    elif context_candidate_mode == "fusion55":
                        source_weights = (0.55, 0.30, 0.15, 0.05)
                    elif context_candidate_mode == "fusion_adaptive":
                        primary_top_probability = (
                            float(rows[0][1][0]) if rows[0][1] else 0.0
                        )
                        if primary_top_probability >= 0.75:
                            source_weights = (0.80, 0.15, 0.05, 0.025)
                        elif primary_top_probability >= 0.50:
                            source_weights = (0.65, 0.25, 0.10, 0.05)
                        else:
                            source_weights = (0.50, 0.30, 0.15, 0.05)
                    else:
                        source_weights = (0.70, 0.20, 0.10, 0.05)
                    fused_scores = {}
                    first_seen = {}
                    for source_index, (candidates, probabilities) in enumerate(rows):
                        weight = source_weights[
                            min(source_index, len(source_weights) - 1)
                        ]
                        for candidate, probability in zip(
                            candidates, probabilities
                        ):
                            candidate = int(candidate)
                            fused_scores[candidate] = fused_scores.get(
                                candidate, 0.0
                            ) + weight * float(probability)
                            first_seen.setdefault(candidate, len(first_seen))
                    selected = sorted(
                        fused_scores,
                        key=lambda candidate: (
                            -fused_scores[candidate],
                            first_seen[candidate],
                        ),
                    )[:level_width]
                    total = sum(fused_scores[candidate] for candidate in selected) or 1.0
                    return selected, [
                        fused_scores[candidate] / total for candidate in selected
                    ]
                raise ValueError(
                    f"unknown context candidate mode: {context_candidate_mode}"
                )
            candidates = TreeRecyclingSpecModel._transition_candidates(
                transitions,
                transition_valid,
                parent_token,
                transition_bin,
                level_width,
                fallback_bin=fallback_transition_bin,
                transition_counts=transition_counts,
                max_conditional_weight=max_conditional_weight,
                preserve_fallback_limit=preserve_fallback_limit,
                min_conditional_count=min_conditional_count,
            )
            return normalized(candidates, ())

        def candidates_for(
            parent_token,
            level_width,
            previous_token=None,
            previous_previous_token=None,
            previous_previous_previous_token=None,
        ):
            candidates, _ = candidate_distribution_for(
                parent_token,
                level_width,
                previous_token=previous_token,
                previous_previous_token=previous_previous_token,
                previous_previous_previous_token=(
                    previous_previous_previous_token
                ),
            )
            return candidates

        if score_priority_layout and fast_score_priority_layout:
            if (
                context_candidate_mode != "fusion"
                or host_fourgram_transitions
                or host_persistent_transitions
                or host_global_candidates
            ):
                raise ValueError(
                    "The C++ score-priority builder supports standard "
                    "trigram fusion without auxiliary candidate sources"
                )
            from method.sam_grounded.fast_tree_builder import (
                build_score_priority_nodes,
            )

            branch_width = width if branch_width is None else branch_width
            resolved_hit_masses = score_hit_masses or (
                0.68,
                0.63,
                0.50,
                0.43,
                0.40,
                0.38,
            )
            node_rows = build_score_priority_nodes(
                root_token=int(root_token),
                root_previous_token=(
                    int(root_previous_token)
                    if root_previous_token is not None
                    else -1
                ),
                root_previous_previous_token=(
                    int(root_previous_previous_token)
                    if root_previous_previous_token is not None
                    else -1
                ),
                width=int(width),
                branch_width=int(branch_width),
                depth=int(depth),
                node_budget=int(node_budget),
                blocked_token_id=int(blocked_token_id),
                unigram_rows=host_transitions or {},
                unigram_scores=host_transition_scores or {},
                context_rows=host_context_transitions or {},
                context_scores=host_context_transition_scores or {},
                trigram_rows=host_trigram_transitions or {},
                trigram_scores=host_trigram_transition_scores or {},
                hit_masses=list(resolved_hit_masses),
            )
            nodes = [
                {
                    "token": int(token),
                    "parent": int(parent),
                    "depth": int(node_depth),
                    "rank": int(rank),
                }
                for token, parent, node_depth, rank in node_rows
            ]
        elif score_priority_layout:
            branch_width = width if branch_width is None else branch_width
            hit_masses = score_hit_masses or (
                0.68,
                0.63,
                0.50,
                0.43,
                0.40,
                0.38,
            )
            selected = {
                (): {
                    "token": int(root_token),
                    "previous_token": root_previous_token,
                    "previous_previous_token": (
                        root_previous_previous_token
                    ),
                    "previous_previous_previous_token": (
                        root_previous_previous_previous_token
                    ),
                    "path_edges": frozenset(),
                }
            }
            selected_nodes = []
            candidate_cache = {}
            pending = []

            def push_children(parent_path, parent_score):
                current_depth = len(parent_path) + 1
                if current_depth > depth:
                    return
                parent = selected[parent_path]
                parent_token = int(parent["token"])
                previous_token = parent.get("previous_token")
                previous_previous_token = parent.get(
                    "previous_previous_token"
                )
                previous_previous_previous_token = parent.get(
                    "previous_previous_previous_token"
                )
                level_width = width if current_depth == 1 else branch_width
                cache_key = (
                    previous_previous_previous_token,
                    previous_previous_token,
                    previous_token,
                    parent_token,
                    int(level_width),
                )
                candidate_distribution = candidate_cache.get(cache_key)
                if candidate_distribution is None:
                    candidate_distribution = candidate_distribution_for(
                        parent_token,
                        level_width,
                        previous_token=previous_token,
                        previous_previous_token=previous_previous_token,
                        previous_previous_previous_token=(
                            previous_previous_previous_token
                        ),
                    )
                    candidate_cache[cache_key] = candidate_distribution
                candidates, probabilities = candidate_distribution
                hit_mass = hit_masses[
                    min(current_depth - 1, len(hit_masses) - 1)
                ]
                seen_candidates = set()
                for candidate_rank, candidate in enumerate(candidates, start=1):
                    candidate = int(candidate)
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    edge = (parent_token, candidate)
                    if (
                        candidate == blocked_token_id
                        or edge in parent["path_edges"]
                    ):
                        continue
                    rank_path = (*parent_path, candidate_rank)
                    score = (
                        parent_score
                        * hit_mass
                        * float(probabilities[candidate_rank - 1])
                    )
                    heapq.heappush(
                        pending,
                        (
                            -score,
                            rank_path,
                            candidate,
                            parent_path,
                            parent["path_edges"] | {edge},
                        ),
                    )

            push_children((), 1.0)
            while pending and len(selected_nodes) < node_budget:
                (
                    negative_score,
                    rank_path,
                    candidate,
                    parent_path,
                    path_edges,
                ) = heapq.heappop(pending)
                node = {
                    "token": candidate,
                    "depth": len(rank_path),
                    "rank": rank_path[-1],
                    "rank_path": rank_path,
                    "parent_path": parent_path,
                    "previous_token": int(selected[parent_path]["token"]),
                    "previous_previous_token": selected[parent_path].get(
                        "previous_token"
                    ),
                    "previous_previous_previous_token": selected[
                        parent_path
                    ].get("previous_previous_token"),
                    "path_edges": path_edges,
                }
                selected[rank_path] = node
                selected_nodes.append(node)
                push_children(rank_path, -negative_score)

            selected_nodes.sort(key=lambda node: node["rank_path"])
            flat_index_by_path = {
                node["rank_path"]: index
                for index, node in enumerate(selected_nodes, start=1)
            }
            nodes = [
                {
                    "token": node["token"],
                    "parent": flat_index_by_path.get(node["parent_path"], 0),
                    "depth": node["depth"],
                    "rank": node["rank"],
                }
                for node in selected_nodes
            ]
        elif priority_layout:
            # The rank diagnostics show a steeply concentrated recycling
            # distribution.  Select the most likely rank paths globally rather
            # than exhausting every second-hop sibling before allocating any
            # third-hop nodes.  The priors are conditional hit rates, so the
            # product estimates the value of adding one particular prefix.
            rank_priors = (
                (0.470, 0.090, 0.045, 0.040, 0.024, 0.016),
                (0.450, 0.080, 0.040, 0.026, 0.016, 0.014),
                (0.355, 0.080, 0.030, 0.015, 0.009, 0.007),
                (0.350, 0.060, 0.030, 0.015, 0.008, 0.006),
            )

            def rank_prior(current_depth, rank):
                row = rank_priors[min(current_depth - 1, len(rank_priors) - 1)]
                if rank <= len(row):
                    return row[rank - 1]
                return row[-1] * (len(row) / rank) ** 2

            branch_width = width if branch_width is None else branch_width
            rank_path_cache = getattr(
                TreeRecyclingSpecModel, "_priority_rank_path_cache", None
            )
            if rank_path_cache is None:
                rank_path_cache = {}
                TreeRecyclingSpecModel._priority_rank_path_cache = rank_path_cache
            rank_path_key = (int(width), int(branch_width), int(depth))
            ranked_paths = rank_path_cache.get(rank_path_key)
            if ranked_paths is None:
                scored_rank_paths = []

                def enumerate_rank_paths(prefix, score):
                    current_depth = len(prefix) + 1
                    if current_depth > depth:
                        return
                    level_width = (
                        width if current_depth == 1 else branch_width
                    )
                    for candidate_rank in range(1, level_width + 1):
                        rank_path = (*prefix, candidate_rank)
                        path_score = score * rank_prior(
                            current_depth, candidate_rank
                        )
                        scored_rank_paths.append((path_score, rank_path))
                        enumerate_rank_paths(rank_path, path_score)

                enumerate_rank_paths((), 1.0)
                scored_rank_paths.sort(key=lambda item: (-item[0], item[1]))
                ranked_paths = tuple(path for _, path in scored_rank_paths)
                rank_path_cache[rank_path_key] = ranked_paths
            selected = {
                (): {
                    "token": int(root_token),
                    "path_edges": frozenset(),
                }
            }
            selected_nodes = []
            candidate_cache = {}
            for rank_path in ranked_paths:
                if len(selected_nodes) >= node_budget:
                    break
                parent_path = rank_path[:-1]
                parent = selected.get(parent_path)
                if parent is None:
                    continue
                parent_token = int(parent["token"])
                level_width = width if len(rank_path) == 1 else branch_width
                cache_key = (parent_token, int(level_width))
                candidates = candidate_cache.get(cache_key)
                if candidates is None:
                    candidates = list(candidates_for(parent_token, level_width))
                    candidate_cache[cache_key] = candidates
                candidate_rank = rank_path[-1]
                if candidate_rank > len(candidates):
                    continue
                candidate = int(candidates[candidate_rank - 1])
                edge = (parent_token, candidate)
                path_edges = parent["path_edges"]
                if candidate == blocked_token_id or edge in path_edges:
                    continue
                node = {
                    "token": candidate,
                    "depth": len(rank_path),
                    "rank": candidate_rank,
                    "rank_path": rank_path,
                    "parent_path": parent_path,
                    "path_edges": path_edges | {edge},
                }
                selected[rank_path] = node
                selected_nodes.append(node)

            # Lexicographic rank paths form a stable depth-first preorder.  The
            # dominant rank-1 chain is therefore a contiguous KV prefix, which
            # skips cache compaction on the most common accepted paths.
            selected_nodes.sort(key=lambda node: node["rank_path"])
            flat_index_by_path = {
                node["rank_path"]: index
                for index, node in enumerate(selected_nodes, start=1)
            }
            nodes = [
                {
                    "token": node["token"],
                    "parent": flat_index_by_path.get(node["parent_path"], 0),
                    "depth": node["depth"],
                    "rank": node["rank"],
                }
                for node in selected_nodes
            ]
        else:
            frontier = [(root_token, 0, frozenset())]
            for current_depth in range(1, depth + 1):
                next_frontier = []
                level_width = width if current_depth == 1 else branch_width
                if level_width is None:
                    level_width = width
                for parent_token, parent_index, path_edges in frontier:
                    if len(nodes) >= node_budget:
                        break
                    seen_candidates = set()
                    candidates = candidates_for(parent_token, level_width)
                    for candidate_rank, candidate in enumerate(
                        candidates, start=1
                    ):
                        candidate = int(candidate)
                        if candidate in seen_candidates:
                            continue
                        seen_candidates.add(candidate)
                        edge = (parent_token, candidate)
                        if candidate == blocked_token_id or edge in path_edges:
                            continue
                        flat_index = len(nodes) + 1
                        nodes.append(
                            {
                                "token": candidate,
                                "parent": parent_index,
                                "depth": current_depth,
                                "rank": candidate_rank,
                            }
                        )
                        next_frontier.append(
                            (candidate, flat_index, path_edges | {edge})
                        )
                        if len(nodes) >= node_budget:
                            break
                frontier = next_frontier
                if not frontier or len(nodes) >= node_budget:
                    break

        if reserved_path_tokens:
            # A suffix-automaton match is a single high-value path rather than
            # another independent transition distribution.  Reuse any prefix
            # already selected by the score-priority tree and append only the
            # missing suffix nodes.  At most a handful of rows are added, so
            # the path improves coverage without displacing globally ranked
            # recycling candidates.
            children = {
                (int(node["parent"]), int(node["token"])): flat_index
                for flat_index, node in enumerate(nodes, start=1)
            }
            parent_index = 0
            parent_token = int(root_token)
            parent_depth = 0
            path_edges = set()
            for raw_token in reserved_path_tokens:
                token = int(raw_token)
                edge = (parent_token, token)
                node_depth = parent_depth + 1
                if (
                    token == blocked_token_id
                    or edge in path_edges
                    or node_depth > int(depth)
                ):
                    break
                child_index = children.get((parent_index, token))
                if child_index is None:
                    child_index = len(nodes) + 1
                    nodes.append(
                        {
                            "token": token,
                            "parent": parent_index,
                            "depth": node_depth,
                            "rank": 0,
                        }
                    )
                    children[(parent_index, token)] = child_index
                path_edges.add(edge)
                parent_index = child_index
                parent_token = token
                parent_depth = node_depth

        topology_key = (
            transitions.device.type,
            transitions.device.index,
            tuple((int(node["parent"]), int(node["depth"])) for node in nodes),
        )
        cached_topology = (
            topology_cache.get(topology_key)
            if topology_cache is not None
            else None
        )
        if cached_topology is not None:
            mask, positions, paths = cached_topology
        else:
            tree_len = len(nodes) + 1
            # These trees contain at most a few dozen nodes.  Build their
            # topology in host memory and submit two compact transfers instead
            # of launching a GPU scalar-write kernel per ancestor relation.
            mask_rows = [[False] * tree_len for _ in range(tree_len)]
            mask_rows[0][0] = True
            position_values = [0] * tree_len
            paths: List[List[int]] = []
            for flat_index, node in enumerate(nodes, start=1):
                mask_rows[flat_index][flat_index] = True
                mask_rows[flat_index][0] = True
                position_values[flat_index] = node["depth"]
                path = [flat_index]
                parent = node["parent"]
                while parent > 0:
                    mask_rows[flat_index][parent] = True
                    path.append(parent)
                    parent = nodes[parent - 1]["parent"]
                paths.append(list(reversed(path)))

            mask = torch.tensor(
                mask_rows, dtype=torch.bool, device=transitions.device
            )[None, None]
            positions = torch.tensor(
                position_values, dtype=torch.long, device=transitions.device
            )
            if topology_cache is not None:
                if len(topology_cache) >= 256:
                    topology_cache.clear()
                topology_cache[topology_key] = (mask, positions, paths)

        flat_tokens = [root_token] + [node["token"] for node in nodes]
        if metadata_out is not None:
            metadata_out["node_ranks"] = [0] + [
                int(node["rank"]) for node in nodes
            ]
            if metadata_out.get("_candidate_trace_diagnostics", False):
                metadata_out["node_parents"] = [
                    int(node["parent"]) for node in nodes
                ]
                metadata_out["node_depths"] = [
                    int(node["depth"]) for node in nodes
                ]
            semantic_previous_tokens = [
                root_previous_token,
                *[
                    int(root_token)
                    if int(node["parent"]) == 0
                    else int(nodes[int(node["parent"]) - 1]["token"])
                    for node in nodes
                ],
            ]
            metadata_out["semantic_previous_tokens"] = (
                semantic_previous_tokens
            )
            metadata_out["semantic_previous_previous_tokens"] = [
                root_previous_previous_token,
                *[
                    root_previous_token
                    if int(node["parent"]) == 0
                    else semantic_previous_tokens[int(node["parent"])]
                    for node in nodes
                ],
            ]
            semantic_previous_previous_tokens = metadata_out[
                "semantic_previous_previous_tokens"
            ]
            metadata_out["semantic_previous_previous_previous_tokens"] = [
                root_previous_previous_previous_token,
                *[
                    root_previous_previous_token
                    if int(node["parent"]) == 0
                    else semantic_previous_previous_tokens[
                        int(node["parent"])
                    ]
                    for node in nodes
                ],
            ]
        return flat_tokens, mask, positions, paths

    @staticmethod
    def _requires_cache_compaction(accepted_path: List[int]) -> bool:
        """Whether selected packed-tree rows are not already a contiguous prefix."""

        return any(
            int(node_index) != expected_index
            for expected_index, node_index in enumerate(
                accepted_path, start=1
            )
        )

    @staticmethod
    def _tree_path_token_ids(flat_tokens, paths):
        """Materialize compact token paths from a packed-tree topology."""

        return [
            [int(flat_tokens[node_index]) for node_index in path]
            for path in paths
        ]

    @staticmethod
    def _longest_matching_path(path_token_ids, future_token_ids) -> int:
        """Return the longest tree-path prefix matching a realized continuation."""

        future = [int(token) for token in future_token_ids]
        best = 0
        for path in path_token_ids:
            matched = 0
            for candidate, target in zip(path, future):
                if int(candidate) != target:
                    break
                matched += 1
            best = max(best, matched)
        return int(best)

    @staticmethod
    def _visual_probe_metrics(view_logits: torch.Tensor, top_k: int):
        """Summarize full-view versus spatially masked next-token logits."""

        if view_logits.ndim != 2 or int(view_logits.shape[0]) < 2:
            raise ValueError(
                "view_logits must contain one full and at least one masked view"
            )
        top1 = torch.argmax(view_logits, dim=-1)
        disagreement = top1[1:].ne(top1[0]).float().mean()
        return {
            "visual_probe_jsd": float(multiview_jsd(view_logits)),
            "visual_probe_topk_union_size": int(
                topk_union_size(view_logits, top_k)
            ),
            "visual_probe_top1_disagreement_rate": float(
                disagreement.item()
            ),
        }

    @staticmethod
    def _best_verified_path(
        output_logits: torch.Tensor,
        flat_tokens: List[int],
        paths: List[List[int]],
        remaining_tokens: int,
        predictions: Optional[torch.Tensor] = None,
    ):
        # Transfer the small packed-tree prediction vector once.  Calling
        # ``Tensor.item`` for every path repeatedly synchronized the same root
        # and parent logits, which was measurable at decoding granularity.
        if predictions is None:
            predictions = torch.argmax(output_logits[0], dim=-1)
        predictions = predictions.tolist()
        best_path = []
        best_accept = 0
        for path in paths:
            accepted = 0
            parent_index = 0
            for node_index in path:
                if int(predictions[parent_index]) != flat_tokens[node_index]:
                    break
                accepted += 1
                parent_index = node_index
            if accepted > best_accept:
                best_accept = accepted
                best_path = path

        best_accept = min(best_accept, max(remaining_tokens - 1, 0))
        accepted_path = best_path[:best_accept]
        query_index = accepted_path[-1] if accepted_path else 0
        correction = int(predictions[query_index])
        accepted_tokens = [flat_tokens[index] for index in accepted_path]
        return accepted_path, accepted_tokens, correction, query_index

    @staticmethod
    def _promote_transition(
        transitions: torch.Tensor,
        transition_valid: torch.Tensor,
        transition_bin: int,
        source_token: int,
        next_token: int,
    ) -> None:
        """Place a retrieved edge first while retaining recycled alternatives."""
        row_bin = int(transition_bin)
        if not bool(transition_valid[row_bin, source_token].item()):
            fallback_bins = torch.nonzero(
                transition_valid[:, source_token], as_tuple=True
            )[0]
            if fallback_bins.numel() > 0:
                fallback_bin = int(fallback_bins[0].item())
                transitions[row_bin, source_token].copy_(
                    transitions[fallback_bin, source_token]
                )
            else:
                transitions[row_bin, source_token].fill_(int(next_token))
            transition_valid[row_bin, source_token] = True

        row = transitions[row_bin, source_token]
        retained = row[row.ne(int(next_token))]
        promoted = torch.full_like(row, int(next_token))
        keep = min(int(retained.numel()), max(int(row.numel()) - 1, 0))
        if keep > 0:
            promoted[1 : 1 + keep] = retained[:keep]
        row.copy_(promoted)

    @torch.no_grad()
    def specgenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
        inputs_embeds=None,
        return_acceptance_len=False,
        return_decode_time=False,
        draft_policy="visual-anchor",
        min_draft_tokens=2,
        max_draft_tokens=10,
        visual_threshold=0.55,
        confidence_threshold=0.75,
        visual_gamma=1.0,
        visual_weight=0.7,
        acceptance_weight=0.3,
        acceptance_ema_decay=0.8,
        grounding_layer=-1,
        confidence_margin_scale=5.0,
        matrix_top_k=4,
        tree_fixed_width=2,
        tree_fixed_depth=4,
        tree_broad_width=4,
        tree_shallow_depth=2,
        tree_node_budget=31,
        cover_num_probes=4,
        cover_anchor_fraction=0.5,
        cover_probe_visual_threshold=0.75,
        cover_min_jsd=0.02,
        visual_lexical_pool_size=64,
        visual_lexical_width=4,
        hst_token_weight=0.5,
        hst_neighbors=16,
        hst_temperature=0.05,
        hst_transport_mode="delta_full",
        hst_source_scope="all_text",
        hst_visual_weight=0.0,
        hst_min_confidence=0.39,
        hst_online_update=True,
        hst_trace_diagnostics=False,
        verification_trace_diagnostics=False,
        verification_layer_diagnostics=False,
        verification_margin_threshold=0.0,
        verification_compact_path_repair=False,
        verification_compact_root_margin_threshold=0.0,
        candidate_trace_diagnostics=False,
        selective_reuse_diagnostics=False,
        selective_reuse_probe_mode="packed-attention",
        visual_cache_key=None,
        return_policy_trace=False,
        disable_repeat_guard=True,
        **kwargs,
    ):
        del top_p, top_k, min_draft_tokens, max_draft_tokens, disable_repeat_guard
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids.shape[0] != 1:
            raise ValueError("Tree Recycling currently supports batch size 1")
        if temperature > 1e-5:
            raise NotImplementedError("Tree Recycling pilot currently supports greedy decoding")

        requested_draft_policy = str(draft_policy)
        collect_candidate_trace = bool(
            return_policy_trace and candidate_trace_diagnostics
        )
        collect_selective_reuse = bool(
            return_policy_trace and selective_reuse_diagnostics
        )
        selective_reuse_probe_mode = str(selective_reuse_probe_mode)
        if selective_reuse_probe_mode not in SELECTIVE_REUSE_PROBE_MODES:
            raise ValueError(
                "selective_reuse_probe_mode must be packed-attention, "
                "teacher-forced-pixel, teacher-forced-content-ablation, or "
                "teacher-forced-counterfactual-bank"
            )
        packed_selective_reuse_probes = bool(
            collect_selective_reuse
            and selective_reuse_probe_mode == "packed-attention"
        )
        cover_enabled = requested_draft_policy.startswith("cover-")
        if cover_enabled:
            draft_policy = requested_draft_policy[len("cover-") :]
            if not draft_policy:
                raise ValueError("COVER policy must name an underlying tree policy")
        cover_num_probes = max(int(cover_num_probes), 1)
        cover_anchor_fraction = float(cover_anchor_fraction)
        if not 0.0 <= cover_anchor_fraction <= 1.0:
            raise ValueError("cover_anchor_fraction must lie in [0, 1]")
        cover_probe_visual_threshold = float(cover_probe_visual_threshold)
        cover_min_jsd = float(cover_min_jsd)
        if not 0.0 <= cover_probe_visual_threshold <= 1.0:
            raise ValueError("cover_probe_visual_threshold must lie in [0, 1]")
        if cover_min_jsd < 0.0:
            raise ValueError("cover_min_jsd must be non-negative")
        visual_lexical_pool_size = max(int(visual_lexical_pool_size), 1)
        visual_lexical_width = max(int(visual_lexical_width), 1)
        visual_hst_enabled = draft_policy in (
            "visual-hst-backoff",
            "visual-hst-backoff-gated",
            "visual-wide-plus2-hst-backoff",
            "visual-wide-plus2-hst-backoff-gated",
            "visual-rootwide-plus2-hst-backoff",
        )
        visual_hst_gate_enabled = draft_policy in (
            "visual-hst-backoff-gated",
            "visual-wide-plus2-hst-backoff-gated",
        )
        visual_lexical_enabled = draft_policy in (
            "visual-lexical-backoff",
            "visual-hst-backoff",
            "visual-hst-backoff-gated",
            "visual-wide-plus2-hst-backoff",
            "visual-wide-plus2-hst-backoff-gated",
            "visual-rootwide-plus2-hst-backoff",
            "visual-wide-plus2-vli-backoff",
        )
        if hst_transport_mode not in TRANSPORT_MODES:
            raise ValueError(f"unknown HST transport mode: {hst_transport_mode}")
        if hst_source_scope not in SOURCE_SCOPES:
            raise ValueError(f"unknown HST source scope: {hst_source_scope}")
        hst_config = HiddenStateTransportConfig(
            token_weight=float(hst_token_weight),
            neighbors=max(int(hst_neighbors), 1),
            temperature=float(hst_temperature),
            transport_mode=str(hst_transport_mode),
            source_scope=str(hst_source_scope),
            visual_weight=float(hst_visual_weight),
        )
        hst_min_confidence = float(hst_min_confidence)
        if not 0.0 <= hst_min_confidence <= 1.0:
            raise ValueError("hst_min_confidence must be in [0, 1]")
        verification_margin_threshold = float(
            verification_margin_threshold
        )
        verification_compact_root_margin_threshold = float(
            verification_compact_root_margin_threshold
        )
        if verification_margin_threshold < 0.0:
            raise ValueError(
                "verification_margin_threshold must be non-negative"
            )
        if verification_compact_root_margin_threshold < 0.0:
            raise ValueError(
                "verification_compact_root_margin_threshold must be "
                "non-negative"
            )
        if verification_layer_diagnostics and verification_margin_threshold <= 0.0:
            raise ValueError(
                "verification_layer_diagnostics requires a positive "
                "verification_margin_threshold"
            )
        if (
            verification_compact_path_repair
            and verification_margin_threshold > 0.0
        ):
            raise ValueError(
                "verification_compact_path_repair and margin recheck are "
                "mutually exclusive"
            )
        if verification_compact_path_repair and visual_hst_enabled:
            raise ValueError(
                "verification_compact_path_repair does not support HST policies"
            )

        grounding_free_policies = {
            "target",
            "fixed",
            "broad",
            "wide-plus2",
            "rank-prior-wide-plus2",
            "rank-prior-deep-wide-plus2",
            "rank-prior-deeper-wide-plus2",
            "rank-prior-deepest-wide-plus2",
            "score-prior-deep-wide-plus2",
            "score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus4",
            "context-score-trigram-deeper-wide-plus4",
            "context-score-trigram-residual2-deeper-wide-plus4",
            "context-score-trigram-residual2-deepest-wide-plus4",
            "context-score-trigram-fusion-deeper-wide-plus4",
            "context-score-trigram-fusion-deepest-wide-plus4",
            "context-score-trigram-fusion55-deepest-wide-plus4",
            "context-score-trigram-fusion-adaptive-deepest-wide-plus4",
            "context-score-trigram-fusion-calibrated-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-bank-deepest-wide-plus4",
            "context-score-trigram-fusion-bank-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-global7-deepest-wide-plus4",
            "context-score-trigram-fusion-global15-deepest-wide-plus4",
            "context-score-trigram-deepest-wide-plus4",
            "context-score-trigram-deeper-wide-plus6",
            "context-score-trigram-deeper-wide-plus8",
            "context-score-fourgram-deeper-wide-plus4",
            "context-score-fourgram-deepest-wide-plus4",
            "context-score-prior-deeper-wide-plus5",
            "context-score-prior-deeper-wide-plus6",
            "context-score-prior-deeper-wide-plus8",
            "context-score-calibrated-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2-node55",
            "context-score-prior-deeper-wide-plus2-node47",
            "context-score-adaptive-safe-deeper-wide-plus2",
            "score-prior-deepest-wide-plus2",
            "score-prior-maxdeep-wide-plus2",
            "score-adaptive-safe-deeper-wide-plus2",
            "score-adaptive-deeper-wide-plus2",
            "narrow",
            "spine",
            "short",
            "hybrid",
        }
        grounding_free_policies.update(TRIGRAM_PLUS4_NODE_BUDGETS)
        grounding_free_policies.update(PERSISTENT_FUSION_NODE_BUDGETS)
        grounding_free_policies.update(PERSISTENT_COMMITTED_POLICIES)
        grounding_free_policies.update(PERSISTENT_STABLE_NODE_BUDGETS)
        grounding_free_policies.update(PERSISTENT_DEPTH_NODE_CONFIGS)
        grounding_free_policies.update(PERSISTENT_ADAPTIVE_DEPTH_POLICIES)
        grounding_free_policies.update(PERSISTENT_OPTIMIZED_DEPTH_CONFIGS)
        grounding_free_policies.update(MATCHED_BUDGET_CONTROL_CONFIGS)
        need_grounding = bool(
            cover_enabled or draft_policy not in grounding_free_policies
        )
        need_confidence = draft_policy in (
            "visual-anchor",
            "visual-hst-backoff-gated",
            "visual-wide-plus2-hst-backoff-gated",
        )
        need_hidden_states = bool(need_grounding or visual_hst_enabled)
        # The default grounding signal and HST both consume only the final
        # normalized decoder state.  Retain the full per-layer tuple solely
        # when an explicit non-final grounding layer is requested.
        need_all_hidden_states = bool(
            need_grounding and int(grounding_layer) != -1
        )

        controller = GroundedDraftController(
            policy=draft_policy,
            min_draft_tokens=0,
            max_draft_tokens=max(tree_node_budget, 1),
            visual_threshold=visual_threshold,
            confidence_threshold=confidence_threshold,
            visual_gamma=visual_gamma,
            visual_weight=visual_weight,
            acceptance_weight=acceptance_weight,
            acceptance_ema_decay=acceptance_ema_decay,
        )
        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        prompt_length = int(input_ids.shape[1])
        arch = self.base_model.config.architectures[0]
        if arch not in SUPPORTED_MULTIMODAL_ARCHITECTURES:
            raise NotImplementedError(
                f"Tree Recycling does not support architecture {arch}"
            )
        max_length = resolve_generation_max_length(
            self.base_model.config,
            prompt_length,
            max_new_tokens,
            max_length,
        )
        tree_fixed_width = max(int(tree_fixed_width), 1)
        tree_fixed_depth = max(int(tree_fixed_depth), 1)
        tree_broad_width = max(int(tree_broad_width), 1)
        tree_shallow_depth = max(int(tree_shallow_depth), 1)
        tree_node_budget = max(int(tree_node_budget), 1)
        uses_augmented_width = draft_policy in (
            "wide-plus2",
            "rank-prior-wide-plus2",
            "rank-prior-deep-wide-plus2",
            "rank-prior-deeper-wide-plus2",
            "rank-prior-deepest-wide-plus2",
            "score-prior-deep-wide-plus2",
            "score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus4",
            "context-score-trigram-deeper-wide-plus4",
            "context-score-trigram-residual2-deeper-wide-plus4",
            "context-score-trigram-residual2-deepest-wide-plus4",
            "context-score-trigram-fusion-deeper-wide-plus4",
            "context-score-trigram-fusion-deepest-wide-plus4",
            "context-score-trigram-fusion55-deepest-wide-plus4",
            "context-score-trigram-fusion-adaptive-deepest-wide-plus4",
            "context-score-trigram-fusion-calibrated-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-bank-deepest-wide-plus4",
            "context-score-trigram-fusion-bank-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-global7-deepest-wide-plus4",
            "context-score-trigram-fusion-global15-deepest-wide-plus4",
            "context-score-trigram-deepest-wide-plus4",
            "context-score-trigram-deeper-wide-plus6",
            "context-score-trigram-deeper-wide-plus8",
            "context-score-fourgram-deeper-wide-plus4",
            "context-score-fourgram-deepest-wide-plus4",
            "context-score-prior-deeper-wide-plus5",
            "context-score-prior-deeper-wide-plus6",
            "context-score-prior-deeper-wide-plus8",
            "context-score-calibrated-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2-node55",
            "context-score-prior-deeper-wide-plus2-node47",
            "context-score-adaptive-safe-deeper-wide-plus2",
            "score-prior-deepest-wide-plus2",
            "score-prior-maxdeep-wide-plus2",
            "score-adaptive-safe-deeper-wide-plus2",
            "score-adaptive-deeper-wide-plus2",
            "visual-wide-plus2",
            "visual-wide-plus2-reverse",
            "visual-rootwide-plus2",
            "visual-rootwide-plus2-hst-backoff",
            "visual-wide-plus2-vli-backoff",
            "visual-wide-plus2-hst-backoff",
            "visual-wide-plus2-hst-backoff-gated",
            "grounded-residual",
            "grounded-residual-reverse",
        ) or draft_policy in CONTEXT_PLUS4_SPECIAL_POLICIES
        context_width_augmentation = {
            "context-score-prior-deeper-wide-plus4": 4,
            "context-score-trigram-deeper-wide-plus4": 4,
            "context-score-trigram-residual2-deeper-wide-plus4": 4,
            "context-score-trigram-residual2-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-deeper-wide-plus4": 4,
            "context-score-trigram-fusion-deepest-wide-plus4": 4,
            "context-score-trigram-fusion55-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-adaptive-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-calibrated-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-bank-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-bank-global15-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-global7-deepest-wide-plus4": 4,
            "context-score-trigram-fusion-global15-deepest-wide-plus4": 4,
            "context-score-trigram-deepest-wide-plus4": 4,
            "context-score-trigram-deeper-wide-plus6": 6,
            "context-score-trigram-deeper-wide-plus8": 8,
            "context-score-fourgram-deeper-wide-plus4": 4,
            "context-score-fourgram-deepest-wide-plus4": 4,
            "context-score-prior-deeper-wide-plus5": 5,
            "context-score-prior-deeper-wide-plus6": 6,
            "context-score-prior-deeper-wide-plus8": 8,
        }.get(draft_policy, 0)
        if draft_policy in CONTEXT_PLUS4_SPECIAL_POLICIES:
            context_width_augmentation = 4
        matrix_top_k = max(
            int(matrix_top_k),
            tree_fixed_width,
            tree_broad_width
            + (
                context_width_augmentation
                if context_width_augmentation > 0
                else (2 if uses_augmented_width else 0)
            ),
            visual_lexical_width if visual_lexical_enabled else 1,
            1,
        )

        if hasattr(self, "tree_recycling_past_key_values"):
            past_key_values = self.tree_recycling_past_key_values
            past_key_values_data = self.tree_recycling_past_key_values_data
            current_length_data = self.tree_recycling_current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.tree_recycling_past_key_values = past_key_values
            self.tree_recycling_past_key_values_data = past_key_values_data
            self.tree_recycling_current_length_data = current_length_data

        image_token_id = resolve_image_token_id(self.base_model.config)
        visual_mask = self._build_visual_token_mask(input_ids).detach()
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        visual_probe_layout = None
        if cover_enabled or packed_selective_reuse_probes:
            if arch != "Qwen2_5_VLForConditionalGeneration":
                raise NotImplementedError(
                    "Counterfactual visual probes currently require Qwen2.5-VL"
                )
            visual_probe_layout = build_visual_probe_layout(
                visual_mask,
                image_grid_thw,
                int(self.base_model.config.vision_config.spatial_merge_size),
                cover_num_probes,
            )
        if arch == "Qwen2_5_VLForConditionalGeneration" and inputs_embeds is None:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                cached_visual = getattr(
                    self, "_gwtr_visual_feature_cache", None
                )
                if (
                    visual_cache_key is not None
                    and cached_visual is not None
                    and cached_visual[0] == visual_cache_key
                ):
                    image_embeds = cached_visual[1]
                else:
                    image_embeds = self.base_model.visual(
                        pixel_values.type(self.base_model.visual.dtype),
                        grid_thw=image_grid_thw,
                    )
                    if visual_cache_key is not None:
                        # Only the immediately active multi-turn request needs
                        # reuse.  A single-entry cache bounds memory and avoids
                        # leaking features across samples or policies.
                        self._gwtr_visual_feature_cache = (
                            visual_cache_key,
                            image_embeds.detach(),
                        )
                image_mask = input_ids.eq(image_token_id)
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise ValueError("Image features and image tokens do not match")
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
                )

        init_output = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
            output_hidden_states=need_all_hidden_states,
            output_last_hidden_state=need_hidden_states,
            **kwargs,
        )
        if draft_policy == "target":
            return self._target_only_from_prefill(
                input_ids=input_ids,
                prefill_output=init_output,
                past_key_values=past_key_values,
                current_length_data=current_length_data,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
                log=log,
                return_acceptance_len=return_acceptance_len,
                return_decode_time=return_decode_time,
                return_policy_trace=return_policy_trace,
            )

        # KV rows before the active cursor are never overwritten by packed-tree
        # verification.  Treat the exact prompt cache as a recovery checkpoint;
        # after a recovery, advance the checkpoint so later repairs replay only
        # the short suffix generated since that point.
        exact_checkpoint_length = int(current_length_data[0].item())
        exact_checkpoint_generated = 0

        def rebase_exact_prefix(sequence_ids: torch.Tensor):
            """Replay only the suffix after the latest exact KV checkpoint."""

            nonlocal exact_checkpoint_generated, exact_checkpoint_length
            current_length_data.fill_(exact_checkpoint_length)
            rebased_output = None
            final_position = int(sequence_ids.shape[1]) - 1
            replay_start = prompt_length + exact_checkpoint_generated
            for token_position in range(replay_start, sequence_ids.shape[1]):
                rebased_output = self.base_model(
                    input_ids=sequence_ids[
                        :, token_position : token_position + 1
                    ],
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=(
                        (need_all_hidden_states or verification_layer_diagnostics)
                        and token_position == final_position
                    ),
                    output_last_hidden_state=(
                        need_hidden_states
                        and token_position == final_position
                    ),
                )
            if rebased_output is None:
                raise RuntimeError(
                    "exact-prefix recovery requires at least one suffix token"
                )
            replayed_tokens = int(sequence_ids.shape[1] - replay_start)
            exact_checkpoint_generated = int(
                sequence_ids.shape[1] - prompt_length
            )
            exact_checkpoint_length = int(sequence_ids.shape[1])
            return rebased_output, replayed_tokens

        hidden = (
            self._layer_hidden(init_output, grounding_layer)
            if need_grounding
            else None
        )
        calibrator = (
            VisualGroundingCalibrator.from_prompt(hidden[0], visual_mask)
            if hidden is not None
            else None
        )
        grounding_score, confidence = self._score_state(
            init_output,
            init_output.logits.shape[1] - 1,
            calibrator,
            grounding_layer,
            confidence_margin_scale,
            need_grounding=need_grounding,
            need_confidence=need_confidence,
        )
        visual_lexical_inventory: List[int] = []
        # Selective-reuse diagnostics use a genuinely image-conditioned
        # proposal source.  Keep it separate from the deployable VLI policies
        # so collecting the diagnostic cannot change the decoded tree.
        selective_visual_inventory: List[int] = []
        excluded_token_ids = set(int(token) for token in self.tokenizer.all_special_ids)
        for attribute in ("image_token_id", "image_token_index", "video_token_id"):
            token_id = getattr(self.base_model.config, attribute, None)
            if token_id is not None:
                excluded_token_ids.add(int(token_id))
        if visual_lexical_enabled and bool(visual_mask.any().item()):
            visual_scores = init_output.logits[0, visual_mask].max(dim=0).values
            visual_lexical_inventory = rank_scores(
                visual_scores,
                visual_lexical_pool_size,
                excluded_token_ids=sorted(excluded_token_ids),
                valid_vocab_size=min(
                    len(self.tokenizer), int(init_output.logits.shape[-1])
                ),
            )
        if collect_selective_reuse and bool(visual_mask.any().item()):
            visual_scores = init_output.logits[0, visual_mask].max(dim=0).values
            selective_visual_inventory = rank_scores(
                visual_scores,
                visual_lexical_pool_size,
                excluded_token_ids=sorted(excluded_token_ids),
                valid_vocab_size=min(
                    len(self.tokenizer), int(init_output.logits.shape[-1])
                ),
            )
        embedding_layer = self.base_model.get_input_embeddings()
        hst_bank = None
        final_prompt_hidden = (
            self._layer_hidden(init_output, -1)
            if visual_hst_enabled
            else None
        )
        hst_parent_hidden = (
            final_prompt_hidden[0, -1]
            if final_prompt_hidden is not None
            else None
        )
        hst_parent_candidate_logits = None
        if visual_hst_enabled and visual_lexical_inventory:
            candidate_ids = torch.tensor(
                visual_lexical_inventory,
                dtype=torch.long,
                device=input_ids.device,
            )
            output_projection = self.base_model.get_output_embeddings()
            hst_bank = HiddenStateTransportBank.from_prompt(
                candidate_token_ids=candidate_ids,
                output_projection_weight=output_projection.weight,
                output_projection_bias=getattr(output_projection, "bias", None),
                prompt_token_ids=input_ids[0],
                prompt_token_embeddings=embedding_layer(input_ids[0]),
                prompt_hidden_states=final_prompt_hidden[0],
                visual_mask=visual_mask,
                excluded_token_ids=sorted(excluded_token_ids),
                additional_capacity=(
                    max_new_tokens + 1 if hst_online_update else 0
                ),
            )
            hst_parent_candidate_logits = init_output.logits[0, -1].index_select(
                0, candidate_ids
            )

        use_grounded_backoff = draft_policy in (
            "grounded-backoff",
            "grounded-backoff-reverse",
        )
        use_grounded_residual = draft_policy in (
            "grounded-residual",
            "grounded-residual-reverse",
        )
        use_grounded_conditionals = use_grounded_backoff or use_grounded_residual
        use_modal_bins = draft_policy.startswith("modal-")
        use_suffix_hybrid = draft_policy in (
            "hybrid",
            "visual-hybrid",
            "modal-hybrid",
            "grounded-hybrid",
            "grounded-hybrid-reverse",
        )
        num_transition_bins = 3 if use_grounded_conditionals else (2 if use_modal_bins else 1)
        use_host_transitions = bool(
            num_transition_bins == 1
            and not cover_enabled
            and not visual_lexical_enabled
            and not use_suffix_hybrid
            and not verification_compact_path_repair
        )
        use_context_transitions = draft_policy.startswith("context-score-")
        use_fourgram_transitions = draft_policy.startswith(
            "context-score-fourgram-"
        )
        use_global_backoff = draft_policy in GLOBAL_BACKOFF_POLICIES
        use_trigram_transitions = bool(
            use_fourgram_transitions
            or draft_policy.startswith("context-score-trigram-")
        )
        use_score_priority = draft_policy.startswith(
            ("score-prior-", "score-adaptive-", "context-score-")
        )
        use_hotpath = "-hotpath-" in draft_policy
        use_suffix_reserve = "-suffix4-" in draft_policy
        use_persistent_ngram = bool(
            use_host_transitions and "-persistent-ngram-" in draft_policy
        )
        use_persistent_unigram = bool(
            use_host_transitions and "-persistent-" in draft_policy
        )
        use_persistent_shadow = bool(
            use_host_transitions and "-shadow-" in draft_policy
        )
        use_committed_transitions_only = bool(
            use_host_transitions and "-committed-" in draft_policy
        )
        skip_cached_prompt_transitions = bool(
            use_host_transitions and "-stable-" in draft_policy
        )
        use_persistent_bank = bool(
            use_host_transitions and "-bank-" in draft_policy
        )
        persistent_bank_transitions = None
        persistent_bank_scores = None
        persistent_bank_counts = None
        persistent_bank_pending = {}
        persistent_shadow_transitions = None
        persistent_shadow_scores = None
        if use_persistent_bank:
            persistent_banks = getattr(
                self, "_gwtr_persistent_unigram_banks", None
            )
            if persistent_banks is None:
                persistent_banks = {}
                self._gwtr_persistent_unigram_banks = persistent_banks
            persistent_bank = persistent_banks.setdefault(
                draft_policy,
                {"transitions": {}, "scores": {}, "counts": {}},
            )
            persistent_bank_transitions = persistent_bank["transitions"]
            persistent_bank_scores = persistent_bank["scores"]
            persistent_bank_counts = persistent_bank["counts"]
        if use_persistent_unigram:
            persistent_caches = getattr(
                self, "_gwtr_persistent_unigram_caches", None
            )
            if persistent_caches is None:
                persistent_caches = {}
                self._gwtr_persistent_unigram_caches = persistent_caches
            persistent_cache = persistent_caches.setdefault(
                draft_policy,
                {"transitions": {}, "scores": {}},
            )
            host_transitions = persistent_cache["transitions"]
            host_transition_scores = persistent_cache["scores"]
            if use_persistent_shadow:
                persistent_shadow_transitions = persistent_cache.setdefault(
                    "shadow_transitions", {}
                )
                persistent_shadow_scores = persistent_cache.setdefault(
                    "shadow_scores", {}
                )
        else:
            host_transitions = {} if use_host_transitions else None
            host_transition_scores = {} if use_score_priority else None
        persistent_unigram_size_at_start = (
            len(host_transitions)
            if use_persistent_unigram
            else (
                len(persistent_bank_transitions)
                if persistent_bank_transitions is not None
                else 0
            )
        )
        if use_persistent_ngram:
            host_context_transitions = persistent_cache.setdefault(
                "context_transitions", {}
            )
            host_context_transition_scores = persistent_cache.setdefault(
                "context_scores", {}
            )
            host_trigram_transitions = persistent_cache.setdefault(
                "trigram_transitions", {}
            )
            host_trigram_transition_scores = persistent_cache.setdefault(
                "trigram_scores", {}
            )
        else:
            host_context_transitions = {} if use_context_transitions else None
            host_context_transition_scores = (
                {} if use_context_transitions else None
            )
            host_trigram_transitions = {} if use_trigram_transitions else None
            host_trigram_transition_scores = (
                {} if use_trigram_transitions else None
            )
        host_fourgram_transitions = {} if use_fourgram_transitions else None
        host_fourgram_transition_scores = (
            {} if use_fourgram_transitions else None
        )
        persistent_context_size_at_start = (
            len(host_context_transitions) if use_persistent_ngram else 0
        )
        persistent_trigram_size_at_start = (
            len(host_trigram_transitions) if use_persistent_ngram else 0
        )
        candidate_trace_persistent_keys_at_start = (
            set(host_transitions)
            if (collect_candidate_trace or collect_selective_reuse)
            and use_persistent_unigram
            and host_transitions is not None
            else set()
        )
        selective_u_top_probabilities_at_start = (
            {
                int(token): float(scores[0])
                for token, scores in host_transition_scores.items()
                if scores
            }
            if collect_selective_reuse
            and use_persistent_unigram
            and host_transition_scores is not None
            else {}
        )
        global_candidate_counts = {} if use_global_backoff else None
        global_candidate_ids = []
        global_candidate_scores = []
        vocab_size = int(self.base_model.config.vocab_size)
        if host_transitions is not None:
            # Plain recycling policies never read the dense GPU tables.  Keep a
            # device sentinel for tree tensor placement and avoid zeroing a
            # multi-megabyte vocab-sized matrix at the start of every sample.
            transitions = torch.empty(
                0, dtype=torch.long, device=input_ids.device
            )
            transition_valid = torch.empty(
                0, dtype=torch.bool, device=input_ids.device
            )
        else:
            transitions = torch.zeros(
                (num_transition_bins, vocab_size, matrix_top_k),
                dtype=torch.long,
                device=input_ids.device,
            )
            transition_valid = torch.zeros(
                (num_transition_bins, vocab_size),
                dtype=torch.bool,
                device=input_ids.device,
            )
        transition_counts = (
            torch.zeros(
                (num_transition_bins, vocab_size),
                dtype=torch.int32,
                device=input_ids.device,
            )
            if use_grounded_conditionals
            else None
        )

        def store_transitions(
            output,
            token_ids,
            token_values=None,
            previous_token_values=None,
            previous_previous_token_values=None,
            previous_previous_previous_token_values=None,
            output_row_indices=None,
        ):
            query_count = int(token_ids.numel())
            if output_row_indices is None:
                transition_logits = output.logits[0, :query_count]
            else:
                transition_logits = output.logits[0].index_select(
                    0, output_row_indices
                )
                if int(transition_logits.shape[0]) != query_count:
                    raise ValueError(
                        "Transition row selection must match token count"
                    )
            topk = transition_logits.topk(matrix_top_k, dim=-1)
            topk_ids = topk.indices
            if host_transitions is not None:
                queued_score_rows = (
                    torch.softmax(topk.values.float(), dim=-1)
                    if use_hotpath and host_transition_scores is not None
                    else None
                )
                rows = topk_ids.tolist()
                score_rows = (
                    (
                        queued_score_rows.tolist()
                        if queued_score_rows is not None
                        else torch.softmax(
                            topk.values.float(), dim=-1
                        ).tolist()
                    )
                    if host_transition_scores is not None
                    else None
                )
                if global_candidate_counts is not None:
                    for row_index, row in enumerate(rows):
                        if not row:
                            continue
                        token = int(row[0])
                        weight = (
                            float(score_rows[row_index][0])
                            if score_rows is not None
                            else 1.0
                        )
                        global_candidate_counts[token] = (
                            global_candidate_counts.get(token, 0.0) + weight
                        )
                    ranked_global = sorted(
                        global_candidate_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:matrix_top_k]
                    global_candidate_ids[:] = [
                        int(token) for token, _ in ranked_global
                    ]
                    total_global_score = (
                        sum(score for _, score in ranked_global) or 1.0
                    )
                    global_candidate_scores[:] = [
                        float(score) / total_global_score
                        for _, score in ranked_global
                    ]
                tokens = (
                    [int(token) for token in token_values]
                    if token_values is not None
                    else [int(token) for token in token_ids.tolist()]
                )
                previous_tokens = (
                    list(previous_token_values)
                    if previous_token_values is not None
                    else None
                )
                previous_previous_tokens = (
                    list(previous_previous_token_values)
                    if previous_previous_token_values is not None
                    else None
                )
                previous_previous_previous_tokens = (
                    list(previous_previous_previous_token_values)
                    if previous_previous_previous_token_values is not None
                    else None
                )
                # CUDA advanced-index assignment retains the first value for
                # duplicate indices within one update.  Mirror that behavior
                # so this host lookup table is semantically identical.
                seen_tokens = set()
                seen_contexts = set()
                seen_trigrams = set()
                seen_fourgrams = set()
                for row_index, (token, row) in enumerate(zip(tokens, rows)):
                    previous_token = (
                        previous_tokens[row_index]
                        if previous_tokens is not None
                        else None
                    )
                    context_key = (
                        (int(previous_token), int(token))
                        if previous_token is not None
                        else None
                    )
                    previous_previous_token = (
                        previous_previous_tokens[row_index]
                        if previous_previous_tokens is not None
                        else None
                    )
                    trigram_key = (
                        (
                            int(previous_previous_token),
                            int(previous_token),
                            int(token),
                        )
                        if previous_previous_token is not None
                        and previous_token is not None
                        else None
                    )
                    previous_previous_previous_token = (
                        previous_previous_previous_tokens[row_index]
                        if previous_previous_previous_tokens is not None
                        else None
                    )
                    fourgram_key = (
                        (
                            int(previous_previous_previous_token),
                            int(previous_previous_token),
                            int(previous_token),
                            int(token),
                        )
                        if previous_previous_previous_token is not None
                        and previous_previous_token is not None
                        and previous_token is not None
                        else None
                    )
                    update_unigram = token not in seen_tokens
                    update_context = bool(
                        context_key is not None
                        and context_key not in seen_contexts
                    )
                    update_trigram = bool(
                        trigram_key is not None
                        and trigram_key not in seen_trigrams
                    )
                    update_fourgram = bool(
                        fourgram_key is not None
                        and fourgram_key not in seen_fourgrams
                    )
                    if (
                        not update_unigram
                        and not update_context
                        and not update_trigram
                        and not update_fourgram
                    ):
                        continue
                    converted_row = [int(value) for value in row]
                    normalized_scores = None
                    if score_rows is not None:
                        normalized_scores = score_rows[row_index]
                    if update_unigram:
                        seen_tokens.add(token)
                        if (
                            persistent_shadow_transitions is not None
                            and token in host_transitions
                            and host_transitions[token] != converted_row
                        ):
                            persistent_shadow_transitions[token] = list(
                                host_transitions[token]
                            )
                            if token in host_transition_scores:
                                persistent_shadow_scores[token] = list(
                                    host_transition_scores[token]
                                )
                        host_transitions[token] = converted_row
                        if normalized_scores is not None:
                            host_transition_scores[token] = normalized_scores
                        if (
                            persistent_bank_transitions is not None
                            and token not in persistent_bank_pending
                            and normalized_scores is not None
                        ):
                            persistent_bank_pending[token] = (
                                converted_row,
                                normalized_scores,
                            )
                    if update_context:
                        seen_contexts.add(context_key)
                        if use_persistent_ngram:
                            self._store_bounded_transition_row(
                                host_context_transitions,
                                host_context_transition_scores,
                                context_key,
                                converted_row,
                                normalized_scores,
                                PERSISTENT_NGRAM_ROW_LIMIT,
                            )
                        else:
                            host_context_transitions[context_key] = converted_row
                            if normalized_scores is not None:
                                host_context_transition_scores[
                                    context_key
                                ] = normalized_scores
                    if update_trigram:
                        seen_trigrams.add(trigram_key)
                        if use_persistent_ngram:
                            self._store_bounded_transition_row(
                                host_trigram_transitions,
                                host_trigram_transition_scores,
                                trigram_key,
                                converted_row,
                                normalized_scores,
                                PERSISTENT_NGRAM_ROW_LIMIT,
                            )
                        else:
                            host_trigram_transitions[trigram_key] = converted_row
                            if normalized_scores is not None:
                                host_trigram_transition_scores[
                                    trigram_key
                                ] = normalized_scores
                    if update_fourgram:
                        seen_fourgrams.add(fourgram_key)
                        host_fourgram_transitions[
                            fourgram_key
                        ] = converted_row
                        if normalized_scores is not None:
                            host_fourgram_transition_scores[
                                fourgram_key
                            ] = normalized_scores
                return
            if use_grounded_conditionals:
                global_bins = torch.zeros_like(token_ids, dtype=torch.long)
                transitions[global_bins, token_ids] = topk_ids
                transition_valid[global_bins, token_ids] = True
                transition_counts.index_put_(
                    (global_bins, token_ids),
                    torch.ones_like(token_ids, dtype=torch.int32),
                    accumulate=True,
                )
                if calibrator is None:
                    return
                output_hidden = self._layer_hidden(output, grounding_layer)
                if output_hidden is None:
                    return
                bins = 1 + calibrator.scores(output_hidden[0, :query_count]).ge(
                    visual_threshold
                ).long()
            elif num_transition_bins == 1 or calibrator is None:
                bins = torch.zeros_like(token_ids, dtype=torch.long)
            else:
                output_hidden = self._layer_hidden(output, grounding_layer)
                if output_hidden is None:
                    bins = torch.zeros_like(token_ids, dtype=torch.long)
                else:
                    bins = calibrator.scores(output_hidden[0, :query_count]).ge(
                        visual_threshold
                    ).long()
            transitions[bins, token_ids] = topk_ids
            transition_valid[bins, token_ids] = True
            if transition_counts is not None:
                transition_counts.index_put_(
                    (bins, token_ids),
                    torch.ones_like(token_ids, dtype=torch.int32),
                    accumulate=True,
                )

        prompt_ids = input_ids[0]
        # Building exact bigram rows for every multimodal prompt position is
        # disproportionately expensive on high-resolution images.  Seed the
        # cheap unigram fallback from prefill and learn context rows online
        # from the much smaller verified generation trees instead.
        if host_transitions is not None:
            prompt_values = [int(token) for token in prompt_ids.tolist()]
            (
                unique_prompt_indices,
                unique_prompt_values,
            ) = self._select_prompt_transition_rows(
                prompt_values,
                cached_tokens=(
                    host_transitions
                    if skip_cached_prompt_transitions
                    else None
                ),
            )
            prompt_transition_row_count = len(unique_prompt_indices)
            if unique_prompt_indices:
                prompt_row_indices = torch.tensor(
                    unique_prompt_indices,
                    dtype=torch.long,
                    device=prompt_ids.device,
                )
                store_transitions(
                    init_output,
                    prompt_ids.index_select(0, prompt_row_indices),
                    token_values=unique_prompt_values,
                    output_row_indices=prompt_row_indices,
                )
        else:
            prompt_transition_row_count = int(prompt_ids.numel())
            store_transitions(init_output, prompt_ids)

        init_token = torch.argmax(init_output.logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, init_token[:, None]], dim=1)
        if use_suffix_hybrid or use_suffix_reserve:
            self.draft.reset()
            self.draft.update_tokens(input_ids[0].tolist())
        current_length_data.fill_(input_ids.shape[1] - 1)
        kwargs = {}
        acceptance_lengths = []
        trace = []
        collect_policy_trace = bool(return_policy_trace)
        idx = -1
        topology_cache = getattr(
            self, "tree_recycling_topology_cache", None
        )
        if topology_cache is None:
            topology_cache = {}
            self.tree_recycling_topology_cache = topology_cache

        for idx in range(max_length - prompt_length):
            generated = int(input_ids.shape[1] - prompt_length)
            if generated >= max_new_tokens or input_ids.shape[1] >= max_length:
                break
            if _has_stop_token(input_ids[0, prompt_length:], stop_token_ids):
                break

            decision = controller.decide(grounding_score, confidence)
            width, depth = self._tree_shape(
                draft_policy,
                grounding_score,
                confidence,
                decision.risk,
                visual_threshold,
                confidence_threshold,
                tree_fixed_width,
                tree_fixed_depth,
                tree_broad_width,
                tree_shallow_depth,
            )
            use_spine = draft_policy == "spine"
            if draft_policy in ("visual-spine", "modal-spine"):
                use_spine = grounding_score >= visual_threshold
            elif draft_policy == "visual-spine-reverse":
                use_spine = grounding_score < visual_threshold
            branch_width = 1 if use_spine else self._tree_branch_width(
                draft_policy,
                grounding_score,
                visual_threshold,
                width,
                tree_fixed_width,
            )
            remaining = int(max_new_tokens - generated)
            is_high_visual = int(grounding_score >= visual_threshold)
            if use_grounded_conditionals:
                conditional_regime = is_high_visual
                if draft_policy == "grounded-backoff-reverse":
                    conditional_regime = 1 - conditional_regime
                transition_bin = 1 + conditional_regime
                fallback_transition_bin = 0
            else:
                transition_bin = is_high_visual if num_transition_bins > 1 else 0
                fallback_transition_bin = None
            if use_fourgram_transitions and input_ids.shape[1] >= 4:
                (
                    root_previous_previous_previous_token,
                    root_previous_previous_token,
                    root_previous_token,
                    root_token,
                ) = [int(token) for token in input_ids[0, -4:].tolist()]
            elif use_trigram_transitions and input_ids.shape[1] >= 3:
                (
                    root_previous_previous_token,
                    root_previous_token,
                    root_token,
                ) = [int(token) for token in input_ids[0, -3:].tolist()]
                root_previous_previous_previous_token = None
            elif use_context_transitions and input_ids.shape[1] >= 2:
                root_previous_token, root_token = [
                    int(token) for token in input_ids[0, -2:].tolist()
                ]
                root_previous_previous_token = None
                root_previous_previous_previous_token = None
            else:
                root_token = int(input_ids[0, -1].item())
                root_previous_token = None
                root_previous_previous_token = None
                root_previous_previous_previous_token = None
            root_context_key = (
                (root_previous_token, root_token)
                if root_previous_token is not None
                else None
            )
            root_trigram_key = (
                (
                    root_previous_previous_token,
                    root_previous_token,
                    root_token,
                )
                if root_previous_previous_token is not None
                and root_previous_token is not None
                else None
            )
            root_fourgram_key = (
                (
                    root_previous_previous_previous_token,
                    root_previous_previous_token,
                    root_previous_token,
                    root_token,
                )
                if root_previous_previous_previous_token is not None
                and root_previous_previous_token is not None
                and root_previous_token is not None
                else None
            )
            root_transition_persistent = False
            if (
                host_fourgram_transition_scores is not None
                and root_fourgram_key in host_fourgram_transition_scores
            ):
                root_transition_top_probability = float(
                    host_fourgram_transition_scores[root_fourgram_key][0]
                )
                root_transition_context_order = 4
            elif (
                host_trigram_transition_scores is not None
                and root_trigram_key in host_trigram_transition_scores
            ):
                root_transition_top_probability = float(
                    host_trigram_transition_scores[root_trigram_key][0]
                )
                root_transition_context_order = 3
            elif (
                host_context_transition_scores is not None
                and root_context_key in host_context_transition_scores
            ):
                root_transition_top_probability = float(
                    host_context_transition_scores[root_context_key][0]
                )
                root_transition_context_order = 2
            elif (
                host_transition_scores is not None
                and root_token in host_transition_scores
            ):
                root_transition_top_probability = float(
                    host_transition_scores[root_token][0]
                )
                root_transition_context_order = 1
            elif (
                persistent_bank_scores is not None
                and root_token in persistent_bank_scores
            ):
                root_transition_top_probability = float(
                    persistent_bank_scores[root_token][0]
                )
                root_transition_context_order = 1
                root_transition_persistent = True
            else:
                root_transition_top_probability = None
                root_transition_context_order = 0
            root_row_valid_before_backoff = (
                (
                    root_token in host_transitions
                    or (
                        persistent_bank_transitions is not None
                        and root_token in persistent_bank_transitions
                    )
                )
                if host_transitions is not None
                else bool(transition_valid[transition_bin, root_token].item())
            )
            global_backoff_active = bool(
                use_global_backoff
                and not root_row_valid_before_backoff
                and global_candidate_ids
            )
            effective_tree_node_budget = self._effective_tree_node_budget(
                draft_policy,
                tree_node_budget,
                root_transition_top_probability,
                root_transition_context_order,
            )
            effective_tree_depth = depth
            if global_backoff_active:
                global_budget, global_depth = GLOBAL_BACKOFF_POLICIES[
                    draft_policy
                ]
                effective_tree_node_budget = min(
                    effective_tree_node_budget, global_budget
                )
                effective_tree_depth = min(depth, global_depth)
            visual_backoff_eligible = self._visual_lexical_backoff_active(
                draft_policy,
                grounding_score,
                visual_threshold,
                root_row_valid_before_backoff,
            ) and bool(visual_lexical_inventory)
            visual_hst_gate_pass = bool(
                not visual_hst_gate_enabled
                or confidence >= hst_min_confidence
            )
            visual_backoff_active = bool(
                visual_backoff_eligible and visual_hst_gate_pass
            )
            visual_hst_backoff_active = bool(
                visual_backoff_active and visual_hst_enabled and hst_bank is not None
            )
            visual_lexical_backoff_active = bool(
                visual_backoff_active and not visual_hst_enabled
            )
            hst_candidates: List[int] = []
            hst_score_margin = None
            hst_score_entropy = None
            hst_top_probability = None
            if visual_backoff_active:
                candidate_count = min(
                    visual_lexical_width,
                    len(visual_lexical_inventory),
                    matrix_top_k,
                )
                if visual_hst_backoff_active:
                    query_token_vector = embedding_layer.weight[root_token]
                    hst_scores = hst_bank.score(
                        query_token_vector=query_token_vector,
                        query_context_hidden=hst_parent_hidden,
                        config=hst_config,
                        query_candidate_logits=hst_parent_candidate_logits,
                        rank_only=not hst_trace_diagnostics,
                    )
                    hst_top_values, hst_indices = hst_scores.topk(
                        candidate_count
                    )
                    if hst_trace_diagnostics and hst_top_values.numel() >= 2:
                        hst_score_margin = float(
                            (hst_top_values[0] - hst_top_values[1]).item()
                        )
                    if hst_trace_diagnostics:
                        hst_probabilities = torch.softmax(
                            hst_scores.float(), dim=0
                        )
                        hst_top_probability = float(
                            hst_probabilities.max().item()
                        )
                    if hst_trace_diagnostics and hst_probabilities.numel() > 1:
                        hst_score_entropy = float(
                            (
                                -(
                                    hst_probabilities
                                    * hst_probabilities.clamp_min(1e-12).log()
                                ).sum()
                                / torch.log(
                                    torch.tensor(
                                        float(hst_probabilities.numel()),
                                        device=hst_probabilities.device,
                                    )
                                )
                            ).item()
                        )
                    injected_candidate_tensor = (
                        hst_bank.candidate_token_ids.index_select(
                            0, hst_indices
                        )
                    )
                    hst_candidates = (
                        [int(token) for token in injected_candidate_tensor.tolist()]
                        if hst_trace_diagnostics
                        else []
                    )
                else:
                    injected_candidates = visual_lexical_inventory[:candidate_count]
                lexical_row = transitions[transition_bin, root_token]
                if visual_hst_backoff_active:
                    lexical_row[:candidate_count].copy_(
                        injected_candidate_tensor.to(dtype=lexical_row.dtype)
                    )
                else:
                    lexical_row[:candidate_count] = torch.tensor(
                        injected_candidates,
                        dtype=lexical_row.dtype,
                        device=lexical_row.device,
                    )
                transition_valid[transition_bin, root_token] = True
                # The formal path test rejected deeper recursive use.  Keep the
                # injected proposal to one visual first hop only.
                width = candidate_count
                depth = 1
                branch_width = 0
            conditional_count = (
                int(transition_counts[transition_bin, root_token].item())
                if use_grounded_conditionals
                else 0
            )
            conditional_mix_weight = (
                min(0.5, conditional_count / (conditional_count + 2.0))
                if conditional_count > 0
                else 0.0
            )
            conditional_global_overlap = None
            if (
                use_grounded_conditionals
                and bool(transition_valid[transition_bin, root_token].item())
                and bool(transition_valid[0, root_token].item())
            ):
                conditional_tokens = set(
                    transitions[transition_bin, root_token].tolist()
                )
                global_tokens = set(transitions[0, root_token].tolist())
                conditional_global_overlap = len(
                    conditional_tokens & global_tokens
                ) / max(len(conditional_tokens | global_tokens), 1)
            root_residual_candidate_count = 0
            if use_grounded_residual:
                root_candidates = self._transition_candidates(
                    transitions,
                    transition_valid,
                    root_token,
                    transition_bin,
                    width,
                    fallback_bin=0,
                    transition_counts=transition_counts,
                    preserve_fallback_limit=tree_broad_width,
                )
                root_residual_candidate_count = max(
                    len(root_candidates) - min(width, tree_broad_width), 0
                )
            sam_draft_tokens = []
            use_sam_spine = False
            if use_suffix_reserve:
                sam_draft_tokens, _ = self.draft.lookup_tokens(root_token)
                sam_draft_tokens = [
                    int(token) for token in sam_draft_tokens[:4]
                ]
            elif use_suffix_hybrid:
                _, sam_draft, _ = self.draft.lookup(root_token)
                sam_draft_tokens = [
                    int(token)
                    for token in sam_draft[:tree_fixed_depth].detach().cpu().tolist()
                ]
                use_sam_spine = bool(sam_draft_tokens)
                if draft_policy == "visual-hybrid":
                    use_sam_spine = (
                        use_sam_spine and grounding_score < visual_threshold
                    )
                if not use_sam_spine:
                    width, depth = tree_broad_width, tree_shallow_depth
                    branch_width = width
                else:
                    if draft_policy in (
                        "grounded-hybrid",
                        "grounded-hybrid-reverse",
                    ):
                        use_broad_root = grounding_score >= visual_threshold
                        if draft_policy == "grounded-hybrid-reverse":
                            use_broad_root = not use_broad_root
                        width = (
                            tree_broad_width
                            if use_broad_root
                            else tree_fixed_width
                        )
                    else:
                        width = tree_broad_width
                    depth = tree_fixed_depth
                    branch_width = 1
                    chain = [root_token] + sam_draft_tokens
                    for source_token, next_token in zip(chain[:-1], chain[1:]):
                        self._promote_transition(
                            transitions,
                            transition_valid,
                            transition_bin,
                            source_token,
                            next_token,
                        )
            tree_metadata = (
                {
                    "_candidate_trace_diagnostics": bool(
                        collect_candidate_trace
                    )
                }
                if verification_trace_diagnostics
                or use_context_transitions
                or collect_candidate_trace
                else None
            )
            if "-trigram-residual2-" in draft_policy:
                context_candidate_mode = "residual2"
            elif "-trigram-fusion-uniform-" in draft_policy:
                context_candidate_mode = "fusion_uniform"
            elif "-trigram-fusion55-" in draft_policy:
                context_candidate_mode = "fusion55"
            elif "-trigram-fusion-adaptive-" in draft_policy:
                context_candidate_mode = "fusion_adaptive"
            elif "-trigram-fusion" in draft_policy:
                context_candidate_mode = "fusion"
            else:
                context_candidate_mode = "strict"
            flat_tokens, tree_mask, tree_positions, paths = self._build_tree(
                root_token,
                transitions,
                transition_valid,
                width,
                effective_tree_depth,
                effective_tree_node_budget,
                image_token_id,
                branch_width=branch_width,
                transition_bin=transition_bin,
                fallback_transition_bin=fallback_transition_bin,
                transition_counts=transition_counts,
                preserve_fallback_limit=(
                    tree_broad_width if use_grounded_residual else 0
                ),
                topology_cache=topology_cache,
                host_transitions=host_transitions,
                host_transition_scores=host_transition_scores,
                root_previous_token=root_previous_token,
                root_previous_previous_token=(
                    root_previous_previous_token
                ),
                root_previous_previous_previous_token=(
                    root_previous_previous_previous_token
                ),
                host_context_transitions=host_context_transitions,
                host_context_transition_scores=(
                    host_context_transition_scores
                ),
                host_trigram_transitions=host_trigram_transitions,
                host_trigram_transition_scores=(
                    host_trigram_transition_scores
                ),
                host_fourgram_transitions=host_fourgram_transitions,
                host_fourgram_transition_scores=(
                    host_fourgram_transition_scores
                ),
                host_persistent_transitions=(
                    persistent_shadow_transitions
                    if persistent_shadow_transitions is not None
                    else persistent_bank_transitions
                ),
                host_global_candidates=(
                    global_candidate_ids if global_backoff_active else None
                ),
                host_global_candidate_scores=(
                    global_candidate_scores
                    if global_backoff_active
                    else None
                ),
                context_candidate_mode=context_candidate_mode,
                metadata_out=tree_metadata,
                priority_layout=draft_policy.startswith("rank-prior-"),
                score_priority_layout=use_score_priority,
                host_persistent_transition_scores=(
                    persistent_shadow_scores
                    if persistent_shadow_scores is not None
                    else persistent_bank_scores
                ),
                score_hit_masses=self._score_priority_hit_masses(
                    draft_policy,
                    root_transition_context_order,
                ),
                copy_candidate_rows=not use_hotpath,
                fast_score_priority_layout=(
                    "-hotpath-cpp-" in draft_policy
                ),
                reserved_path_tokens=(
                    sam_draft_tokens if use_suffix_reserve else None
                ),
            )
            num_nodes = len(flat_tokens) - 1
            selective_u_paths = []
            selective_gc_paths = []
            selective_u_root_candidates = []
            selective_gc_root_candidates = []
            selective_v_root_candidates = []
            selective_uv_root_candidates = []
            selective_u_source_available = False
            selective_u_available_before_request = False
            selective_u_row_top_probability_before_request = None
            selective_u_row_top_probability = None
            selective_u_persistent_row_top_probability = None
            selective_g_source_available = False
            selective_c_source_available = False
            selective_v_source_available = False
            selective_u_node_count = 0
            selective_gc_node_count = 0
            selective_uv_candidate_budget = 0
            selective_uv_added_candidate = None
            selective_uv_displaced_candidate = None
            if collect_selective_reuse:
                persistent_u_rows = (
                    persistent_shadow_transitions
                    if persistent_shadow_transitions is not None
                    else persistent_bank_transitions
                )
                persistent_u_scores = (
                    persistent_shadow_scores
                    if persistent_shadow_scores is not None
                    else persistent_bank_scores
                )
                selective_u_source_available = bool(
                    (host_transitions is not None and root_token in host_transitions)
                    or (
                        persistent_u_rows is not None
                        and root_token in persistent_u_rows
                    )
                )
                selective_u_available_before_request = bool(
                    root_token in candidate_trace_persistent_keys_at_start
                )
                selective_u_row_top_probability_before_request = (
                    selective_u_top_probabilities_at_start.get(root_token)
                )
                if (
                    host_transition_scores is not None
                    and root_token in host_transition_scores
                    and host_transition_scores[root_token]
                ):
                    selective_u_row_top_probability = float(
                        host_transition_scores[root_token][0]
                    )
                if (
                    persistent_u_scores is not None
                    and root_token in persistent_u_scores
                    and persistent_u_scores[root_token]
                ):
                    selective_u_persistent_row_top_probability = float(
                        persistent_u_scores[root_token][0]
                    )
                selective_g_source_available = bool(
                    host_trigram_transitions is not None
                    and root_trigram_key is not None
                    and root_trigram_key in host_trigram_transitions
                )
                selective_c_source_available = bool(
                    host_context_transitions is not None
                    and root_context_key is not None
                    and root_context_key in host_context_transitions
                )
                selective_v_source_available = bool(selective_visual_inventory)

                def build_source_shadow(
                    *,
                    unigram_rows,
                    unigram_scores,
                    persistent_rows=None,
                    persistent_scores=None,
                    context_rows=None,
                    context_scores=None,
                    trigram_rows=None,
                    trigram_scores=None,
                ):
                    shadow_flat, _, _, shadow_paths = self._build_tree(
                        root_token,
                        transitions,
                        transition_valid,
                        width,
                        effective_tree_depth,
                        effective_tree_node_budget,
                        image_token_id,
                        branch_width=branch_width,
                        transition_bin=transition_bin,
                        fallback_transition_bin=fallback_transition_bin,
                        transition_counts=transition_counts,
                        preserve_fallback_limit=(
                            tree_broad_width if use_grounded_residual else 0
                        ),
                        topology_cache=topology_cache,
                        host_transitions=unigram_rows,
                        host_transition_scores=unigram_scores,
                        root_previous_token=root_previous_token,
                        root_previous_previous_token=(
                            root_previous_previous_token
                        ),
                        root_previous_previous_previous_token=(
                            root_previous_previous_previous_token
                        ),
                        host_context_transitions=context_rows,
                        host_context_transition_scores=context_scores,
                        host_trigram_transitions=trigram_rows,
                        host_trigram_transition_scores=trigram_scores,
                        host_persistent_transitions=persistent_rows,
                        host_persistent_transition_scores=persistent_scores,
                        context_candidate_mode=context_candidate_mode,
                        priority_layout=draft_policy.startswith("rank-prior-"),
                        score_priority_layout=use_score_priority,
                        score_hit_masses=self._score_priority_hit_masses(
                            draft_policy,
                            root_transition_context_order,
                        ),
                        copy_candidate_rows=True,
                        fast_score_priority_layout=False,
                    )
                    root_candidates = []
                    seen_root_candidates = set()
                    for path in shadow_paths:
                        if not path:
                            continue
                        candidate = int(shadow_flat[path[0]])
                        if candidate in seen_root_candidates:
                            continue
                        seen_root_candidates.add(candidate)
                        root_candidates.append(candidate)
                        if len(root_candidates) >= 8:
                            break
                    return (
                        self._tree_path_token_ids(shadow_flat, shadow_paths),
                        root_candidates,
                        len(shadow_flat) - 1,
                    )

                (
                    selective_u_paths,
                    selective_u_root_candidates,
                    selective_u_node_count,
                ) = build_source_shadow(
                    unigram_rows=host_transitions,
                    unigram_scores=host_transition_scores,
                    persistent_rows=persistent_u_rows,
                    persistent_scores=persistent_u_scores,
                )
                (
                    selective_gc_paths,
                    selective_gc_root_candidates,
                    selective_gc_node_count,
                ) = build_source_shadow(
                    # A non-None empty table selects the host-table path while
                    # deliberately removing the unigram source.
                    unigram_rows={},
                    unigram_scores={},
                    context_rows=host_context_transitions,
                    context_scores=host_context_transition_scores,
                    trigram_rows=host_trigram_transitions,
                    trigram_scores=host_trigram_transition_scores,
                )
                selective_v_root_candidates = selective_visual_inventory[:8]
                # Freeze a one-slot visual injection at an exactly matched
                # root-candidate budget.  This shadow candidate row is never
                # used by the live decoder.
                selective_uv_candidate_budget = min(
                    8, len(selective_u_root_candidates)
                )
                if selective_uv_candidate_budget > 0:
                    selective_uv_root_candidates = fuse_equal_budget_candidates(
                        selective_u_root_candidates,
                        selective_visual_inventory,
                        budget=selective_uv_candidate_budget,
                        inventory_slots=1,
                    )
                    added = [
                        token
                        for token in selective_uv_root_candidates
                        if token not in selective_u_root_candidates
                    ]
                    displaced = [
                        token
                        for token in selective_u_root_candidates[
                            :selective_uv_candidate_budget
                        ]
                        if token not in selective_uv_root_candidates
                    ]
                    selective_uv_added_candidate = added[0] if added else None
                    selective_uv_displaced_candidate = (
                        displaced[0] if displaced else None
                    )
            cover_probe_active = bool(
                cover_enabled
                and grounding_score >= cover_probe_visual_threshold
                and num_nodes > 0
                and remaining > 1
            )
            selective_probe_active = bool(
                packed_selective_reuse_probes and remaining > 0
            )
            packed_probe_active = bool(
                cover_probe_active or selective_probe_active
            )
            max_path_depth = max((len(path) for path in paths), default=0)
            used_path_depth = min(max_path_depth, max(remaining - 1, 0))
            record = ({
                "iteration": len(trace),
                **decision.to_dict(),
                "requested_draft_policy": requested_draft_policy,
                "cover_enabled": bool(cover_enabled),
                "cover_probe_active": cover_probe_active,
                "selective_reuse_diagnostics": bool(
                    collect_selective_reuse
                ),
                "selective_reuse_probe_mode": (
                    selective_reuse_probe_mode
                    if collect_selective_reuse
                    else None
                ),
                "visual_probe_active": packed_probe_active,
                "cover_probe_visual_threshold": cover_probe_visual_threshold,
                "cover_min_jsd": cover_min_jsd,
                "tree_width": int(width),
                "tree_branch_width": int(branch_width),
                "tree_depth": int(effective_tree_depth),
                "raw_draft_len": int(num_nodes),
                "used_draft_len": int(used_path_depth),
                "verified_candidate_tree_nodes": int(
                    num_nodes if remaining > 1 else 0
                ),
                "verified_probe_nodes": int(
                    cover_num_probes
                    if packed_probe_active
                    else 0
                ),
                "verified_tree_nodes": int(
                    (
                        num_nodes
                        + (cover_num_probes if packed_probe_active else 0)
                    )
                    if num_nodes > 0 and remaining > 1
                    else 0
                ),
                "transition_bin": int(transition_bin),
                "num_transition_bins": int(num_transition_bins),
                "host_transition_table": bool(use_host_transitions),
                "context_candidate_mode": context_candidate_mode,
                "persistent_unigram_enabled": bool(
                    use_persistent_unigram or use_persistent_bank
                ),
                "persistent_unigram_mode": (
                    "bank"
                    if use_persistent_bank
                    else (
                        "latest-ngram"
                        if use_persistent_ngram
                        else (
                            "latest-committed"
                            if use_committed_transitions_only
                            else (
                                "latest-shadow"
                                if use_persistent_shadow
                                else "none"
                                if not use_persistent_unigram
                                else "latest"
                            )
                        )
                    )
                ),
                "transition_update_scope": (
                    "committed"
                    if use_committed_transitions_only
                    else "full-tree"
                ),
                "prompt_transition_refresh_mode": (
                    "unseen-only"
                    if skip_cached_prompt_transitions
                    else "all-unique"
                ),
                "prompt_transition_row_count": int(
                    prompt_transition_row_count
                ),
                "persistent_unigram_size_at_start": int(
                    persistent_unigram_size_at_start
                ),
                "persistent_unigram_size": int(
                    len(host_transitions)
                    if use_persistent_unigram
                    else (
                        len(persistent_bank_transitions)
                        if persistent_bank_transitions is not None
                        else 0
                    )
                ),
                "persistent_context_size_at_start": int(
                    persistent_context_size_at_start
                ),
                "persistent_trigram_size_at_start": int(
                    persistent_trigram_size_at_start
                ),
                "persistent_context_size": int(
                    len(host_context_transitions)
                    if use_persistent_ngram
                    else 0
                ),
                "persistent_trigram_size": int(
                    len(host_trigram_transitions)
                    if use_persistent_ngram
                    else 0
                ),
                "persistent_shadow_size": int(
                    len(persistent_shadow_transitions)
                    if persistent_shadow_transitions is not None
                    else 0
                ),
                "root_transition_persistent": bool(
                    root_transition_persistent
                ),
                "root_transition_top_probability": (
                    root_transition_top_probability
                ),
                "root_transition_context_order": int(
                    root_transition_context_order
                ),
                "global_backoff_enabled": bool(use_global_backoff),
                "global_backoff_active": global_backoff_active,
                "global_backoff_candidate_count": int(
                    len(global_candidate_ids) if global_backoff_active else 0
                ),
                "effective_tree_node_budget": int(
                    effective_tree_node_budget
                ),
                "conditional_transition_count": int(conditional_count),
                "conditional_mix_weight": float(conditional_mix_weight),
                "conditional_global_overlap": conditional_global_overlap,
                "root_residual_candidate_count": int(
                    root_residual_candidate_count
                ),
                "visual_lexical_enabled": bool(visual_lexical_enabled),
                "visual_lexical_inventory_size": int(
                    len(visual_lexical_inventory)
                ),
                "visual_lexical_root_row_valid": bool(
                    root_row_valid_before_backoff
                ),
                "visual_lexical_backoff_active": bool(
                    visual_lexical_backoff_active
                ),
                "visual_hst_enabled": bool(visual_hst_enabled),
                "visual_hst_backoff_eligible": bool(
                    visual_backoff_eligible and visual_hst_enabled
                ),
                "visual_hst_gate_enabled": bool(visual_hst_gate_enabled),
                "visual_hst_gate_pass": bool(visual_hst_gate_pass),
                "visual_hst_min_confidence": (
                    hst_min_confidence if visual_hst_gate_enabled else None
                ),
                "visual_hst_online_update": bool(hst_online_update),
                "visual_hst_backoff_active": bool(visual_hst_backoff_active),
                "visual_hst_config": hst_config.key if visual_hst_enabled else None,
                "visual_hst_source_count": int(
                    hst_bank.source_count if hst_bank is not None else 0
                ),
                "visual_hst_candidate_count": int(
                    candidate_count if visual_hst_backoff_active else 0
                ),
                "visual_hst_candidate_ids": (
                    list(hst_candidates) if hst_trace_diagnostics else []
                ),
                "visual_hst_static_candidate_ids": (
                    list(visual_lexical_inventory[: len(hst_candidates)])
                    if hst_trace_diagnostics
                    else []
                ),
                "visual_hst_score_margin": hst_score_margin,
                "visual_hst_score_entropy": hst_score_entropy,
                "visual_hst_top_probability": hst_top_probability,
                "visual_lexical_width": int(
                    visual_lexical_width if visual_lexical_enabled else 0
                ),
                "sam_draft_len": int(len(sam_draft_tokens)),
                "used_sam_spine": bool(use_sam_spine),
                "remaining_tokens": remaining,
                "verification_margin_threshold": (
                    verification_margin_threshold
                ),
                "verification_boundary_recheck": False,
                "verification_compact_path_repair": False,
                "verification_compact_root_recheck": False,
            } if collect_policy_trace else {})
            if collect_selective_reuse:
                record.update(
                    {
                        "selective_source_top_k": 8,
                        "selective_generated_offset": int(generated),
                        "selective_u_source_available": bool(
                            selective_u_source_available
                        ),
                        "selective_u_available_before_request": bool(
                            selective_u_available_before_request
                        ),
                        "selective_u_row_top_probability_before_request": (
                            selective_u_row_top_probability_before_request
                        ),
                        "selective_u_row_top_probability": (
                            selective_u_row_top_probability
                        ),
                        "selective_u_persistent_row_top_probability": (
                            selective_u_persistent_row_top_probability
                        ),
                        "selective_g_source_available": bool(
                            selective_g_source_available
                        ),
                        "selective_c_source_available": bool(
                            selective_c_source_available
                        ),
                        "selective_gc_source_available": bool(
                            selective_g_source_available
                            or selective_c_source_available
                        ),
                        "selective_u_root_candidate_token_ids": (
                            selective_u_root_candidates
                        ),
                        "selective_gc_root_candidate_token_ids": (
                            selective_gc_root_candidates
                        ),
                        "selective_v_source": "prompt_visual_max",
                        "selective_v_source_available": bool(
                            selective_v_source_available
                        ),
                        "selective_v_root_candidate_token_ids": (
                            selective_v_root_candidates
                        ),
                        "selective_uv_root_candidate_token_ids": (
                            selective_uv_root_candidates
                        ),
                        "selective_uv_candidate_budget": int(
                            selective_uv_candidate_budget
                        ),
                        "selective_uv_visual_slots": 1,
                        "selective_uv_added_candidate_token_id": (
                            selective_uv_added_candidate
                        ),
                        "selective_uv_displaced_candidate_token_id": (
                            selective_uv_displaced_candidate
                        ),
                        "selective_u_tree_nodes": int(
                            selective_u_node_count
                        ),
                        "selective_gc_tree_nodes": int(
                            selective_gc_node_count
                        ),
                        "_selective_u_path_token_ids": selective_u_paths,
                        "_selective_gc_path_token_ids": selective_gc_paths,
                    }
                )
            if collect_candidate_trace:
                root_source_candidate_token_ids = {}
                root_source_candidate_scores = {}

                def add_root_source(name, rows, scores, key):
                    if rows is None or key is None or key not in rows:
                        return
                    root_source_candidate_token_ids[name] = [
                        int(token) for token in rows[key]
                    ]
                    if scores is not None and key in scores:
                        root_source_candidate_scores[name] = [
                            float(score) for score in scores[key]
                        ]

                add_root_source(
                    "G",
                    host_trigram_transitions,
                    host_trigram_transition_scores,
                    root_trigram_key,
                )
                add_root_source(
                    "C",
                    host_context_transitions,
                    host_context_transition_scores,
                    root_context_key,
                )
                add_root_source(
                    "U",
                    host_transitions,
                    host_transition_scores,
                    root_token,
                )
                add_root_source(
                    "persistent_U_bank",
                    persistent_bank_transitions,
                    persistent_bank_scores,
                    root_token,
                )

                node_parents = list(tree_metadata.get("node_parents", []))
                parent_indices = {int(parent) for parent in node_parents}
                leaf_indices = [
                    node_index
                    for node_index in range(1, len(flat_tokens))
                    if node_index not in parent_indices
                ]
                leaf_paths = [
                    paths[node_index - 1] for node_index in leaf_indices
                ]
                record.update(
                    {
                        "root_token_id": int(root_token),
                        "root_context_token_ids": [
                            int(token)
                            for token in (
                                root_previous_previous_token,
                                root_previous_token,
                                root_token,
                            )
                            if token is not None
                        ],
                        "generated_prefix_tail_token_ids": [
                            int(token)
                            for token in input_ids[
                                0, max(prompt_length, input_ids.shape[1] - 16) :
                            ].tolist()
                        ],
                        "root_source_candidate_token_ids": (
                            root_source_candidate_token_ids
                        ),
                        "root_source_candidate_scores": (
                            root_source_candidate_scores
                        ),
                        "root_u_available_before_request": bool(
                            root_token
                            in candidate_trace_persistent_keys_at_start
                        ),
                        "fused_root_candidate_token_ids": [
                            int(flat_tokens[node_index])
                            for node_index, parent_index in enumerate(
                                node_parents, start=1
                            )
                            if int(parent_index) == 0
                        ],
                        "candidate_tree_token_ids": [
                            int(token) for token in flat_tokens
                        ],
                        "candidate_tree_parent_indices": [
                            -1,
                            *[int(parent) for parent in node_parents],
                        ],
                        "candidate_tree_depths": [
                            0,
                            *[
                                int(node_depth)
                                for node_depth in tree_metadata.get(
                                    "node_depths", []
                                )
                            ],
                        ],
                        "candidate_tree_ranks": [
                            int(rank)
                            for rank in tree_metadata.get("node_ranks", [])
                        ],
                        "candidate_leaf_path_token_ids": [
                            [
                                int(flat_tokens[node_index])
                                for node_index in path
                            ]
                            for path in leaf_paths
                        ],
                        "suffix_candidate_token_ids": [
                            int(token) for token in sam_draft_tokens
                        ],
                    }
                )
            if collect_policy_trace and not trace and calibrator is not None:
                record["grounding_calibration"] = calibrator.diagnostics()

            if num_nodes == 0 or remaining <= 1:
                root_input = input_ids[:, -1:]
                if selective_probe_active:
                    kv_base_len = int(current_length_data[0].item())
                    verifier_input = root_input.expand(
                        1, cover_num_probes + 1
                    )
                    probe_attention_mask = build_counterfactual_attention_mask(
                        kv_base_len,
                        visual_probe_layout,
                        self.base_model.dtype,
                        input_ids.device,
                    )
                    position_ids = torch.full(
                        (cover_num_probes + 1,),
                        kv_base_len,
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    position_ids = (
                        position_ids.unsqueeze(0) + self.base_model.rope_deltas
                    )
                    position_ids = position_ids.unsqueeze(0).expand(
                        3, -1, -1
                    )
                    output = self.base_model(
                        input_ids=verifier_input,
                        attention_mask=probe_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                        output_hidden_states=need_all_hidden_states,
                        output_last_hidden_state=need_hidden_states,
                    )
                    record.update(
                        self._visual_probe_metrics(
                            output.logits[0], matrix_top_k
                        )
                    )
                    record.update(
                        {
                            "visual_probe_num_regions": int(
                                cover_num_probes
                            ),
                            "visual_probe_num_visual_tokens": int(
                                visual_probe_layout.num_visual_tokens
                            ),
                            "visual_probe_used_grid_metadata": bool(
                                visual_probe_layout.used_grid_metadata
                            ),
                        }
                    )
                else:
                    output = self.base_model(
                        input_ids=root_input,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                        output_hidden_states=need_all_hidden_states,
                        output_last_hidden_state=need_hidden_states,
                    )
                store_transitions(
                    output,
                    input_ids[0, -1:],
                    token_values=[root_token],
                    previous_token_values=[root_previous_token]
                    if use_context_transitions
                    else None,
                    previous_previous_token_values=[
                        root_previous_previous_token
                    ]
                    if use_trigram_transitions
                    else None,
                    previous_previous_previous_token_values=[
                        root_previous_previous_previous_token
                    ]
                    if use_fourgram_transitions
                    else None,
                )
                next_id = torch.argmax(output.logits[:, 0, :], dim=-1)
                input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)
                if verification_trace_diagnostics:
                    top = torch.topk(
                        output.logits[0, 0].float(), k=2
                    )
                    top_values = top.values
                    record.update(
                        {
                            "verification_committed_token_ids": [
                                int(next_id.item())
                            ],
                            "verification_token_margins": [
                                float((top_values[0] - top_values[1]).item())
                            ],
                            "verification_top_token_ids": [
                                [int(token_id) for token_id in top.indices.tolist()]
                            ],
                            "verification_min_margin": float(
                                (top_values[0] - top_values[1]).item()
                            ),
                            "verification_accepted_path": [],
                        }
                    )
                if use_suffix_hybrid or use_suffix_reserve:
                    self.draft.update_tokens(next_id.tolist())
                current_length_data.fill_(input_ids.shape[1] - 1)
                accepted_path = []
                accepted_tokens = []
                correction = int(next_id.item())
                accept_len = 0
                query_index = 0
            else:
                kv_base_len = int(current_length_data[0].item())
                tree_input = torch.tensor(
                    [flat_tokens], dtype=input_ids.dtype, device=input_ids.device
                )
                verifier_input = tree_input
                verifier_positions = tree_positions
                probe_attention_mask = None
                if packed_probe_active:
                    probe_input = torch.full(
                        (1, cover_num_probes),
                        root_token,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    verifier_input = torch.cat([tree_input, probe_input], dim=1)
                    verifier_positions = torch.cat(
                        [
                            tree_positions,
                            torch.zeros(
                                cover_num_probes,
                                dtype=tree_positions.dtype,
                                device=tree_positions.device,
                            ),
                        ]
                    )
                    probe_attention_mask = build_tree_counterfactual_attention_mask(
                        kv_base_len,
                        tree_mask,
                        visual_probe_layout,
                        self.base_model.dtype,
                        input_ids.device,
                    )
                position_ids = verifier_positions + kv_base_len
                if arch == "Qwen2_5_VLForConditionalGeneration":
                    position_ids = (
                        position_ids.unsqueeze(0) + self.base_model.rope_deltas
                    )
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                else:
                    position_ids = position_ids.unsqueeze(0)
                if packed_probe_active:
                    output = self.base_model(
                        input_ids=verifier_input,
                        attention_mask=probe_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                        output_hidden_states=(
                            need_all_hidden_states
                            or verification_layer_diagnostics
                        ),
                        output_last_hidden_state=need_hidden_states,
                    )
                else:
                    if hasattr(self.base_model, "language_model"):
                        language_model = self.base_model.language_model
                        tree_mask_model = getattr(
                            language_model, "model", language_model
                        )
                    else:
                        tree_mask_model = self.base_model.model
                    tree_mask_model.tree_mask = tree_mask
                    try:
                        output = self.base_model(
                            input_ids=verifier_input,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            return_dict=True,
                            use_cache=True,
                            output_hidden_states=(
                                need_all_hidden_states
                                or verification_layer_diagnostics
                            ),
                            output_last_hidden_state=need_hidden_states,
                        )
                    finally:
                        tree_mask_model.tree_mask = None

                queued_predictions = (
                    torch.argmax(
                        output.logits[0, : len(flat_tokens)], dim=-1
                    )
                    if use_hotpath
                    else None
                )
                if not use_committed_transitions_only:
                    store_transitions(
                        output,
                        tree_input[0],
                        token_values=flat_tokens,
                        previous_token_values=(
                            tree_metadata["semantic_previous_tokens"]
                            if use_context_transitions
                            else None
                        ),
                        previous_previous_token_values=(
                            tree_metadata[
                                "semantic_previous_previous_tokens"
                            ]
                            if use_trigram_transitions
                            else None
                        ),
                        previous_previous_previous_token_values=(
                            tree_metadata[
                                "semantic_previous_previous_previous_tokens"
                            ]
                            if use_fourgram_transitions
                            else None
                        ),
                    )
                if packed_probe_active:
                    actual_tree_length = int(tree_input.shape[1])
                    visual_probe_logits = torch.cat(
                        [
                            output.logits[0, :1],
                            output.logits[0, actual_tree_length:],
                        ],
                        dim=0,
                    )
                    record.update(
                        self._visual_probe_metrics(
                            visual_probe_logits, matrix_top_k
                        )
                    )
                    record.update(
                        {
                            "visual_probe_num_regions": int(
                                cover_num_probes
                            ),
                            "visual_probe_num_visual_tokens": int(
                                visual_probe_layout.num_visual_tokens
                            ),
                            "visual_probe_used_grid_metadata": bool(
                                visual_probe_layout.used_grid_metadata
                            ),
                        }
                    )
                if cover_probe_active:
                    previous_row = transitions[
                        transition_bin, root_token
                    ].clone()
                    recycled = torch.tensor(
                        cover_candidates(
                            visual_probe_logits,
                            matrix_top_k,
                            anchor_fraction=cover_anchor_fraction,
                        ),
                        dtype=transitions.dtype,
                        device=transitions.device,
                    )
                    view_jsd = multiview_jsd(visual_probe_logits)
                    recycle_applied = view_jsd >= cover_min_jsd
                    if recycle_applied:
                        transitions[transition_bin, root_token].copy_(recycled)
                        transition_valid[transition_bin, root_token] = True
                    record.update(
                        {
                            "cover_num_probes": int(cover_num_probes),
                            "cover_anchor_fraction": cover_anchor_fraction,
                            "cover_view_jsd": view_jsd,
                            "cover_topk_union_size": topk_union_size(
                                visual_probe_logits, matrix_top_k
                            ),
                            "cover_recycle_applied": recycle_applied,
                            "cover_candidate_row_changed": recycle_applied
                            and not torch.equal(previous_row, recycled),
                        }
                    )
                (
                    accepted_path,
                    accepted_tokens,
                    correction,
                    query_index,
                ) = self._best_verified_path(
                    output.logits[:, : len(flat_tokens)],
                    flat_tokens,
                    paths,
                    remaining,
                    predictions=queued_predictions,
                )
                accept_len = len(accepted_path)
                if use_committed_transitions_only:
                    committed_rows = [0, *accepted_path]
                    committed_row_indices = torch.tensor(
                        committed_rows,
                        dtype=torch.long,
                        device=tree_input.device,
                    )

                    def committed_metadata_values(name):
                        values = tree_metadata[name]
                        return [values[index] for index in committed_rows]

                    store_transitions(
                        output,
                        tree_input[0].index_select(
                            0, committed_row_indices
                        ),
                        token_values=[flat_tokens[index] for index in committed_rows],
                        previous_token_values=(
                            committed_metadata_values(
                                "semantic_previous_tokens"
                            )
                            if use_context_transitions
                            else None
                        ),
                        previous_previous_token_values=(
                            committed_metadata_values(
                                "semantic_previous_previous_tokens"
                            )
                            if use_trigram_transitions
                            else None
                        ),
                        previous_previous_previous_token_values=(
                            committed_metadata_values(
                                "semantic_previous_previous_previous_tokens"
                            )
                            if use_fourgram_transitions
                            else None
                        ),
                        output_row_indices=committed_row_indices,
                    )
                compact_path_repair_applied = False
                verification_committed_logits_override = None
                if verification_compact_root_margin_threshold > 0.0:
                    guard_top_values = torch.topk(
                        output.logits[0, 0].float(), k=2
                    ).values
                    guard_root_margin = float(
                        (guard_top_values[0] - guard_top_values[1]).item()
                    )
                else:
                    guard_root_margin = None
                if (
                    guard_root_margin is not None
                    and guard_root_margin
                    < verification_compact_root_margin_threshold
                ):
                    packed_root_prediction = int(
                        torch.argmax(output.logits[0, 0]).item()
                    )
                    current_length_data.fill_(kv_base_len)
                    output = self.base_model(
                        input_ids=tree_input[:, :1],
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=need_all_hidden_states,
                        output_last_hidden_state=need_hidden_states,
                    )
                    store_transitions(output, tree_input[0, :1])
                    correction = int(
                        torch.argmax(output.logits[0, 0]).item()
                    )
                    accepted_path = []
                    accepted_tokens = []
                    accept_len = 0
                    query_index = 0
                    record.update(
                        {
                            "verification_compact_root_recheck": True,
                            "verification_compact_root_margin": (
                                guard_root_margin
                            ),
                            "verification_compact_root_packed_token_id": (
                                packed_root_prediction
                            ),
                            "verification_compact_root_rechecked_token_id": (
                                correction
                            ),
                            "verification_compact_root_changed_token": bool(
                                correction != packed_root_prediction
                            ),
                        }
                    )
                if verification_compact_path_repair and accept_len > 0:
                    packed_output = output
                    packed_accepted_path = list(accepted_path)
                    packed_accepted_tokens = list(accepted_tokens)
                    packed_accept_len = int(accept_len)
                    packed_correction = int(correction)
                    # The root row is the only packed cache entry retained.
                    # Re-run the selected descendants as one compact causal
                    # path, validate deeper accepted tokens, and commit only
                    # this sibling-free cache prefix.
                    current_length_data.fill_(kv_base_len + 1)
                    repair_input = tree_input[:, packed_accepted_path]
                    repair_output = self.base_model(
                        input_ids=repair_input,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=need_all_hidden_states,
                        output_last_hidden_state=need_hidden_states,
                    )
                    repair_predictions = torch.argmax(
                        repair_output.logits[0], dim=-1
                    ).tolist()
                    repaired_accept_len = 1
                    for token_offset in range(1, packed_accept_len):
                        if (
                            int(repair_predictions[token_offset - 1])
                            != packed_accepted_tokens[token_offset]
                        ):
                            break
                        repaired_accept_len += 1
                    accepted_path = packed_accepted_path[:repaired_accept_len]
                    accepted_tokens = packed_accepted_tokens[
                        :repaired_accept_len
                    ]
                    accept_len = repaired_accept_len
                    correction = int(
                        repair_predictions[repaired_accept_len - 1]
                    )
                    current_length_data.fill_(
                        kv_base_len + 1 + repaired_accept_len
                    )
                    output = repair_output
                    query_index = repaired_accept_len - 1
                    store_transitions(
                        repair_output,
                        repair_input[0, :repaired_accept_len],
                    )
                    verification_committed_logits_override = torch.cat(
                        [
                            packed_output.logits[0, :1],
                            repair_output.logits[0, :repaired_accept_len],
                        ],
                        dim=0,
                    )
                    compact_path_repair_applied = True
                    record.update(
                        {
                            "verification_compact_path_repair": True,
                            "verification_packed_accept_len": (
                                packed_accept_len
                            ),
                            "verification_repaired_accept_len": int(
                                repaired_accept_len
                            ),
                            "verification_compact_path_truncated": bool(
                                repaired_accept_len < packed_accept_len
                            ),
                            "verification_packed_correction_token_id": (
                                packed_correction
                            ),
                            "verification_repaired_correction_token_id": (
                                correction
                            ),
                            "verification_compact_path_changed_correction": (
                                correction != packed_correction
                            ),
                        }
                    )
                if verification_margin_threshold > 0.0:
                    packed_committed_rows = [0, *accepted_path]
                    packed_committed_logits = output.logits[
                        0, packed_committed_rows
                    ].float()
                    packed_top_values = torch.topk(
                        packed_committed_logits, k=2, dim=-1
                    ).values
                    packed_committed_margins = (
                        packed_top_values[:, 0] - packed_top_values[:, 1]
                    ).tolist()
                    packed_root_logits = packed_committed_logits[0]
                    packed_root_margin = float(
                        packed_committed_margins[0]
                    )
                    packed_min_margin = min(
                        float(value)
                        for value in packed_committed_margins
                    )
                else:
                    packed_root_logits = None
                    packed_root_margin = None
                    packed_min_margin = None
                    packed_committed_margins = []
                if (
                    packed_min_margin is not None
                    and packed_min_margin < verification_margin_threshold
                ):
                    packed_output = output
                    packed_accept_len = len(accepted_path)
                    packed_committed_token_ids = [
                        *accepted_tokens,
                        int(correction),
                    ]
                    packed_root_prediction = int(
                        torch.argmax(packed_root_logits).item()
                    )
                    # A q_len=1 root recomputation is not sufficient here:
                    # earlier accepted tree nodes may already have left
                    # q_len-dependent numerical drift in the active KV cache.
                    # Restore the latest exact checkpoint and replay only the
                    # generated suffix since that checkpoint with q_len=1.
                    output, replayed_tokens = rebase_exact_prefix(
                        input_ids
                    )
                    if verification_layer_diagnostics:
                        record["verification_layer_diagnostics"] = (
                            self._layerwise_verification_diagnostics(
                                packed_output,
                                output,
                            )
                        )
                    store_transitions(output, tree_input[0, :1])
                    correction = int(
                        torch.argmax(output.logits[0, -1]).item()
                    )
                    single_recheck_prediction = correction
                    recheck_changed_token = bool(
                        correction != packed_root_prediction
                    )
                    prefix_rebased = True
                    rebased_root_prediction = correction
                    accepted_path = []
                    accepted_tokens = []
                    query_index = 0
                    accept_len = 0
                    record.update(
                        {
                            "verification_boundary_recheck": True,
                            "verification_packed_root_margin": (
                                packed_root_margin
                            ),
                            "verification_packed_min_margin": (
                                packed_min_margin
                            ),
                            "verification_packed_token_margins": [
                                float(value)
                                for value in packed_committed_margins
                            ],
                            "verification_packed_accept_len": int(
                                packed_accept_len
                            ),
                            "verification_packed_committed_token_ids": (
                                packed_committed_token_ids
                            ),
                            "verification_packed_root_token_id": (
                                packed_root_prediction
                            ),
                            "verification_rechecked_root_token_id": (
                                single_recheck_prediction
                            ),
                            "verification_recheck_changed_token": bool(
                                recheck_changed_token
                            ),
                            "verification_prefix_rebased": prefix_rebased,
                            "verification_rebased_root_token_id": (
                                rebased_root_prediction
                            ),
                            "verification_rebased_generated_tokens": (
                                int(input_ids.shape[1] - prompt_length)
                                if prefix_rebased
                                else 0
                            ),
                            "verification_replayed_tokens": replayed_tokens,
                        }
                    )
                if verification_trace_diagnostics:
                    if verification_committed_logits_override is not None:
                        committed_logits = (
                            verification_committed_logits_override.float()
                        )
                    else:
                        committed_rows = [0, *accepted_path]
                        committed_logits = output.logits[
                            0, committed_rows
                        ].float()
                    top = torch.topk(
                        committed_logits, k=2, dim=-1
                    )
                    top_values = top.values
                    margins = (top_values[:, 0] - top_values[:, 1]).tolist()
                    record.update(
                        {
                            "verification_committed_token_ids": [
                                *accepted_tokens,
                                int(correction),
                            ],
                            "verification_token_margins": [
                                float(value) for value in margins
                            ],
                            "verification_top_token_ids": [
                                [int(token_id) for token_id in row]
                                for row in top.indices.tolist()
                            ],
                            "verification_min_margin": min(
                                (float(value) for value in margins),
                                default=None,
                            ),
                            "verification_accepted_path": list(accepted_path),
                            "verification_accepted_ranks": (
                                [
                                    int(tree_metadata["node_ranks"][node_index])
                                    for node_index in accepted_path
                                ]
                                if tree_metadata is not None
                                else []
                            ),
                        }
                    )
                if visual_hst_backoff_active and hst_trace_diagnostics:
                    root_logits = output.logits[0, 0].float()
                    root_prediction = int(torch.argmax(root_logits).item())
                    root_top_values = torch.topk(root_logits, k=2).values
                    record["visual_hst_root_logit_margin"] = float(
                        (root_top_values[0] - root_top_values[1]).item()
                    )
                    record["visual_hst_root_target_rank"] = (
                        hst_candidates.index(root_prediction) + 1
                        if root_prediction in hst_candidates
                        else None
                    )

                cache_compaction_applied = bool(
                    not compact_path_repair_applied
                    and self._requires_cache_compaction(accepted_path)
                )
                if cache_compaction_applied:
                    select_indices = torch.tensor(
                        [kv_base_len]
                        + [kv_base_len + node for node in accepted_path],
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    for cache_data in past_key_values_data:
                        selected = cache_data.index_select(
                            -2, select_indices.to(cache_data.device)
                        )
                        destination = cache_data.narrow(
                            -2, kv_base_len, selected.shape[-2]
                        )
                        destination.copy_(selected, non_blocking=True)
                if collect_policy_trace:
                    record["cache_compaction_applied"] = bool(
                        cache_compaction_applied
                    )
                current_length_data.fill_(kv_base_len + len(accepted_path) + 1)
                tokens_to_add = torch.tensor(
                    [accepted_tokens + [correction]],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )[:, :remaining]
                input_ids = torch.cat([input_ids, tokens_to_add], dim=1)
                if use_suffix_hybrid or use_suffix_reserve:
                    self.draft.update_tokens(
                        (accepted_tokens + [correction])[:remaining]
                    )

            if hst_bank is not None:
                output_hidden = self._layer_hidden(output, -1)[0]
                context_hidden = hst_parent_hidden
                if hst_online_update:
                    for node_index in [0, *accepted_path]:
                        consumed_token = int(flat_tokens[node_index])
                        consumed_id = torch.tensor(
                            [consumed_token],
                            dtype=torch.long,
                            device=input_ids.device,
                        )
                        post_hidden = output_hidden[node_index]
                        hst_bank.append(
                            source_token_vector=embedding_layer(consumed_id)[0],
                            source_context_hidden=context_hidden,
                            source_post_hidden=post_hidden,
                        )
                        context_hidden = post_hidden
                hst_parent_hidden = output_hidden[query_index]
                hst_parent_candidate_logits = output.logits[
                    0, query_index
                ].index_select(0, hst_bank.candidate_token_ids)

            grounding_score, confidence = self._score_state(
                output,
                query_index,
                calibrator,
                grounding_layer,
                confidence_margin_scale,
                need_grounding=need_grounding,
                need_confidence=need_confidence,
            )
            controller.observe(accept_len, used_path_depth)
            acceptance_lengths.append(int(accept_len))
            if collect_policy_trace:
                record.update(
                    {
                        "accept_len": int(accept_len),
                        "accept_ratio": (
                            float(accept_len) / used_path_depth
                            if used_path_depth > 0
                            else 0.0
                        ),
                        "next_grounding_score": float(grounding_score),
                        "next_confidence": float(confidence),
                        "acceptance_ema_after": float(
                            controller.acceptance_ema
                        ),
                    }
                )
                if collect_candidate_trace:
                    record.update(
                        {
                            "accepted_path_node_indices": [
                                int(node_index)
                                for node_index in accepted_path
                            ],
                            "accepted_token_ids": [
                                int(token) for token in accepted_tokens
                            ],
                            "correction_token_id": int(correction),
                        }
                    )
                trace.append(record)

        if persistent_bank_transitions is not None:
            for token, (new_row, new_scores) in persistent_bank_pending.items():
                (
                    bank_row,
                    bank_scores,
                    bank_count,
                ) = self._merge_persistent_transition_row(
                    persistent_bank_transitions.get(token, ()),
                    persistent_bank_scores.get(token, ()),
                    persistent_bank_counts.get(token, 0),
                    new_row,
                    new_scores,
                    matrix_top_k,
                )
                persistent_bank_transitions[token] = bank_row
                persistent_bank_scores[token] = bank_scores
                persistent_bank_counts[token] = bank_count

        generated_ids = input_ids[0, prompt_length:]
        if collect_selective_reuse:
            generated_token_values = [
                int(token) for token in generated_ids.tolist()
            ]
            for record in trace:
                generated_offset = record.pop(
                    "selective_generated_offset", None
                )
                u_paths = record.pop(
                    "_selective_u_path_token_ids", []
                )
                gc_paths = record.pop(
                    "_selective_gc_path_token_ids", []
                )
                if generated_offset is None:
                    continue
                future = generated_token_values[int(generated_offset) :]
                target_token = future[0] if future else None
                max_candidate_accept = max(
                    int(record.get("remaining_tokens", len(future))) - 1,
                    0,
                )
                u_accept = min(
                    self._longest_matching_path(u_paths, future),
                    max_candidate_accept,
                )
                gc_accept = min(
                    self._longest_matching_path(gc_paths, future),
                    max_candidate_accept,
                )
                matched_budget = min(len(u_paths), len(gc_paths))
                u_matched_accept = min(
                    self._longest_matching_path(
                        u_paths[:matched_budget], future
                    ),
                    max_candidate_accept,
                )
                gc_matched_accept = min(
                    self._longest_matching_path(
                        gc_paths[:matched_budget], future
                    ),
                    max_candidate_accept,
                )
                fixed_accept = int(record.get("accept_len", 0))
                record.update(
                    {
                        "selective_output_position": int(generated_offset),
                        "selective_target_token_id": target_token,
                        "selective_u_top8_hit": (
                            target_token
                            in record.get(
                                "selective_u_root_candidate_token_ids", []
                            )
                            if target_token is not None
                            else None
                        ),
                        "selective_gc_top8_hit": (
                            target_token
                            in record.get(
                                "selective_gc_root_candidate_token_ids", []
                            )
                            if target_token is not None
                            else None
                        ),
                        "selective_v_top8_hit": (
                            target_token
                            in record.get(
                                "selective_v_root_candidate_token_ids", []
                            )
                            if target_token is not None
                            else None
                        ),
                        "selective_uv_top8_hit": (
                            target_token
                            in record.get(
                                "selective_uv_root_candidate_token_ids", []
                            )
                            if target_token is not None
                            else None
                        ),
                        "selective_uv_minus_u_top8_hit": (
                            int(
                                target_token
                                in record.get(
                                    "selective_uv_root_candidate_token_ids", []
                                )
                            )
                            - int(
                                target_token
                                in record.get(
                                    "selective_u_root_candidate_token_ids", []
                                )
                            )
                            if target_token is not None
                            else None
                        ),
                        "selective_u_shadow_accept_len": int(u_accept),
                        "selective_gc_shadow_accept_len": int(gc_accept),
                        "selective_gc_minus_u_accept_len": int(
                            gc_accept - u_accept
                        ),
                        "selective_matched_tree_node_budget": int(
                            matched_budget
                        ),
                        "selective_u_matched_accept_len": int(
                            u_matched_accept
                        ),
                        "selective_gc_matched_accept_len": int(
                            gc_matched_accept
                        ),
                        "selective_gc_minus_u_matched_accept_len": int(
                            gc_matched_accept - u_matched_accept
                        ),
                        "selective_exclusive_source_oracle_accept_len": int(
                            max(u_accept, gc_accept)
                        ),
                        "selective_fixed_accept_len": fixed_accept,
                        "selective_shadow_accept_cap": int(
                            max_candidate_accept
                        ),
                    }
                )
        keep_tokens = min(int(generated_ids.numel()), int(max_new_tokens))
        for token_index, token_id in enumerate(generated_ids[:keep_tokens].tolist()):
            if int(token_id) in stop_token_ids:
                keep_tokens = token_index + 1
                break
        input_ids = input_ids[:, : prompt_length + keep_tokens]
        outputs = (input_ids,)
        if log:
            outputs += (keep_tokens, idx)
        if return_acceptance_len:
            outputs += (acceptance_lengths,)
        if return_decode_time:
            outputs += (0.0,)
        if return_policy_trace:
            outputs += (trace,)
        return outputs[0] if len(outputs) == 1 else outputs
