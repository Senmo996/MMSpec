"""Grounded broad/shallow versus narrow/deep token-recycling trees."""

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
from .spec_model import SpecModel as _GroundedSamSpecModel
from .visual_lexical_inventory import rank_scores


class TreeRecyclingSpecModel(_GroundedSamSpecModel):
    """Full-tree token recycling with multimodal width/depth allocation."""

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
        if policy == "wide-plus2":
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
    ):
        # Node indices in the returned flat sequence start at one; zero is the
        # already-generated root token that has not yet entered the KV cache.
        nodes = []
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
                for candidate in candidates:
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

        tree_len = len(nodes) + 1
        # These trees contain at most a few dozen nodes.  Build their topology
        # in host memory and submit two compact transfers instead of launching
        # a separate GPU scalar-write kernel for every ancestor relation.
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
        )
        positions = torch.tensor(
            position_values, dtype=torch.long, device=transitions.device
        )

        flat_tokens = [root_token] + [node["token"] for node in nodes]
        return flat_tokens, mask[None, None], positions, paths

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
    def _best_verified_path(
        output_logits: torch.Tensor,
        flat_tokens: List[int],
        paths: List[List[int]],
        remaining_tokens: int,
    ):
        # Transfer the small packed-tree prediction vector once.  Calling
        # ``Tensor.item`` for every path repeatedly synchronized the same root
        # and parent logits, which was measurable at decoding granularity.
        predictions = torch.argmax(output_logits[0], dim=-1).tolist()
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
            "narrow",
            "spine",
            "short",
            "hybrid",
        }
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
        max_length = min(int(max_length), prompt_length + int(max_new_tokens))
        tree_fixed_width = max(int(tree_fixed_width), 1)
        tree_fixed_depth = max(int(tree_fixed_depth), 1)
        tree_broad_width = max(int(tree_broad_width), 1)
        tree_shallow_depth = max(int(tree_shallow_depth), 1)
        tree_node_budget = max(int(tree_node_budget), 1)
        uses_augmented_width = draft_policy in (
            "wide-plus2",
            "visual-wide-plus2",
            "visual-wide-plus2-reverse",
            "visual-rootwide-plus2",
            "visual-rootwide-plus2-hst-backoff",
            "visual-wide-plus2-vli-backoff",
            "visual-wide-plus2-hst-backoff",
            "visual-wide-plus2-hst-backoff-gated",
            "grounded-residual",
            "grounded-residual-reverse",
        )
        matrix_top_k = max(
            int(matrix_top_k),
            tree_fixed_width,
            tree_broad_width + (2 if uses_augmented_width else 0),
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

        if self.base_model.config.architectures[0] != "Qwen2_5_VLForConditionalGeneration":
            raise NotImplementedError("Tree Recycling pilot currently targets Qwen2.5-VL")
        image_token_id = self.base_model.config.image_token_id
        visual_mask = self._build_visual_token_mask(input_ids).detach()
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        cover_layout = None
        if cover_enabled:
            cover_layout = build_visual_probe_layout(
                visual_mask,
                image_grid_thw,
                int(self.base_model.config.vision_config.spatial_merge_size),
                cover_num_probes,
            )
        if inputs_embeds is None:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                image_embeds = self.base_model.visual(
                    pixel_values.type(self.base_model.visual.dtype),
                    grid_thw=image_grid_thw,
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
        excluded_token_ids = set(int(token) for token in self.tokenizer.all_special_ids)
        for attribute in ("image_token_id", "video_token_id"):
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
        vocab_size = int(self.base_model.config.vocab_size)
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
        transition_counts = torch.zeros(
            (num_transition_bins, vocab_size),
            dtype=torch.int32,
            device=input_ids.device,
        )

        def store_transitions(output, token_ids):
            query_count = int(token_ids.numel())
            topk_ids = output.logits[0, :query_count].topk(
                matrix_top_k, dim=-1
            ).indices
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
            transition_counts.index_put_(
                (bins, token_ids),
                torch.ones_like(token_ids, dtype=torch.int32),
                accumulate=True,
            )

        prompt_ids = input_ids[0]
        store_transitions(init_output, prompt_ids)

        init_token = torch.argmax(init_output.logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, init_token[:, None]], dim=1)
        if use_suffix_hybrid:
            self.draft.reset()
            self.draft.update(input_ids[0].detach().cpu())
        current_length_data.fill_(input_ids.shape[1] - 1)
        kwargs = {}
        acceptance_lengths = []
        trace = []
        idx = -1

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
            root_token = int(input_ids[0, -1].item())
            root_row_valid_before_backoff = bool(
                transition_valid[transition_bin, root_token].item()
            )
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
            if use_suffix_hybrid:
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
            flat_tokens, tree_mask, tree_positions, paths = self._build_tree(
                root_token,
                transitions,
                transition_valid,
                width,
                depth,
                tree_node_budget,
                image_token_id,
                branch_width=branch_width,
                transition_bin=transition_bin,
                fallback_transition_bin=fallback_transition_bin,
                transition_counts=transition_counts,
                preserve_fallback_limit=(
                    tree_broad_width if use_grounded_residual else 0
                ),
            )
            num_nodes = len(flat_tokens) - 1
            cover_probe_active = bool(
                cover_enabled
                and grounding_score >= cover_probe_visual_threshold
                and num_nodes > 0
                and remaining > 1
            )
            max_path_depth = max((len(path) for path in paths), default=0)
            used_path_depth = min(max_path_depth, max(remaining - 1, 0))
            record = {
                "iteration": len(trace),
                **decision.to_dict(),
                "requested_draft_policy": requested_draft_policy,
                "cover_enabled": bool(cover_enabled),
                "cover_probe_active": cover_probe_active,
                "cover_probe_visual_threshold": cover_probe_visual_threshold,
                "cover_min_jsd": cover_min_jsd,
                "tree_width": int(width),
                "tree_branch_width": int(branch_width),
                "tree_depth": int(depth),
                "raw_draft_len": int(num_nodes),
                "used_draft_len": int(used_path_depth),
                "verified_candidate_tree_nodes": int(
                    num_nodes if remaining > 1 else 0
                ),
                "verified_probe_nodes": int(
                    cover_num_probes
                    if cover_probe_active
                    else 0
                ),
                "verified_tree_nodes": int(
                    (
                        num_nodes
                        + (cover_num_probes if cover_probe_active else 0)
                    )
                    if num_nodes > 0 and remaining > 1
                    else 0
                ),
                "transition_bin": int(transition_bin),
                "num_transition_bins": int(num_transition_bins),
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
            }
            if not trace and calibrator is not None:
                record["grounding_calibration"] = calibrator.diagnostics()

            if num_nodes == 0 or remaining <= 1:
                output = self.base_model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    output_hidden_states=need_all_hidden_states,
                    output_last_hidden_state=need_hidden_states,
                )
                store_transitions(output, input_ids[0, -1:])
                next_id = torch.argmax(output.logits[:, -1, :], dim=-1)
                input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)
                if verification_trace_diagnostics:
                    top_values = torch.topk(
                        output.logits[0, -1].float(), k=2
                    ).values
                    record.update(
                        {
                            "verification_committed_token_ids": [
                                int(next_id.item())
                            ],
                            "verification_token_margins": [
                                float((top_values[0] - top_values[1]).item())
                            ],
                            "verification_min_margin": float(
                                (top_values[0] - top_values[1]).item()
                            ),
                            "verification_accepted_path": [],
                        }
                    )
                if use_suffix_hybrid:
                    self.draft.update(next_id)
                current_length_data.fill_(input_ids.shape[1] - 1)
                accepted_path = []
                accept_len = 0
                query_index = 0
            else:
                kv_base_len = int(current_length_data[0].item())
                tree_input = torch.tensor(
                    [flat_tokens], dtype=input_ids.dtype, device=input_ids.device
                )
                verifier_input = tree_input
                verifier_positions = tree_positions
                cover_attention_mask = None
                if cover_probe_active:
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
                    cover_attention_mask = build_tree_counterfactual_attention_mask(
                        kv_base_len,
                        tree_mask,
                        cover_layout,
                        self.base_model.dtype,
                        input_ids.device,
                    )
                position_ids = verifier_positions + kv_base_len
                position_ids = position_ids.unsqueeze(0) + self.base_model.rope_deltas
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                if cover_probe_active:
                    output = self.base_model(
                        input_ids=verifier_input,
                        attention_mask=cover_attention_mask,
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
                    self.base_model.model.tree_mask = tree_mask
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
                        self.base_model.model.tree_mask = None

                store_transitions(output, tree_input[0])
                if cover_probe_active:
                    actual_tree_length = int(tree_input.shape[1])
                    cover_view_logits = torch.cat(
                        [
                            output.logits[0, :1],
                            output.logits[0, actual_tree_length:],
                        ],
                        dim=0,
                    )
                    previous_row = transitions[
                        transition_bin, root_token
                    ].clone()
                    recycled = torch.tensor(
                        cover_candidates(
                            cover_view_logits,
                            matrix_top_k,
                            anchor_fraction=cover_anchor_fraction,
                        ),
                        dtype=transitions.dtype,
                        device=transitions.device,
                    )
                    view_jsd = multiview_jsd(cover_view_logits)
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
                                cover_view_logits, matrix_top_k
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
                )
                accept_len = len(accepted_path)
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
                    top_values = torch.topk(
                        committed_logits, k=2, dim=-1
                    ).values
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
                            "verification_min_margin": min(
                                (float(value) for value in margins),
                                default=None,
                            ),
                            "verification_accepted_path": list(accepted_path),
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
                if use_suffix_hybrid:
                    self.draft.update(tokens_to_add[0])

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
                    "acceptance_ema_after": float(controller.acceptance_ema),
                }
            )
            trace.append(record)

        generated_ids = input_ids[0, prompt_length:]
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
