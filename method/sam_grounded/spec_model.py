"""Grounding-aware, training-free SAM speculative decoding.

This implementation keeps SAM's exact target verification unchanged.  It wraps
the existing suffix-automaton draft lookup and caps the *next* proposal before
verification, using hidden states already returned by the target pass.  No
extra forward pass, auxiliary model, or learned parameter is introduced.
"""

from typing import Any, Dict, Optional

import torch

from method.sam.spec_model_sam import SpecModel as _BaseSpecModel
from method.vispec.spec_model_ours import _collect_stop_token_ids

from .controller import (
    GroundedDraftController,
    VisualGroundingCalibrator,
    confidence_from_logits,
)


class SpecModel(_BaseSpecModel):
    """SAM with target-only, fixed, and visually adaptive draft policies."""

    def _build_visual_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        arch = self.base_model.config.architectures[0]
        if arch in ("LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration"):
            image_token_id = getattr(
                self.base_model.config,
                "image_token_index",
                getattr(self.base_model.config, "image_token_id", None),
            )
            if image_token_id is None:
                return torch.zeros_like(input_ids[0], dtype=torch.bool)
            return input_ids[0].eq(image_token_id)

        if arch == "Qwen2_5_VLForConditionalGeneration":
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            image_token_id = getattr(self.base_model.config, "image_token_id", None)
            video_token_id = getattr(self.base_model.config, "video_token_id", None)
            if image_token_id is not None:
                mask |= input_ids[0].eq(image_token_id)
            if video_token_id is not None:
                mask |= input_ids[0].eq(video_token_id)
            return mask

        return torch.zeros_like(input_ids[0], dtype=torch.bool)

    @staticmethod
    def _layer_hidden(output: Any, layer: int) -> Optional[torch.Tensor]:
        if int(layer) == -1:
            last_hidden_state = getattr(output, "last_hidden_state", None)
            if last_hidden_state is not None:
                return last_hidden_state
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            return None
        try:
            return hidden_states[layer]
        except IndexError as exc:
            raise ValueError(
                f"grounding_layer={layer} is invalid for {len(hidden_states)} hidden-state tensors"
            ) from exc

    @torch.no_grad()
    def specgenerate(
        self,
        *args,
        draft_policy: str = "visual-soft",
        min_draft_tokens: int = 2,
        max_draft_tokens: int = 40,
        visual_threshold: float = 0.55,
        confidence_threshold: float = 0.75,
        visual_gamma: float = 1.0,
        visual_weight: float = 0.7,
        acceptance_weight: float = 0.3,
        acceptance_ema_decay: float = 0.8,
        grounding_layer: int = -1,
        confidence_margin_scale: float = 5.0,
        return_policy_trace: bool = False,
        return_acceptance_len: bool = False,
        return_decode_time: bool = False,
        disable_repeat_guard: bool = True,
        **kwargs,
    ):
        if args:
            input_ids = args[0]
        else:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids is required for multimodal grounding calibration")
        if input_ids.shape[0] != 1:
            raise ValueError("Grounding-aware SAM currently supports batch size 1")
        prompt_length = int(input_ids.shape[1])
        max_new_tokens = int(kwargs.get("max_new_tokens", 512))
        log_requested = bool(kwargs.get("log", False))

        controller = GroundedDraftController(
            policy=draft_policy,
            min_draft_tokens=min_draft_tokens,
            max_draft_tokens=max_draft_tokens,
            visual_threshold=visual_threshold,
            confidence_threshold=confidence_threshold,
            visual_gamma=visual_gamma,
            visual_weight=visual_weight,
            acceptance_weight=acceptance_weight,
            acceptance_ema_decay=acceptance_ema_decay,
        )
        visual_mask = self._build_visual_token_mask(input_ids).detach()
        original_lookup = self.draft.lookup
        trace = []
        state: Dict[str, Any] = {
            "seen_init": False,
            "calibrator": None,
            "grounding_score": 0.0,
            "confidence": 1.0,
            "pre_cache_length": None,
            "pending_output": None,
            "pending_record": None,
        }

        def score_output(output, query_index: int):
            hidden = self._layer_hidden(output, grounding_layer)
            if hidden is None:
                return 0.0, 1.0
            query_index = max(0, min(int(query_index), hidden.shape[1] - 1))
            calibrator = state["calibrator"]
            grounding_score = (
                calibrator.score(hidden[0, query_index])
                if calibrator is not None
                else 0.0
            )
            logits = getattr(output, "logits", None)
            confidence = (
                confidence_from_logits(logits[0, query_index], confidence_margin_scale)
                if logits is not None
                else 1.0
            )
            return grounding_score, confidence

        def finalize_pending(accepted_override: Optional[int] = None):
            output = state["pending_output"]
            record = state["pending_record"]
            if output is None or record is None:
                return

            accepted = accepted_override
            if accepted is None:
                before = state["pre_cache_length"]
                current = None
                if hasattr(self, "current_length_data"):
                    current = int(self.current_length_data[0].item())
                if before is not None and current is not None:
                    accepted = max(current - before - 1, 0)
                else:
                    accepted = 0
            accepted = min(int(accepted), int(record["used_draft_len"]))

            next_score, next_confidence = score_output(output, accepted)
            controller.observe(accepted, int(record["used_draft_len"]))
            record.update(
                {
                    "accept_len": accepted,
                    "accept_ratio": (
                        float(accepted) / record["used_draft_len"]
                        if record["used_draft_len"] > 0
                        else 0.0
                    ),
                    "next_grounding_score": next_score,
                    "next_confidence": next_confidence,
                    "acceptance_ema_after": float(controller.acceptance_ema),
                }
            )
            state["grounding_score"] = next_score
            state["confidence"] = next_confidence
            state["pending_output"] = None
            state["pending_record"] = None

        def pre_forward_hook(_module, _hook_args):
            if state["seen_init"] and hasattr(self, "current_length_data"):
                state["pre_cache_length"] = int(self.current_length_data[0].item())

        def post_forward_hook(_module, _hook_args, output):
            hidden = self._layer_hidden(output, grounding_layer)
            if not state["seen_init"]:
                state["seen_init"] = True
                if hidden is not None and hidden.shape[1] == visual_mask.numel():
                    state["calibrator"] = VisualGroundingCalibrator.from_prompt(
                        hidden[0], visual_mask
                    )
                if hidden is not None:
                    state["grounding_score"], state["confidence"] = score_output(
                        output, hidden.shape[1] - 1
                    )
                return
            state["pending_output"] = output

        def adaptive_lookup(start_token):
            # By the next lookup, the base implementation has committed the
            # accepted prefix and rewound its reusable KV cache length.
            finalize_pending()
            candidate_type, draft_tokens, buffers_kwargs = original_lookup(start_token)
            raw_len = int(draft_tokens.numel())
            decision = controller.decide(
                state["grounding_score"], state["confidence"]
            )
            generated = 0
            if hasattr(self, "current_length_data"):
                generated = max(
                    int(self.current_length_data[0].item()) + 1 - prompt_length,
                    0,
                )
            remaining_tokens = max(max_new_tokens - generated, 0)
            # Every verification iteration appends one target token after the
            # accepted draft prefix, so at most remaining-1 draft tokens may be
            # proposed.  This makes max_new_tokens invariant to draft grouping.
            remaining_draft_capacity = max(remaining_tokens - 1, 0)
            used_len = min(raw_len, decision.budget, remaining_draft_capacity)
            if used_len < raw_len:
                draft_tokens = draft_tokens[:used_len]

            record = {
                "iteration": len(trace),
                **decision.to_dict(),
                "raw_draft_len": raw_len,
                "used_draft_len": used_len,
                "remaining_tokens": remaining_tokens,
            }
            if not trace and state["calibrator"] is not None:
                record["grounding_calibration"] = state["calibrator"].diagnostics()
            trace.append(record)
            state["pending_record"] = record
            return candidate_type, draft_tokens, buffers_kwargs

        pre_handle = self.base_model.register_forward_pre_hook(pre_forward_hook)
        post_handle = self.base_model.register_forward_hook(post_forward_hook)
        self.draft.lookup = adaptive_lookup
        original_repeat_guard = self._find_consecutive_ngram_repeat
        if disable_repeat_guard:
            self._find_consecutive_ngram_repeat = lambda *unused_args, **unused_kwargs: 0
        base_result = None
        try:
            # Acceptance lengths are required internally to finalize the final
            # trace row even when the caller only requests policy diagnostics.
            base_result = super().specgenerate(
                *args,
                return_acceptance_len=True,
                return_decode_time=return_decode_time,
                **kwargs,
            )
        finally:
            self.draft.lookup = original_lookup
            self._find_consecutive_ngram_repeat = original_repeat_guard
            pre_handle.remove()
            post_handle.remove()

        if not isinstance(base_result, tuple):
            raise RuntimeError("SAM did not return the internally requested acceptance lengths")
        acceptance_index = -2 if return_decode_time else -1
        acceptance_lengths = base_result[acceptance_index]
        if state["pending_record"] is not None:
            final_accept = acceptance_lengths[-1] if acceptance_lengths else 0
            finalize_pending(accepted_override=int(final_accept))

        # Canonicalize the public sequence at the first stop token and at the
        # requested generation limit.  This also protects callers from legacy
        # SAM behavior that could append a verified chunk beyond the limit.
        result_parts = list(base_result)
        output_ids = result_parts[0]
        generated_ids = output_ids[0, prompt_length:]
        keep_tokens = min(int(generated_ids.numel()), max_new_tokens)
        stop_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        for index, token_id in enumerate(generated_ids[:keep_tokens].tolist()):
            if int(token_id) in stop_ids:
                keep_tokens = index + 1
                break
        if int(generated_ids.numel()) != keep_tokens:
            result_parts[0] = output_ids[:, : prompt_length + keep_tokens]
            if log_requested and len(result_parts) >= 2:
                result_parts[1] = keep_tokens
            base_result = tuple(result_parts)

        # The base result always ends in the internally requested acceptance
        # list.  Preserve the public SAM tuple layout unless the caller asks for
        # that field and/or our policy trace.
        if return_decode_time:
            public = list(base_result[:acceptance_index])
        else:
            public = list(base_result[:-1])
        if return_acceptance_len:
            public.append(acceptance_lengths)
        if return_decode_time:
            public.append(base_result[-1])
        if return_policy_trace:
            public.append(trace)
        return public[0] if len(public) == 1 else tuple(public)
