"""Training-free token-recycling drafts with multimodal budget control."""

from typing import Optional

import torch

from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import (
    _collect_stop_token_ids,
    _has_stop_token,
)

from .controller import (
    GroundedDraftController,
    VisualGroundingCalibrator,
    confidence_from_logits,
)
from .spec_model import SpecModel as _GroundedSamSpecModel


class RecyclingSpecModel(_GroundedSamSpecModel):
    """Use target-generated token transitions as the draft model.

    Unlike the repository's diagnostic Recycling implementation, this path does
    not alter target tokens with anti-loop heuristics.  Proposal-cycle checks
    only shorten a draft and therefore preserve target verification semantics.
    """

    def _score_state(
        self,
        output,
        query_index: int,
        calibrator: Optional[VisualGroundingCalibrator],
        grounding_layer: int,
        confidence_margin_scale: float,
    ):
        hidden = self._layer_hidden(output, grounding_layer)
        query_index = max(0, min(int(query_index), output.logits.shape[1] - 1))
        score = (
            calibrator.score(hidden[0, query_index])
            if calibrator is not None and hidden is not None
            else 0.0
        )
        confidence = confidence_from_logits(
            output.logits[0, query_index], confidence_margin_scale
        )
        return score, confidence

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
        return_policy_trace=False,
        disable_repeat_guard=True,
        **kwargs,
    ):
        del top_p, top_k, disable_repeat_guard
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids.shape[0] != 1:
            raise ValueError("Grounded Recycling currently supports batch size 1")
        if temperature > 1e-5:
            raise NotImplementedError("Grounded Recycling pilot currently supports greedy decoding")

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
        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        prompt_length = int(input_ids.shape[1])
        max_length = min(int(max_length), prompt_length + int(max_new_tokens))
        matrix_top_k = max(int(matrix_top_k), 1)

        if hasattr(self, "recycling_past_key_values"):
            past_key_values = self.recycling_past_key_values
            current_length_data = self.recycling_current_length_data
            current_length_data.zero_()
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
            except Exception:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
            self.recycling_past_key_values = past_key_values
            self.recycling_past_key_values_data = past_key_values_data
            self.recycling_current_length_data = current_length_data

        arch = self.base_model.config.architectures[0]
        if arch != "Qwen2_5_VLForConditionalGeneration":
            raise NotImplementedError(
                "The grounded Recycling pilot currently targets Qwen2.5-VL"
            )

        image_token_id = self.base_model.config.image_token_id
        visual_mask = self._build_visual_token_mask(input_ids).detach()
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if inputs_embeds is None:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.base_model.visual.dtype)
                image_embeds = self.base_model.visual(
                    pixel_values, grid_thw=image_grid_thw
                )
                image_mask = input_ids.eq(image_token_id)
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise ValueError("Image features and image tokens do not match")
                expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    expanded_mask,
                    image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
                )

        init_output = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = self._layer_hidden(init_output, grounding_layer)
        calibrator = (
            VisualGroundingCalibrator.from_prompt(hidden[0], visual_mask)
            if hidden is not None
            else None
        )
        grounding_score, confidence = self._score_state(
            init_output,
            query_index=init_output.logits.shape[1] - 1,
            calibrator=calibrator,
            grounding_layer=grounding_layer,
            confidence_margin_scale=confidence_margin_scale,
        )

        vocab_size = int(self.base_model.config.vocab_size)
        transitions = torch.zeros(
            (vocab_size, matrix_top_k), dtype=torch.long, device=input_ids.device
        )
        transition_valid = torch.zeros(
            vocab_size, dtype=torch.bool, device=input_ids.device
        )
        prompt_ids = input_ids[0]
        transitions[prompt_ids] = init_output.logits[0].topk(
            matrix_top_k, dim=-1
        ).indices
        transition_valid[prompt_ids] = True

        init_token = torch.argmax(init_output.logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, init_token[:, None]], dim=1)
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
            last_token = int(input_ids[0, -1].item())
            raw_draft = []
            current_token = last_token
            seen_transitions = set()
            for _ in range(max_draft_tokens):
                if not bool(transition_valid[current_token].item()):
                    break
                next_token = int(transitions[current_token, 0].item())
                edge = (current_token, next_token)
                if next_token == image_token_id or edge in seen_transitions:
                    break
                seen_transitions.add(edge)
                raw_draft.append(next_token)
                current_token = next_token

            remaining = int(max_new_tokens - generated)
            used_len = min(len(raw_draft), decision.budget, max(remaining - 1, 0))
            draft_tokens = raw_draft[:used_len]
            record = {
                "iteration": len(trace),
                **decision.to_dict(),
                "raw_draft_len": len(raw_draft),
                "used_draft_len": used_len,
                "remaining_tokens": remaining,
            }
            if not trace and calibrator is not None:
                record["grounding_calibration"] = calibrator.diagnostics()

            last_tok = input_ids[:, -1:]
            if used_len == 0:
                output = self.base_model(
                    input_ids=last_tok,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    output_hidden_states=True,
                )
                next_id = torch.argmax(output.logits[:, -1, :], dim=-1)
                transitions[last_token] = output.logits[0, -1].topk(
                    matrix_top_k
                ).indices
                transition_valid[last_token] = True
                accept_len = 0
                input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)
                current_length_data.fill_(input_ids.shape[1] - 1)
                grounding_score, confidence = self._score_state(
                    output,
                    query_index=0,
                    calibrator=calibrator,
                    grounding_layer=grounding_layer,
                    confidence_margin_scale=confidence_margin_scale,
                )
            else:
                draft = torch.tensor(
                    [draft_tokens], dtype=input_ids.dtype, device=input_ids.device
                )
                chunk = torch.cat([last_tok, draft], dim=1)
                output = self.base_model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    output_hidden_states=True,
                )
                chunk_topk = output.logits[0].topk(matrix_top_k, dim=-1).indices
                transitions[chunk[0]] = chunk_topk
                transition_valid[chunk[0]] = True

                predictions = torch.argmax(output.logits, dim=-1)
                same = predictions[:, :used_len].eq(draft)
                mismatch = (~same).squeeze(0).nonzero(as_tuple=True)[0]
                accept_len = (
                    int(mismatch[0].item()) if mismatch.numel() > 0 else used_len
                )
                tokens_to_add = torch.cat(
                    [draft[:, :accept_len], predictions[:, accept_len : accept_len + 1]],
                    dim=1,
                )
                tokens_to_add = tokens_to_add[:, :remaining]
                input_ids = torch.cat([input_ids, tokens_to_add], dim=1)
                current_length_data.fill_(input_ids.shape[1] - 1)
                grounding_score, confidence = self._score_state(
                    output,
                    query_index=accept_len,
                    calibrator=calibrator,
                    grounding_layer=grounding_layer,
                    confidence_margin_scale=confidence_margin_scale,
                )

            controller.observe(accept_len, used_len)
            record.update(
                {
                    "accept_len": int(accept_len),
                    "accept_ratio": (
                        float(accept_len) / used_len if used_len > 0 else 0.0
                    ),
                    "next_grounding_score": float(grounding_score),
                    "next_confidence": float(confidence),
                    "acceptance_ema_after": float(controller.acceptance_ema),
                }
            )
            trace.append(record)
            acceptance_lengths.append(int(accept_len))

        # Stop tokens are part of the generated sequence; discard only any
        # verified suffix after the first one.
        generated_ids = input_ids[0, prompt_length:]
        keep_tokens = min(int(generated_ids.numel()), int(max_new_tokens))
        for token_index, token_id in enumerate(generated_ids[:keep_tokens].tolist()):
            if int(token_id) in stop_token_ids:
                keep_tokens = token_index + 1
                break
        input_ids = input_ids[:, : prompt_length + keep_tokens]
        new_token = keep_tokens

        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_lengths,)
        if return_decode_time:
            outputs += (0.0,)
        if return_policy_trace:
            outputs += (trace,)
        return outputs[0] if len(outputs) == 1 else outputs
