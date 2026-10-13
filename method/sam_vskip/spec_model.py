import time

import torch

from method.sam.spec_model_sam import SpecModel as _BaseSpecModel
from method.vispec.spec_model_ours import (
    _collect_stop_token_ids,
    _has_stop_token,
    _normalize_token_ids,
)


class SpecModel(_BaseSpecModel):
    """SAM variant: use current-step attention to skip speculative accepts on visual-heavy steps."""

    def _build_visual_key_mask(self, input_ids):
        if input_ids is None or input_ids.numel() == 0:
            return None

        arch = self.base_model.config.architectures[0]
        if arch in ("LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration"):
            image_token_id = getattr(
                self.base_model.config,
                "image_token_index",
                getattr(self.base_model.config, "image_token_id", None),
            )
            if image_token_id is None:
                return None
            return (input_ids[0] == image_token_id).to(torch.bool)

        if arch == "Qwen2_5_VLForConditionalGeneration":
            image_token_id = getattr(self.base_model.config, "image_token_id", None)
            video_token_id = getattr(self.base_model.config, "video_token_id", None)
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            if image_token_id is not None:
                mask = mask | (input_ids[0] == image_token_id)
            if video_token_id is not None:
                mask = mask | (input_ids[0] == video_token_id)
            return mask

        return None

    def _visual_attention_ratio(self, attentions, visual_key_mask, query_index=-1):
        if attentions is None or visual_key_mask is None:
            return 0.0
        if isinstance(attentions, (tuple, list)):
            if len(attentions) == 0:
                return 0.0
            attn = attentions[-1]
        else:
            attn = attentions
        if attn is None or attn.numel() == 0:
            return 0.0

        q_len = attn.shape[2]
        if query_index < 0:
            query_index = q_len + query_index
        query_index = max(0, min(query_index, q_len - 1))
        query_attn = attn[:, :, query_index : query_index + 1, :]

        key_len = query_attn.shape[-1]
        if visual_key_mask.shape[-1] < key_len:
            pad = torch.zeros(
                key_len - visual_key_mask.shape[-1],
                dtype=visual_key_mask.dtype,
                device=visual_key_mask.device,
            )
            key_mask = torch.cat([visual_key_mask, pad], dim=0)
        else:
            key_mask = visual_key_mask[:key_len]
        key_mask = key_mask.to(query_attn.device, query_attn.dtype).view(1, 1, 1, -1)

        ratio = (query_attn * key_mask).sum(dim=-1).mean()
        return float(ratio.item())

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
        skip_when_visual=True,
        visual_attn_threshold=0.35,
        visual_probe_only=False,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        if is_llama3:
            stop_token_ids.update(
                _normalize_token_ids(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            )
        max_length = max_length - self.total_tokens - 10
        repeat_times = 3
        min_repeat_ngram = 2
        max_repeat_ngram = 32
        visual_attn_threshold = float(visual_attn_threshold)

        if temperature > 1e-5:
            from method.vispec.utils import prepare_logits_processor

            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"

        input_ids = input_ids.clone()

        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            from method.vispec.kv_cache import initialize_past_key_values

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
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        arch = self.base_model.config.architectures[0]
        # LLaVA's native forward owns feature packing and image-token merging.
        # Current Transformers already packs in get_image_features and keeps
        # image_newline on the inner model. Let prefill perform this once, using
        # input_ids + pixel_values, exactly as in the AR path below the adapter.
        if arch == "Qwen2_5_VLForConditionalGeneration":
            image_token_id = self.base_model.config.image_token_id
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")

            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
                    n_image_tokens = (input_ids == image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        input_len = input_ids.shape[1]
        visual_key_mask = self._build_visual_key_mask(input_ids)
        if visual_key_mask is None:
            visual_key_mask = torch.zeros(input_len, dtype=torch.bool, device=input_ids.device)
        else:
            visual_key_mask = visual_key_mask.to(input_ids.device)
        use_visual_gate = skip_when_visual and bool(visual_key_mask.any().item())

        if return_acceptance_len:
            acceptance_len = []

        init_out = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=False,
            **kwargs,
        )
        init_logits = init_out.logits[:, -1, :]

        inputs_embeds = None
        kwargs = {}

        if logits_processor is not None:
            proc = logits_processor(None, init_logits)
            probs = torch.softmax(proc, dim=-1)
            init_token = torch.multinomial(probs, 1).squeeze(-1)
        else:
            init_token = torch.argmax(init_logits, dim=-1)

        input_ids = torch.cat((input_ids, init_token.unsqueeze(-1).to(input_ids.device)), dim=1)

        prompt_ids = input_ids[0].detach().cpu().tolist()
        self.draft.reset()
        self.draft.update(torch.tensor(prompt_ids, dtype=torch.long))

        if hasattr(self, "current_length_data"):
            current_length_data.fill_(input_ids.shape[1] - 1)

        if return_decode_time:
            torch.cuda.synchronize()
            start_time = time.time()

        idx = 0
        for idx in range(max_length):
            curr_len = input_ids.shape[1]
            if curr_len - input_len >= max_new_tokens:
                break

            if _has_stop_token(input_ids[0, input_len:], stop_token_ids):
                break

            ids = input_ids[0].detach().cpu().tolist()
            _, draft_tokens, _ = self.draft.lookup(ids[-1])

            last_tok = input_ids[:, -1:]
            if draft_tokens.numel() == 0:
                out = self.base_model(
                    input_ids=last_tok,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=False,
                )
                next_logits = out.logits[:, -1, :]
                if logits_processor is not None:
                    proc = logits_processor(None, next_logits)
                    probs = torch.softmax(proc, dim=-1)
                    next_id = torch.multinomial(probs, 1).squeeze(1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1)
                input_ids = torch.cat((input_ids, next_id.unsqueeze(0)), dim=-1)
                visual_key_mask = torch.cat(
                    [visual_key_mask, torch.zeros(1, dtype=torch.bool, device=visual_key_mask.device)], dim=0
                )
                if hasattr(self, "current_length_data"):
                    current_length_data.fill_(input_ids.shape[1] - 1)
                if return_acceptance_len:
                    acceptance_len.append(0)
                self.draft.update(next_id)
                repeat_n = self._find_consecutive_ngram_repeat(
                    input_ids[0, input_len:],
                    repeat_times=repeat_times,
                    min_ngram=min_repeat_ngram,
                    max_ngram=max_repeat_ngram,
                )
                if repeat_n > 0:
                    input_ids = input_ids[:, :-repeat_n]
                    if hasattr(self, "current_length_data"):
                        current_length_data.fill_(input_ids.shape[1] - 1)
                    break
                continue

            draft_tokens = draft_tokens.to(dtype=input_ids[0].dtype, device=input_ids[0].device).view(1, -1)
            chunk = torch.cat((last_tok, draft_tokens), dim=1)

            out = self.base_model(
                input_ids=chunk,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=use_visual_gate,
            )
            logits = out.logits

            D = draft_tokens.shape[1]
            draft_pred_logits = logits[:, 0 : D + 1, :]
            draft_pred_ids = torch.argmax(draft_pred_logits, dim=-1)

            same = draft_pred_ids[:, :-1] == draft_tokens
            mismatch = (~same).squeeze(0).nonzero(as_tuple=True)[0]
            accept_len = int(mismatch[0].item()) if mismatch.numel() > 0 else D

            if use_visual_gate:
                visual_ratio = self._visual_attention_ratio(
                    out.attentions,
                    visual_key_mask,
                    query_index=0,
                )
                if visual_ratio >= visual_attn_threshold and (not visual_probe_only):
                    accept_len = 0

            if return_acceptance_len:
                acceptance_len.append(accept_len)

            if accept_len > 0:
                input_ids = torch.cat([input_ids, draft_tokens[:, :accept_len]], dim=1)
                self.draft.update(draft_tokens[0, :accept_len])

            next_id = draft_pred_ids[:, accept_len]
            if next_id is not None:
                self.draft.update(next_id)
            input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)

            appended = accept_len + 1
            visual_key_mask = torch.cat(
                [visual_key_mask, torch.zeros(appended, dtype=torch.bool, device=visual_key_mask.device)],
                dim=0,
            )

            if hasattr(self, "current_length_data"):
                current_length_data.fill_(input_ids.shape[1] - 1)

            repeat_n = self._find_consecutive_ngram_repeat(
                input_ids[0, input_len:],
                repeat_times=repeat_times,
                min_ngram=min_repeat_ngram,
                max_ngram=max_repeat_ngram,
            )
            if repeat_n > 0:
                input_ids = input_ids[:, :-repeat_n]
                if hasattr(self, "current_length_data"):
                    current_length_data.fill_(input_ids.shape[1] - 1)
                break

            if _has_stop_token(input_ids[0, input_len:], stop_token_ids):
                break

        new_token = input_ids.shape[1] - input_len
        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            torch.cuda.synchronize()
            outputs += (time.time() - start_time,)
        return outputs[0] if len(outputs) == 1 else outputs
