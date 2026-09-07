import copy
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.utils import *
from .samd.sam_draft import DraftModel


def _get_model_class(arch_type):
    if arch_type == "Qwen2_5_VLForConditionalGeneration":
        from method.vispec.modeling_qwen2_5_vl_kv import (
            Qwen2_5_VLForConditionalGeneration,
        )
        return Qwen2_5_VLForConditionalGeneration
    if arch_type == "LlavaForConditionalGeneration":
        from method.llava_adapter import CustomLlavaForConditionalGeneration
        return CustomLlavaForConditionalGeneration
    if arch_type == "LlavaNextForConditionalGeneration":
        from method.llava_adapter import CustomLlavaNextForConditionalGeneration
        return CustomLlavaNextForConditionalGeneration
    raise NotImplementedError(f"Model type {arch_type} is not supported.")

class SpecModel(nn.Module):

    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        total_token=30,
    ):
        """Initialize SAM speculative decoding model.
        
        SAM uses n-gram based draft from history, no separate spec model needed.
        Similar to lookahead decoding.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.draft = DraftModel(device=base_model.device)
        self.total_tokens = total_token  # Store total_token directly

        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def _find_consecutive_ngram_repeat(
        self,
        token_ids: torch.Tensor,
        repeat_times: int = 3,
        min_ngram: int = 2,
        max_ngram: int = 32,
    ) -> int:
        """Return repeated n-gram length if tail has consecutive repeats, else 0.

        Example (repeat_times=3):
            [..., A, B, A, B, A, B]  -> True for n=2
            [..., X, X, X]           -> True for n=1
        """
        if token_ids is None or token_ids.numel() == 0:
            return 0
        if token_ids.dim() != 1:
            token_ids = token_ids.view(-1)
        if repeat_times < 2:
            return 0

        length = int(token_ids.numel())
        min_n = max(int(min_ngram), 1)
        max_n = min(int(max_ngram), length // repeat_times)
        if max_n < min_n:
            return 0

        for n in range(min_n, max_n + 1):
            span = n * repeat_times
            tail = token_ids[-span:]
            pattern = tail[:n]
            matched = True
            for i in range(1, repeat_times):
                if not torch.equal(tail[i * n : (i + 1) * n], pattern):
                    matched = False
                    break
            if matched:
                return n
        return 0

    @classmethod
    def from_pretrained(
        cls,
        Type="LLaMA",
        base_model_path=None,
        spec_model_path=None,  # Not used, kept for API compatibility
        total_token=30,
        depth=3,  # Not used, kept for API compatibility
        top_k=8,  # Not used, kept for API compatibility  
        threshold=1.0,  # Not used, kept for API compatibility
        **kwargs,
    ):
        """Load SAM model from pretrained base model.
        
        SAM doesn't require a separate spec model - it uses n-gram based
        draft from generation history, similar to lookahead decoding.
        
        Args:
            base_model_path: Path to the base model
            spec_model_path: Not used (kept for API compatibility)
            total_token: Maximum number of draft tokens
            depth, top_k, threshold: Not used (kept for API compatibility)
        """
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        model_class = _get_model_class(Type)
        base_model = model_class.from_pretrained(base_model_path, **kwargs)

        # SAM doesn't need spec model - just create instance with base model
        model = cls(
            base_model,
            base_model_path,
            total_token,
        )

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        inputs_embeds=None,
        output_real_hidden=False,
        **kwargs,
    ):
        if (
            inputs_embeds is not None
            and self.base_model.config.architectures[0]
            == "LlavaNextForConditionalGeneration"
        ):
            input_ids = None
            kwargs = {}

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
                **kwargs,
            )
            if output_orig:
                orig = outputs.logits
            hidden_states = outputs.hidden_states[-1]

        if output_real_hidden:
            return None, orig, hidden_states, outputs.hidden_states
        if output_orig:
            return None, orig, hidden_states
        else:
            return None, hidden_states

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
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.total_tokens - 10
        repeat_times = 3
        min_repeat_ngram = 2
        max_repeat_ngram = 32

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None
        
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
            except:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        if (
            self.base_model.config.architectures[0]
            == "Qwen2_5_VLForConditionalGeneration"
        ):
            image_token_id = self.base_model.config.image_token_id
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            # video_grid_thw = kwargs.get("video_grid_thw")

            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(
                        pixel_values, grid_thw=image_grid_thw
                    )
                    n_image_tokens = (
                        (input_ids == image_token_id)
                        .sum()
                        .item()
                    )
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask, image_embeds
                    )

                    # special_image_mask = mask

        # Prompt lookup decoding (PLD)

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # Initialize acceptance length tracking if needed
        if return_acceptance_len:
            acceptance_len = []

        init_out = self.forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_orig=True,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        init_logits = init_out[1][:, -1, :]

        inputs_embeds = None
        kwargs = {}

        if logits_processor is not None:
            proc = logits_processor(None, init_logits)
            probs = torch.softmax(proc, dim=-1)
            init_token = torch.multinomial(probs, 1).squeeze(-1)
        else:
            init_token = torch.argmax(init_logits, dim=-1)

        input_ids = torch.cat((input_ids, init_token.unsqueeze(-1).to(input_ids.device)), dim=1)

        # Seed SAM automaton with prompt tokens + init_token
        prompt_ids = input_ids[0].detach().cpu().tolist()
        self.draft.reset()
        self.draft.update(torch.tensor(prompt_ids, dtype=torch.long))

        if hasattr(self, "current_length_data"):
            current_length_data.fill_(input_ids.shape[1] - 1)

        for idx in range(max_length):
            curr_len = input_ids.shape[1]
            if curr_len - input_len >= max_new_tokens:
                break
            
            # Check for EOS token
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if is_llama3 and stop_token_id in input_ids[0, input_len:]:
                break
            
            ids = input_ids[0].detach().cpu().tolist()
            _, draft_tokens, _ = self.draft.lookup(ids[-1])

            cache_len = int(current_length_data[0].item()) if hasattr(self, "current_length_data") else 0
            last_tok = input_ids[:, -1:]
            # draft_tokens = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
            if draft_tokens.numel() == 0:
                out = self.forward(
                    input_ids=last_tok,
                    past_key_values=past_key_values,
                    output_orig=True,
                    **kwargs,
                )
                next_logits = out[1][:, -1, :]
                if logits_processor is not None:
                    proc = logits_processor(None, next_logits)
                    probs = torch.softmax(proc, dim=-1)
                    next_id = torch.multinomial(probs, 1).squeeze(1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1)
                input_ids = torch.cat((input_ids, next_id.unsqueeze(0)), dim=-1)
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
                    # Drop the newest repeated phrase once to avoid tail artifacts.
                    input_ids = input_ids[:, :-repeat_n]
                    if hasattr(self, "current_length_data"):
                        current_length_data.fill_(input_ids.shape[1] - 1)
                    break
                continue

            draft_tokens = draft_tokens.to(dtype=input_ids[0].dtype, device=input_ids[0].device).view(1, -1)
            chunk = torch.cat((last_tok, draft_tokens), dim=1)

            out = self.forward(
                input_ids=chunk,
                past_key_values=past_key_values,
                output_orig=True,
                **kwargs,
            )
            logits = out[1]

            D = draft_tokens.shape[1]
            
            draft_pred_logits = logits[:, 0: D + 1, :]
            draft_pred_ids = torch.argmax(draft_pred_logits, dim=-1)

            same = (draft_pred_ids[:, :-1] == draft_tokens)
            mismatch = (~same).squeeze(0).nonzero(as_tuple=True)[0]
            accept_len = int(mismatch[0].item()) if mismatch.numel() > 0 else D

            if return_acceptance_len:
                acceptance_len.append(accept_len)

            if accept_len > 0:
                input_ids = torch.cat([input_ids, draft_tokens[:, :accept_len]], dim=1)
                # Feed accepted draft tokens to SAM automaton
                self.draft.update(draft_tokens[0, :accept_len])

            next_id = draft_pred_ids[:, accept_len]
            if next_id is not None:
                self.draft.update(next_id)
            input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)

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

        new_token = input_ids.shape[1] - input_len
        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            outputs += (0.0,)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
