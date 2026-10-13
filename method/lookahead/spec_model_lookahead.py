
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Optional, List, Tuple
from transformers import AutoTokenizer, AutoConfig

from method.vispec.utils import *
from method.vispec.kv_cache import initialize_past_key_values
from .common.lookahead_cache import LookaheadCache


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        raw_ids = token_ids
    else:
        raw_ids = [token_ids]
    normalized = []
    for token_id in raw_ids:
        if token_id is None:
            continue
        if isinstance(token_id, torch.Tensor):
            if token_id.numel() == 1:
                token_id = token_id.item()
            else:
                normalized.extend(int(x) for x in token_id.view(-1).tolist())
                continue
        try:
            normalized.append(int(token_id))
        except (TypeError, ValueError):
            continue
    return normalized


def _collect_stop_token_ids(tokenizer, model):
    stop_ids = set(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        stop_ids.update(
            _normalize_token_ids(getattr(generation_config, "eos_token_id", None))
        )
    config = getattr(model, "config", None)
    if config is not None:
        stop_ids.update(_normalize_token_ids(getattr(config, "eos_token_id", None)))
    stop_ids.update(_normalize_token_ids(getattr(tokenizer, "eod_id", None)))
    return stop_ids


class SpecModel(nn.Module):
    def __init__(self, base_model, tokenizer=None, temperature=0.0, top_p=0.0, top_k=0, repetition_penalty=0.0, max_new_tokens=512):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        
        # Initialize Lookahead Cache
        self.lookahead_cache = LookaheadCache()
        
    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        base_model_path=None,
        spec_model_path=None,
        **kwargs
    ):
        # Load Base Model (Qwen2.5-VL support)
        # Copied/Adapted from spec_model_medusa.py
        
        torch_dtype = kwargs.get("torch_dtype", "auto")
        device_map = kwargs.get("device_map", "auto")
        low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage", True)
        model_load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        if kwargs.get("attn_implementation") is not None:
            model_load_kwargs["attn_implementation"] = kwargs["attn_implementation"]
        
        # Check architecture to load correct class
        config = AutoConfig.from_pretrained(base_model_path)
        arch = config.architectures[0]
        
        if arch == "Qwen2_5_VLForConditionalGeneration":
            from method.vispec.modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path,
                **model_load_kwargs,
            )
        elif arch == "LlavaForConditionalGeneration":
             from method.llava_adapter import CustomLlavaForConditionalGeneration
             model = CustomLlavaForConditionalGeneration.from_pretrained(
                base_model_path,
                **model_load_kwargs,
             )
        elif arch == "LlavaNextForConditionalGeneration":
             from method.llava_adapter import CustomLlavaNextForConditionalGeneration
             model = CustomLlavaNextForConditionalGeneration.from_pretrained(
                base_model_path,
                **model_load_kwargs,
             )
        else:
             # Fallback or generic
             from transformers import AutoModelForCausalLM
             model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_load_kwargs,
             )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
        
        # Instantiate SpecModel
        # Decoding parameters can be passed or defaulted
        return cls(
            model, 
            tokenizer=tokenizer,
            max_new_tokens=512 # Default, will be overridden by specgenerate
        )

    def _lookahead_prepare_inputs(self, decoding_ids, decoding_masks, past_length):
        """
        Prepare inputs for the speculative forward pass.
        decoding_ids: List[int], the tokens to process (including the root/last accepted token)
        decoding_masks: np.ndarray, the dependency mask [len, len]
        past_length: int, length of past key values
        """
        device = self.base_model.device
        decoding_length = len(decoding_ids)
        
        # 1. Input IDs
        new_input_ids = torch.tensor([decoding_ids], dtype=torch.long, device=device)
        
        # 2. Attention Mask
        # Construct full 4D mask [1, 1, new_len, past_len + new_len]
        # Query (new) can attend to Past (all) -> 1s
        query_past_mask = torch.ones((1, 1, decoding_length, past_length), dtype=torch.long, device=device)
        
        # Local mask from decoding_masks
        local_mask = torch.tensor(decoding_masks, dtype=torch.long, device=device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0) # [1, 1, new_len, new_len]
        
        full_mask = torch.cat([query_past_mask, local_mask], dim=-1)
        
        # Convert to causal mask format for Qwen2.5-VL (0 for allow, min_val for mask)
        dtype = self.base_model.dtype
        # min_val = torch.finfo(dtype).min
        min_val = -10000.0 # Safe value for softmax
        
        causal_mask = torch.zeros_like(full_mask, dtype=dtype)
        causal_mask.masked_fill_(full_mask == 0, min_val)
        
        # 3. Position IDs
        # Derived from local mask depth + past_length
        local_depth = local_mask.sum(dim=-1).squeeze(0).squeeze(0) # [new_len]
        position_ids = past_length + local_depth - 1
        position_ids = position_ids.unsqueeze(0).long() # [1, new_len]
        
        # Handle Qwen2.5-VL specific rope_deltas if present
        # In vispec, Qwen2.5-VL usually has rope_deltas attribute on the model wrapper or base?
        # Qwen2_5_VLForConditionalGeneration (from kv file) has `rope_deltas`.
        if hasattr(self.base_model, "rope_deltas") and self.base_model.rope_deltas is not None:
             position_ids = position_ids.unsqueeze(0) + self.base_model.rope_deltas
             
        return new_input_ids, causal_mask, position_ids

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
        return_acceptance_len=False,
        return_decode_time=False,
        inputs_embeds=None,
        **kwargs
    ):
        # Lookahead-specific parameters
        decoding_length = kwargs.get('decoding_length', 64)
        branch_length = kwargs.get('branch_length', 12)
        arch = self.base_model.config.architectures[0]
        is_qwen = arch == "Qwen2_5_VLForConditionalGeneration"
        is_llava = arch in (
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        )
        
        # Image/Video grid
        image_grid_thw = kwargs.get('image_grid_thw')
        video_grid_thw = kwargs.get('video_grid_thw')
        pixel_values = kwargs.get('pixel_values')
        image_sizes = kwargs.get('image_sizes')
        
        all_accept_lengths = []
        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        
        # Handle QwenVL image embeddings (copied from spec_model_medusa.py)
        if (
            self.base_model.config.architectures[0]
            == "Qwen2_5_VLForConditionalGeneration"
        ):
            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(
                        pixel_values, grid_thw=image_grid_thw
                    )
                    n_image_tokens = (
                        (input_ids == self.base_model.config.image_token_id)
                        .sum()
                        .item()
                    )
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == self.base_model.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask, image_embeds
                    )
        
        # 1. Init Lookahead Cache with input prompt
        prompt_list = input_ids[0].tolist()
        self.lookahead_cache.put(prompt_list, branch_length=branch_length, mode='input', idx=0)
        
        # 2. Init KV Cache (Vispec style)
        past_key_values, past_key_values_data_list, current_length_data = initialize_past_key_values(self.base_model)
        
        input_len = input_ids.shape[1]
        
        if return_decode_time:
            torch.cuda.synchronize()
            start_time = time.time()
        
        # 3. Prefill (Standard Forward)
        # Use inputs_embeds if available (for VL models with images)
        if inputs_embeds is not None and is_qwen:
            outputs = self.base_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw
            )
        elif is_llava:
            prefill_kwargs = {}
            if pixel_values is not None:
                prefill_kwargs["pixel_values"] = pixel_values
                if arch == "LlavaNextForConditionalGeneration" and image_sizes is not None:
                    prefill_kwargs["image_sizes"] = image_sizes
            outputs = self.base_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                **prefill_kwargs,
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        self.lookahead_cache.stream_put([next_token.item()], branch_length=branch_length, mode='output', idx=0)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_token_count = 1
        idx = 0
        
        # Loop
        while new_token_count < max_new_tokens:
            idx += 1
            # A. Retrieve Drafts
            d_ids, d_masks, _ = self.lookahead_cache.hier_get(input_ids[0].tolist(), decoding_length=decoding_length, branch_length=branch_length)
            
            # IMPORTANT: d_ids[0] is the query token itself! Skip it to get actual draft tokens
            # Reference: pretrained_model.py line 779: draft_ids = decoding_kwargs.get('decoding_ids', [])[1:]
            draft_ids = d_ids[1:] if len(d_ids) > 1 else []
            draft_masks = d_masks[1:, 1:] if len(d_ids) > 1 else np.ones((1, 1), dtype=np.int64)
            
            last_token_id = input_ids[0, -1].item()
            full_ids = [last_token_id] + draft_ids
            
            L = len(draft_ids)
            full_len = 1 + L
            full_mask_np = np.zeros((full_len, full_len), dtype=int)
            full_mask_np[0, 0] = 1 
            if L > 0:
                full_mask_np[1:, 0] = 1 
                full_mask_np[1:, 1:] = draft_masks 
            
            # B. Prepare Inputs
            past_length = current_length_data[0].item()
            new_input_ids, attention_mask, position_ids = self._lookahead_prepare_inputs(
                full_ids, full_mask_np, past_length
            )
            
            # C. Forward Speculative
            outputs = self.base_model(
                new_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits # [1, full_len, vocab]
            
            # D. Verification (simplified linear chain verification)
            # Reference: pretrained_model.py lines 805-860
            # For each position i, predict the next token and check if it matches draft_ids[i]
            accepted_tokens_list = []
            valid_path_indices = [0]  # Start with the root token position
            
            # full_ids = [last_token_id] + draft_ids
            # Position 0 (last_token_id) predicts what should match draft_ids[0]
            # Position i predicts what should match draft_ids[i]
            for i in range(full_len):
                pred_token = torch.argmax(logits[0, i]).item()
                accepted_tokens_list.append(pred_token)
                valid_path_indices.append(i + 1)  # The logit at i predicts token at i+1
                
                # Check if next draft token matches our prediction
                # Position i predicts position i+1. draft_ids[i] is the expected token.
                if i < len(draft_ids):
                    if draft_ids[i] != pred_token:
                        # Mismatch: the draft token doesn't match prediction
                        break
                else:
                    # No more draft tokens to verify
                    break
            
            # valid_path_indices should only include positions we actually used
            valid_path_indices = valid_path_indices[:len(accepted_tokens_list)]

            remaining_tokens = max_new_tokens - new_token_count
            if remaining_tokens <= 0:
                break
            if len(accepted_tokens_list) > remaining_tokens:
                accepted_tokens_list = accepted_tokens_list[:remaining_tokens]
                valid_path_indices = valid_path_indices[:remaining_tokens]

            reached_stop_token = False
            for token_idx, token_id in enumerate(accepted_tokens_list):
                if token_id in stop_token_ids:
                    accepted_tokens_list = accepted_tokens_list[: token_idx + 1]
                    valid_path_indices = valid_path_indices[: token_idx + 1]
                    reached_stop_token = True
                    break
            
            all_accept_lengths.append(len(accepted_tokens_list) - 1)  # exclude bonus token, consistent with eagle/medusa convention

            # E. Update State
            for t in accepted_tokens_list:
                input_ids = torch.cat([input_ids, torch.tensor([[t]], device=input_ids.device)], dim=1)
            
            new_token_count += len(accepted_tokens_list)
            
            self.lookahead_cache.stream_put(accepted_tokens_list, branch_length=branch_length, 
                                            final=(new_token_count >= max_new_tokens), mode='output', idx=0)
            
            # Update KV Cache
            select_indices = torch.tensor(valid_path_indices, device=past_key_values_data_list[0].device) + past_length
            prev_input_len = past_length
            
            for past_key_values_data in past_key_values_data_list:
                tgt = past_key_values_data[..., select_indices, :]
                dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)
                
            current_length_data.fill_(prev_input_len + len(valid_path_indices))
            
            if new_token_count >= max_new_tokens:
                break
            
            if reached_stop_token:
                break
        
        # Build outputs tuple (matching spec_model_medusa.py pattern)
        outputs = (input_ids,)
        if log:
            outputs += (new_token_count, idx)
        if return_acceptance_len:
            outputs += (all_accept_lengths,)
        if return_decode_time:
            torch.cuda.synchronize()
            end_time = time.time()
            outputs += (end_time - start_time,)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward(self, input_ids, **kwargs):
        return self.specgenerate(input_ids, **kwargs)
