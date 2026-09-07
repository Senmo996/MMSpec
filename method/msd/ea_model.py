import copy
import json
import time
import sys
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

# Add LLaVA from MSD to path for LlavaLlamaForCausalLM
_msd_llava_path = os.path.join(os.path.dirname(__file__), "..", "MSD", "LLaVA")
if os.path.exists(_msd_llava_path) and _msd_llava_path not in sys.path:
    sys.path.insert(0, _msd_llava_path)

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_qwen2vl_kv import Qwen2VLForConditionalGeneration as KVQwen2VLForCausalLM 
from method.eagle.modeling_qwen2_5_vl_kv import (
    Qwen2_5_VLForConditionalGeneration as KVQwen2_5VLForCausalLM,
)
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .configs import EConfig, Qwen2VLConfig


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


def _has_stop_token(generated_ids, stop_ids):
    if not stop_ids:
        return False
    if isinstance(generated_ids, torch.Tensor):
        generated_set = set(int(x) for x in generated_ids.view(-1).tolist())
    else:
        generated_set = set(int(x) for x in generated_ids)
    return bool(generated_set & stop_ids)


def _truncate_at_first_stop_token(input_ids, input_len, stop_ids):
    """Truncate generated sequence at the first stop token (inclusive)."""
    generated = input_ids[0, input_len:]
    if generated.numel() == 0:
        return input_ids, 0, False
    if not stop_ids:
        return input_ids, int(generated.shape[0]), False

    first_stop_pos = None
    for idx, token_id in enumerate(generated.tolist()):
        if int(token_id) in stop_ids:
            first_stop_pos = idx
            break

    if first_stop_pos is None:
        return input_ids, int(generated.shape[0]), False

    truncated = input_ids[:, : input_len + first_stop_pos + 1]
    return truncated, int(first_stop_pos + 1), True


class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
        # Handle different model structures (HF LLaVA uses language_model.lm_head)
        if hasattr(base_model, 'lm_head'):
            lm_head = base_model.lm_head
        elif hasattr(base_model, 'language_model') and hasattr(base_model.language_model, 'lm_head'):
            lm_head = base_model.language_model.lm_head
        else:
            raise AttributeError("Cannot find lm_head in base_model")
        
        self.hidden_size = lm_head.weight.shape[-1]
        self.vocab_size = lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False,trust_remote_code=True)
        if "Qwen" in self.base_model_name_or_path:
            config = Qwen2VLConfig.from_pretrained(ea_model_path)
        else:
            config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)

        low_memory=False

        # Handle different model structures for getting device
        try:
            device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        except:
            try:
                device = base_model.language_model.model.layers[-1].self_attn.q_proj.weight.device
            except:
                try:
                    device = base_model.model.h[-1].attn.c_attn.weight.device
                except:
                    device = lm_head.weight.device
                    
        if device != lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        # Shape validation (fail-fast): every checkpoint key that exists in
        # the model must have exactly the same shape — no silent truncation.
        model_sd = self.ea_layer.state_dict()
        for k, v in ea_layer_state_dict.items():
            if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape):
                raise RuntimeError(
                    f"[msd] Shape mismatch for '{k}': "
                    f"checkpoint={tuple(v.shape)}, model={tuple(model_sd[k].shape)}. "
                    f"Please check that the spec config matches the checkpoint."
                )
        missing_keys, unexpected_keys = self.ea_layer.load_state_dict(
            ea_layer_state_dict, strict=False
        )
        if len(missing_keys) > 0:
            print(f"missing_keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"unexpected_keys: {unexpected_keys}")
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

        self.acclen = 0
        self.accnum = 0
        
        # Store reference to lm_head for easy access
        self._lm_head = lm_head
        
        # Model family flags used by input embedding helpers.
        self._is_qwen2vl = getattr(self.config, "model_type", None) in {"qwen2_vl", "qwen2_5_vl"}
        self._is_hf_llava = hasattr(base_model, "language_model")

    def _get_language_model(self):
        """Get the language model part (handles both original and HF LLaVA)."""
        if self._is_hf_llava:
            language_model = self.base_model.language_model
            return getattr(language_model, "model", language_model)
        else:
            return self.base_model.model
    
    def _get_lm_head(self):
        """Get the LM head (handles both original and HF LLaVA)."""
        return self._lm_head
    
    def get_inputs_embeds(self, input_ids, pixel_values=None, image_sizes=None, image_grid_thw=None, attention_mask=None):
        """Get input embeddings for multimodal input (HF LLaVA compatible).
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            pixel_values: Image tensor from processor [batch, channels, height, width]
            image_sizes: Optional list of image sizes
            
        Returns:
            inputs_embeds: Combined text and image embeddings
            attention_mask: Attention mask
        """
        if self._is_qwen2vl:
            input_ids = input_ids.to(self.base_model.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.base_model.device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.base_model.device)

            inputs_embeds = get_input_embeds_qwen2vl(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                model=self.base_model,
            )
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
                )
            else:
                attention_mask = attention_mask.to(inputs_embeds.device)
            return inputs_embeds, attention_mask

        if self._is_hf_llava:
            # For HF LLaVA, manually merge image features with text embeddings
            if pixel_values is not None:
                # Move tensors to correct device and dtype
                pixel_values = pixel_values.to(self.base_model.device, self.base_model.dtype)
                input_ids = input_ids.to(self.base_model.device)
                
                # Get image features using the vision tower and projector
                image_outputs = self.base_model.vision_tower(
                    pixel_values,
                    output_hidden_states=True
                )
                selected_image_feature = image_outputs.hidden_states[-2]
                selected_image_feature = selected_image_feature[:, 1:]  # Remove CLS token
                image_features = self.base_model.multi_modal_projector(selected_image_feature)
                # image_features shape: [batch, 576, hidden_size]
                
                # Get text embeddings
                text_embeds = self.base_model.get_input_embeddings()(input_ids)
                # text_embeds shape: [batch, seq_len, hidden_size]
                
                # Find image token position (token ID 32000 for LLaVA)
                image_token_id = self.base_model.config.image_token_index if hasattr(self.base_model.config, 'image_token_index') else 32000
                
                batch_size = input_ids.shape[0]
                new_embeds_list = []
                
                for batch_idx in range(batch_size):
                    # HF LLaVA may represent an image as either:
                    # 1) a single image marker token, or
                    # 2) a contiguous block of repeated image marker tokens (e.g., 576x).
                    image_positions = (input_ids[batch_idx] == image_token_id).nonzero(as_tuple=True)[0]

                    if len(image_positions) > 0:
                        n_image_tokens = int(image_positions.numel())
                        n_image_features = int(image_features.shape[1])

                        if n_image_tokens == n_image_features:
                            # Preferred path: in-place replacement, sequence length unchanged.
                            merged = text_embeds[batch_idx].clone()
                            merged[image_positions] = image_features[batch_idx]
                            new_embeds_list.append(merged)
                        elif n_image_tokens == 1:
                            # Backward-compatible path: expand one marker to visual patch embeddings.
                            image_pos = image_positions[0].item()
                            before_image = text_embeds[batch_idx, :image_pos]
                            after_image = text_embeds[batch_idx, image_pos + 1:]
                            merged = torch.cat(
                                [before_image, image_features[batch_idx], after_image], dim=0
                            )
                            new_embeds_list.append(merged)
                        else:
                            raise ValueError(
                                "HF LLaVA image token/feature mismatch: "
                                f"{n_image_tokens} image tokens vs {n_image_features} image features"
                            )
                    else:
                        # No image token, use text embeddings as-is
                        new_embeds_list.append(text_embeds[batch_idx])
                
                # Stack back to batch
                inputs_embeds = torch.stack(new_embeds_list, dim=0)
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
                
                return inputs_embeds, attention_mask
            else:
                # No image, just get text embeddings
                text_embeds = self.base_model.get_input_embeddings()(input_ids)
                attention_mask = torch.ones(text_embeds.shape[:2], device=text_embeds.device, dtype=torch.long)
                return text_embeds, attention_mask
        else:
            # For original LLaVA, use the existing method
            return self.base_model.get_inputs_embeds(input_ids, pixel_values, image_sizes)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path, trust_remote_code=True).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=='LlavaLlamaForCausalLM':
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            kwargs.pop('low_cpu_mem_usage', None)
            base_model_model_name = get_model_name_from_path(base_model_path)
            base_tokenizer, base_model, image_processor, base_context_len = load_pretrained_model(base_model_path, None, base_model_model_name, **kwargs)
        elif Type=='LlavaForConditionalGeneration':
            # HuggingFace transformers version of LLaVA (llava-hf)
            from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
            base_model = LlavaForConditionalGeneration.from_pretrained(
                base_model_path, **kwargs
            )
            base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            processor = AutoProcessor.from_pretrained(base_model_path)
            image_processor = processor.image_processor
            base_context_len = 2048
        elif Type=='Qwen2VLForConditionalGeneration':
            base_model=KVQwen2VLForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
            from transformers import AutoProcessor, AutoTokenizer
            base_tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, use_fast=False, trust_remote_code=True
            )
            image_processor = AutoProcessor.from_pretrained(
                base_model_path, trust_remote_code=True
            )
            base_context_len = getattr(base_model.config, "max_position_embeddings", 32768)
        elif Type=='Qwen2_5_VLForConditionalGeneration':
            base_model = KVQwen2_5VLForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
            from transformers import AutoProcessor, AutoTokenizer
            base_tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, use_fast=False, trust_remote_code=True
            )
            image_processor = AutoProcessor.from_pretrained(
                base_model_path, trust_remote_code=True
            )
            base_context_len = getattr(base_model.config, "max_position_embeddings", 32768)
        else:
            raise NotImplementedError(f"Unsupported base model architecture for MSD: {Type}")


        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath) and not os.path.isdir(ea_model_path):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path) and not os.path.isdir(ea_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location="cpu")
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path) and not os.path.isdir(ea_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)

        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )



        if total_token==-1:
            try:
                device = base_model.model.layers[-1].self_attn.q_proj.weight.device
            except:
                try:
                    device = base_model.language_model.model.layers[-1].self_attn.q_proj.weight.device
                except:
                    try:
                        device = base_model.model.h[-1].attn.c_attn.weight.device
                    except:
                        # Fallback to lm_head device
                        if hasattr(base_model, 'lm_head'):
                            device = base_model.lm_head.weight.device
                        else:
                            device = base_model.language_model.lm_head.weight.device

            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            try:
                # Try custom KV cache (works for standalone LLM models)
                (
                    past_key_values,
                    past_key_values_data_list,
                    current_length_data,
                ) = initialize_past_key_values(base_model)

                for i in range(len(cans)):
                    length = cans[i]
                    input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                    current_length_data.zero_()
                    for pkvd in past_key_values_data_list:
                        pkvd.zero_()
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(20):
                        torch.cuda.synchronize()
                        with torch.no_grad():
                            outputs = model.base_model(input_ids, past_key_values=past_key_values)
                        current_length_data.zero_()
                        for pkvd in past_key_values_data_list:
                            pkvd.zero_()
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append((end_time - start_time) / x[i])
            except (ValueError, TypeError):
                # VLM wrappers (e.g. LlavaForConditionalGeneration) reject
                # list-form past_key_values — fall back to no KV cache
                times = []
                for i in range(len(cans)):
                    length = cans[i]
                    input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(20):
                        torch.cuda.synchronize()
                        with torch.no_grad():
                            outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append((end_time - start_time) / x[i])

            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1

        if 'image_processor' in locals():
            return base_tokenizer, model, image_processor, base_context_len

        return model, None

    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            output_attentions=False,
    ):

        with torch.inference_mode():
            # Pass input through the base model (use helper for HF LLaVA compatibility)
            language_model = self._get_language_model()
            if inputs_embeds is not None:
                outputs = language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            else:
                outputs = language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )
            if output_orig:
                orig = self._get_lm_head()(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states


    @torch.no_grad()
    def msdgenerate(
            self,
            input_ids,
            inputs_embeds=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
    ):
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        input_len = input_ids.shape[1]
        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        reset_tree_mode(self)

        # HF LLaVA uses DynamicCache and different tree functions
        if self._is_hf_llava:
            from transformers import DynamicCache
            from .utils import (
                initialize_tree_hf,
                tree_decoding_hf,
                update_inference_inputs_hf,
                evaluate_posterior as evaluate_posterior_hf,
            )
            
            past_key_values = DynamicCache()
            
            # Initialize tree with HF-compatible function
            # Returns ea_input_ids with -200 marker and ea_inputs_embeds with image embeddings
            result = initialize_tree_hf(
                input_ids, self, past_key_values, logits_processor, inputs_embeds
            )
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token, past_key_values, ea_input_ids, ea_inputs_embeds = result
            new_token = 0

            for idx in range(max_length):
                draft_tokens = draft_tokens.to(input_ids.device)
                
                # Tree decoding with HF-compatible function
                logits, hidden_state_new, outputs = tree_decoding_hf(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                    tree_mask=tree_mask,  # Pass tree attention mask
                )
                past_key_values = outputs.past_key_values
                
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]
                best_candidate, accept_length, sample_p = evaluate_posterior_hf(
                    logits, candidates, logits_processor
                )
                self.acclen += accept_length
                self.accnum += 1
                
                # Update inputs with HF-compatible function
                # Pass ea_input_ids and ea_inputs_embeds for proper -200 mechanism
                result = update_inference_inputs_hf(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values,
                    self,
                    hidden_state_new,
                    sample_p,
                    ea_input_ids,
                    ea_inputs_embeds  # Pass ea_inputs_embeds for -200 mechanism
                )
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token, past_key_values, ea_input_ids, ea_inputs_embeds = result

                input_ids, new_token, hit_stop = _truncate_at_first_stop_token(
                    input_ids, input_len, stop_token_ids
                )
                if hit_stop:
                    break
                if new_token >= max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        else:
            # Original logic for non-HF models using custom KVCache
            if hasattr(self, "past_key_values"):
                past_key_values = self.past_key_values
                past_key_values_data = self.past_key_values_data
                current_length_data = self.current_length_data
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data

            draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor, inputs_embeds
            )
            new_token = 0

            for idx in range(max_length):
                self._get_language_model().tree_mask = tree_mask

                draft_tokens = draft_tokens.to(input_ids.device)
                logits, hidden_state_new, outputs = tree_decoding(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]
                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
                self.acclen += accept_length
                self.accnum += 1
                
                input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p
                )
                input_ids, new_token, hit_stop = _truncate_at_first_stop_token(
                    input_ids, input_len, stop_token_ids
                )
                if hit_stop:
                    break
                if new_token >= max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx


    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            inputs_embeds=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
    ):
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()
        input_len = input_ids.shape[1]
        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        reset_tree_mode(self)

        # For HF LLaVA, use DynamicCache instead of custom KVCache
        if self._is_hf_llava:
            from transformers import DynamicCache
            past_key_values = DynamicCache()
            
            # First forward pass with image embeddings
            if inputs_embeds is not None:
                outputs = self.base_model(
                    input_ids=None, 
                    inputs_embeds=inputs_embeds, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
            else:
                outputs = self.base_model(
                    input_ids=input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
            past_key_values = outputs.past_key_values
            
            new_token = 0
            for idx in range(max_length):
                if logits_processor is not None:
                    logits = outputs.logits[:, -1]
                    logits = logits_processor(None, logits)
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    input_id = torch.multinomial(probabilities, 1)
                else:
                    input_id = outputs.logits[:, -1:].argmax(dim=-1)
                
                outputs = self.base_model(
                    input_ids=input_id, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                input_ids, new_token, hit_stop = _truncate_at_first_stop_token(
                    input_ids, input_len, stop_token_ids
                )
                if hit_stop:
                    break
                if new_token >= max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        else:
            # Original logic for non-HF models using custom KVCache
            padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
            
            if hasattr(self, "past_key_values"):
                past_key_values = self.past_key_values
                past_key_values_data = self.past_key_values_data
                current_length_data = self.current_length_data
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data

            if inputs_embeds is not None:
                if self.base_model.config.model_type in {"qwen2_vl", "qwen2_5_vl"}:
                    outputs = self.base_model(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
                else:
                    outputs = self.base_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            else:
                outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)

            new_token = 0
            for idx in range(max_length):
                if logits_processor is not None:
                    logits = outputs.logits[:, -1]
                    logits = logits_processor(None, logits)
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    input_id = torch.multinomial(probabilities, 1)
                else:
                    input_id = outputs.logits[:, -1:].argmax(dim=-1)
                outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                input_ids, new_token, hit_stop = _truncate_at_first_stop_token(
                    input_ids, input_len, stop_token_ids
                )
                if hit_stop:
                    break
                if new_token >= max_new_tokens:
                    break
                if input_ids.shape[1] > max_length:
                    break
        
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx
