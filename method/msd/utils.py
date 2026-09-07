import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

class temp_cache:
    use_msd = False
    sample_accept_token = []
    sample_reject_token = []
    answer_file = None

    total_time = 0
    total_in_num = 0
    total_out_num = 0
    acceptance_len = 0

    eval_model = None
    n_alpha = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    

    @classmethod
    def reset(cls):
        cls.use_msd = False
        cls.sample_accept_token = []
        cls.sample_reject_token = []
        cls.total_time = 0
        cls.acceptance_len = 0
        cls.total_out_num = 0
        cls.total_in_num = 0
        cls.n_alphas = [0,0,0,0,0,0,0,0,0,0,0,0,0]

class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def initialize_tree(input_ids, model, past_key_values, logits_processor, inputs_embeds=None):

    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True, inputs_embeds=inputs_embeds
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    # first forward can gengerate a token,concat it to origin input_ids
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    # Clone the output hidden states

    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor, inputs_embeds)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token


def reset_tree_mode(
        model,
):
    # Prefer the wrapper helper because current HF LLaVA exposes LlamaModel
    # directly while older releases exposed a causal-LM wrapper around it.
    if hasattr(model, '_get_language_model'):
        lang_model = model._get_language_model()
    elif hasattr(model.base_model, 'language_model'):
        language_model = model.base_model.language_model
        lang_model = getattr(language_model, 'model', language_model)
    elif hasattr(model.base_model, 'model'):
        lang_model = model.base_model.model
    else:
        return
    
    lang_model.tree_mask = None
    lang_model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        output_attentions=False,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

    if -200 in input_ids:
        position_ids += 575
    # temp_cache is defined in this module

    # tree_candidates = tree_candidates[:, :2]
    # position_ids = position_ids[:2]
    # model.base_model.model.tree_mask = None
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        output_attentions=output_attentions,
    )



    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    if -200 in input_ids:
        prev_input_len += 575 
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head,logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token


@torch.no_grad()
def get_input_embeds_qwenvl(input_ids, model):

    input_embeds = model.transformer.wte(input_ids)

    if torch.any(input_ids == model.transformer.config.visual['image_start_id']):
        bos_pos = torch.where(input_ids == model.transformer.config.visual['image_start_id'])
        eos_pos = torch.where(input_ids == model.transformer.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        images = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1 : b - 1].tolist()
            image = image[:image.index(model.transformer.config.visual['image_start_id'] + 2)]
            images.append(bytes(image).decode('utf-8'))

        images = model.transformer.visual.encode(images)
        assert images.shape[0] == len(images)

        for idx, (i, a, b) in enumerate(img_pos):
            input_embeds[i][a + 1 : b] = images[idx]

    return input_embeds

def get_input_embeds_qwen2vl(input_ids, pixel_values, image_grid_thw, model):
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            if hasattr(model.visual, "get_dtype"):
                visual_dtype = model.visual.get_dtype()
            else:
                try:
                    visual_dtype = next(model.visual.parameters()).dtype
                except StopIteration:
                    visual_dtype = inputs_embeds.dtype
            pixel_values = pixel_values.to(dtype=visual_dtype)
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds



# ============================================================================
# HF LLaVA Compatible Functions for DynamicCache
# ============================================================================


def cache_sequence_length(past_key_values):
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache and key_cache[0] is not None:
        return int(key_cache[0].shape[-2])
    return 0


def _cache_layer_count(past_key_values):
    if hasattr(past_key_values, "layers"):
        return len(past_key_values.layers)
    return len(getattr(past_key_values, "key_cache", ()))


def _cache_layer_tensors(past_key_values, layer_idx):
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        if not getattr(layer, "is_initialized", True):
            return None, None
        return layer.keys, layer.values
    return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]


def _set_cache_layer_tensors(past_key_values, layer_idx, keys, values):
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        layer.keys = keys
        layer.values = values
    else:
        past_key_values.key_cache[layer_idx] = keys
        past_key_values.value_cache[layer_idx] = values


def select_cache_sequence(past_key_values, prefix_length, select_indices):
    """Keep the prefix and selected tree positions in a DynamicCache."""
    for layer_idx in range(_cache_layer_count(past_key_values)):
        keys, values = _cache_layer_tensors(past_key_values, layer_idx)
        if keys is None or values is None:
            continue
        layer_select_indices = select_indices.to(keys.device)
        prefix_keys = keys[..., :prefix_length, :]
        prefix_values = values[..., :prefix_length, :]
        selected_keys = keys.index_select(-2, layer_select_indices)
        selected_values = values.index_select(-2, layer_select_indices.to(values.device))
        _set_cache_layer_tensors(
            past_key_values,
            layer_idx,
            torch.cat([prefix_keys, selected_keys], dim=-2),
            torch.cat([prefix_values, selected_values], dim=-2),
        )

def initialize_tree_hf(input_ids, model, past_key_values, logits_processor, inputs_embeds=None):
    """
    Initialize tree for HF LLaVA using -200 marker mechanism.
    
    Creates input_ids with -200 at image position and inputs_embeds with 576 image embeddings.
    EA layer will handle -200 expansion correctly with its stable_kv.
    """
    from transformers import DynamicCache
    
    # Forward through HF LLaVA
    if inputs_embeds is not None:
        outputs = model.base_model(
            input_ids=None, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, use_cache=True, output_hidden_states=True,
        )
    else:
        outputs = model.base_model(
            input_ids=input_ids, past_key_values=past_key_values,
            use_cache=True, output_hidden_states=True,
        )
    
    orig = outputs.logits
    hidden_states = outputs.hidden_states[-1]
    
    # Sample first token
    if logits_processor is not None:
        logits = logits_processor(None, orig[:, -1])
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])[None, None]
    
    # Calculate image info from input_ids directly (HF LLaVA often repeats image marker tokens).
    input_seq_len = input_ids.shape[1]
    image_token_id = getattr(model.base_model.config, 'image_token_index', 32000)
    image_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    
    if len(image_pos) > 0 and inputs_embeds is not None:
        image_start = image_pos[0].item()
        image_end = image_start
        # Only collapse the first contiguous image marker block.
        while (
            image_end + 1 < input_seq_len
            and input_ids[0, image_end + 1].item() == image_token_id
        ):
            image_end += 1
        num_image_tokens = image_end - image_start + 1
        
        # Create ea_input_ids with a single -200 marker for the whole image block.
        before = input_ids[0, :image_start]
        after = input_ids[0, image_end + 1 :]
        marker = torch.tensor([-200], dtype=input_ids.dtype, device=input_ids.device)
        ea_input_ids = torch.cat([before, marker, after]).unsqueeze(0)
        
        # For topK_genrate call, add the token
        ea_input_ids_with_token = torch.cat([ea_input_ids, token], dim=1)
        
        # Create ea_inputs_embeds with image embeddings at correct position
        text_before = model.ea_layer.embed_tokens(before.unsqueeze(0))
        text_after = model.ea_layer.embed_tokens(after.unsqueeze(0))
        token_embed = model.ea_layer.embed_tokens(token)
        image_embeds = inputs_embeds[0, image_start:image_start + num_image_tokens, :].unsqueeze(0)
        
        ea_inputs_embeds = torch.cat([
            text_before, image_embeds, text_after, token_embed
        ], dim=1)
    else:
        ea_input_ids = input_ids.clone()
        ea_input_ids_with_token = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
        ea_inputs_embeds = None
    
    # Generate draft tokens
    lm_head = model._get_lm_head()
    temp_cache.use_msd = True
    try:
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            hidden_states, ea_input_ids_with_token, lm_head, logits_processor, ea_inputs_embeds
        )
    finally:
        temp_cache.use_msd = False
    
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, orig, hidden_states, token, outputs.past_key_values, ea_input_ids, ea_inputs_embeds



def tree_decoding_hf(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        tree_mask=None,  # Tree attention mask
):
    """
    Tree decoding for HF LLaVA models using DynamicCache.
    
    Args:
        model: EaModel wrapping HF LLaVA
        tree_candidates: Draft token candidates
        past_key_values: DynamicCache from previous step
        tree_position_ids: Position IDs for tree tokens
        input_ids: Current input IDs
        retrieve_indices: Indices for retrieving correct path
        tree_mask: Tree attention mask for proper tree attention
        
    Returns:
        logits, hidden_state, outputs
    """
    # For HF LLaVA: use KV cache length for position_ids, not input_ids length
    # input_ids doesn't include expanded image tokens, but KV cache does
    kv_seq_len = cache_sequence_length(past_key_values) or input_ids.shape[1]
    position_ids = tree_position_ids + kv_seq_len
    
    if position_ids is not None and position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    
    # Create attention_mask for tree decoding if tree_mask is provided
    attention_mask = None
    if tree_mask is not None:
        # tree_mask is [1, 1, tree_len, tree_len]
        # We need to create a full attention mask that combines:
        # 1. Causal attention for past KV cache
        # 2. Tree attention for new tokens
        tree_len = tree_candidates.shape[1]
        
        # Move tree_mask to correct device
        tree_mask = tree_mask.to(tree_candidates.device)
        
        # Create full attention mask: [batch, 1, tree_len, kv_seq_len + tree_len]
        # All tree tokens can attend to all past tokens
        past_mask = torch.ones((1, 1, tree_len, kv_seq_len), device=tree_candidates.device, dtype=tree_mask.dtype)
        
        # Tree attention for new tokens - use provided tree_mask
        # tree_mask should be [1, 1, tree_len, tree_len] with 1=attend, 0=mask
        full_attention_mask = torch.cat([past_mask, tree_mask], dim=-1)
        
        # Get model dtype and convert attention mask to match query dtype
        model_dtype = next(model.base_model.parameters()).dtype
        
        # Convert to HF attention mask format (0 = attend, -inf = mask)
        attention_mask = (1.0 - full_attention_mask) * torch.finfo(model_dtype).min
        attention_mask = attention_mask.to(model_dtype)
    
    # Forward through HF LLaVA language model
    outputs = model.base_model(
        input_ids=tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
    )
    
    tree_logits = outputs.logits
    hidden_state = outputs.hidden_states[-1]
    
    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


@torch.no_grad()
def update_inference_inputs_hf(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values,  # DynamicCache
        model,
        hidden_state_new,
        sample_p,
        ea_input_ids,      # Input IDs with -200 marker for EA layer
        ea_inputs_embeds   # Embeddings with image features for EA layer
):
    """
    Update inference inputs for HF LLaVA using -200 marker mechanism.
    
    Uses ea_input_ids (with -200 marker) and ea_inputs_embeds (with image embeddings)
    for proper EA layer handling with stable_kv.
    
    IMPORTANT: For HF LLaVA, input_ids length != KV cache length because image tokens
    are expanded from 1 to 576 tokens in the KV cache. We compute the original KV cache
    length (before tree decoding) as: input_ids.shape[1] + 575 (if image present).
    """
    prev_input_len = input_ids.shape[1]
    
    # Derive pre-tree KV length directly from cache state. This is robust across
    # different multimodal tokenization layouts.
    kv_total_len = cache_sequence_length(past_key_values)
    if kv_total_len > 0:
        tree_len = hidden_state_new.shape[1]
        kv_len_before_tree = kv_total_len - tree_len
    else:
        kv_len_before_tree = prev_input_len
    
    # Map the best candidate indices to the original indices in the KV cache
    # Use kv_len_before_tree (KV cache length before tree tokens were added)
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + kv_len_before_tree
    )
    
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    
    # Also update ea_input_ids with the same accepted tokens
    ea_input_ids = torch.cat(
        [ea_input_ids, candidates[None, best_candidate, : accept_length + 1].to(ea_input_ids.device)], dim=-1
    )
    
    # Update DynamicCache - we need to truncate and keep only accepted tokens
    # The cache currently has all tree tokens, we need to select only the accepted path
    
    select_cache_sequence(
        past_key_values,
        prefix_length=kv_len_before_tree,
        select_indices=select_indices,
    )
    
    # Get accepted hidden states and generate new draft tokens
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    
    # Generate new draft tokens
    lm_head = model._get_lm_head()
    
    # Update ea_inputs_embeds with embeddings for accepted tokens
    if ea_inputs_embeds is not None:
        accepted_tokens = candidates[best_candidate, : accept_length + 1].to(input_ids.device)
        accepted_embeds = model.ea_layer.embed_tokens(accepted_tokens.unsqueeze(0))
        ea_inputs_embeds = torch.cat([ea_inputs_embeds, accepted_embeds], dim=1)
    
    temp_cache.use_msd = True
    try:
        # Pass ea_input_ids (with -200) and ea_inputs_embeds (with image embeddings)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((ea_input_ids, token.to(ea_input_ids.device)), dim=1),
            head=lm_head,
            logits_processor=logits_processor,
            inputs_embeds=ea_inputs_embeds
        )
    finally:
        temp_cache.use_msd = False
    
    new_token += accept_length + 1
    
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token, past_key_values, ea_input_ids, ea_inputs_embeds


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
