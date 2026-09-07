from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from method.llava_adapter import (
    _build_tree_attention_mask,
    _patch_tree_mask_for_llama_model,
)
from method.sam_grounded.spec_model import (
    resolve_generation_max_length,
    resolve_image_token_id,
)


def test_resolve_llava_image_token_index():
    config = SimpleNamespace(
        architectures=["LlavaNextForConditionalGeneration"],
        image_token_index=32000,
        image_token_id=123,
    )

    assert resolve_image_token_id(config) == 32000


def test_llava_max_new_tokens_override_legacy_total_length_cap():
    config = SimpleNamespace(
        architectures=["LlavaNextForConditionalGeneration"],
    )

    assert resolve_generation_max_length(config, 2500, 200, 2048) == 2700


def test_qwen_keeps_legacy_total_length_cap():
    config = SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
    )

    assert resolve_generation_max_length(config, 1800, 200, 1900) == 1900


def test_build_tree_attention_mask_keeps_prefix_and_blocks_siblings():
    tree_mask = torch.tensor(
        [[[[1, 0], [1, 1]]]],
        dtype=torch.float32,
    )
    input_tensor = torch.zeros(1, 2, 4, dtype=torch.float16)

    mask = _build_tree_attention_mask(tree_mask, 3, input_tensor)

    assert mask.shape == (1, 1, 2, 5)
    assert torch.equal(mask[..., :3], torch.zeros_like(mask[..., :3]))
    assert mask[0, 0, 0, 3].item() == 0.0
    assert mask[0, 0, 0, 4].item() == torch.finfo(torch.float16).min
    assert torch.equal(mask[0, 0, 1, 3:], torch.zeros(2, dtype=torch.float16))


def test_current_transformers_llama_tree_mask_patch_runs_with_cache():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    language_model = LlamaForCausalLM(config)
    wrapper = SimpleNamespace(language_model=language_model)
    _patch_tree_mask_for_llama_model(wrapper)
    cache = DynamicCache(config=config)

    language_model(
        input_ids=torch.tensor([[1, 2, 3]]),
        past_key_values=cache,
        use_cache=True,
    )
    language_model.model.tree_mask = torch.tensor(
        [[[[1, 0], [1, 1]]]],
        dtype=torch.float32,
    )
    output = language_model(
        input_ids=torch.tensor([[4, 5]]),
        past_key_values=cache,
        position_ids=torch.tensor([[3, 4]]),
        use_cache=True,
    )

    assert output.logits.shape == (1, 2, 32)
    assert cache.get_seq_length() == 5
    assert language_model.model._mmspec_tree_mask_patched is True


def test_current_llava_layout_patches_direct_llama_model():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    direct_model = LlamaForCausalLM(config).model

    _patch_tree_mask_for_llama_model(
        SimpleNamespace(language_model=direct_model)
    )

    assert direct_model._mmspec_tree_mask_patched is True
