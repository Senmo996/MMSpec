from types import SimpleNamespace

import torch
from torch import nn

from method.vispec.qshape_fixed_row import (
    FixedRowLinear,
    enable_qshape_exact_root_attention,
    enable_qshape_fixed_row_attention,
    enable_qshape_fixed_row_linears,
)


def test_fixed_row_linear_makes_root_shape_invariant():
    torch.manual_seed(3)
    linear = nn.Linear(8, 12, bias=False)
    wrapped = FixedRowLinear(linear, fixed_rows=4)
    root = torch.randn(1, 1, 8)
    packed = torch.cat([root, torch.randn(1, 2, 8)], dim=1)

    single_output = wrapped(root)
    packed_output = wrapped(packed)

    assert single_output.shape == (1, 1, 12)
    assert packed_output.shape == (1, 3, 12)
    assert torch.equal(single_output[0, 0], packed_output[0, 0])
    assert wrapped.weight is linear.weight


def test_fixed_row_linear_falls_back_above_fixed_rows():
    torch.manual_seed(5)
    linear = nn.Linear(4, 6, bias=True)
    wrapped = FixedRowLinear(linear, fixed_rows=4)
    value = torch.randn(1, 5, 4)

    assert torch.equal(wrapped(value), linear(value))


def test_enable_qshape_fixed_row_linears_wraps_decoder_and_head():
    layers = []
    for _ in range(2):
        mlp = SimpleNamespace(
            gate_proj=nn.Linear(4, 8, bias=False),
            up_proj=nn.Linear(4, 8, bias=False),
            down_proj=nn.Linear(8, 4, bias=False),
        )
        layers.append(SimpleNamespace(mlp=mlp, self_attn=SimpleNamespace()))
    model = nn.Module()
    model.model = SimpleNamespace(layers=layers)
    model.lm_head = nn.Linear(4, 10, bias=False)

    wrapped = enable_qshape_fixed_row_linears(model, fixed_rows=8)

    assert wrapped == 7
    assert isinstance(model.lm_head, FixedRowLinear)
    assert all(
        isinstance(getattr(layer.mlp, name), FixedRowLinear)
        for layer in layers
        for name in ("gate_proj", "up_proj", "down_proj")
    )
    assert enable_qshape_fixed_row_linears(model, fixed_rows=16) == 0
    assert model.lm_head.fixed_rows == 16
    assert (
        enable_qshape_fixed_row_attention(
            model, fixed_rows=32, fixed_key_block=256
        )
        == 2
    )
    assert all(layer.self_attn.qshape_fixed_rows == 32 for layer in layers)
    assert all(
        layer.self_attn.qshape_fixed_key_block == 256 for layer in layers
    )
    assert enable_qshape_exact_root_attention(model, max_rows=32) == 2
    assert all(
        layer.self_attn.qshape_exact_root_attention_rows == 32
        for layer in layers
    )
