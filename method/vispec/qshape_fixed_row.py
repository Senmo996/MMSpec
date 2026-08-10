"""Fixed-row wrappers for q-length-stable short-sequence linear layers."""

from __future__ import annotations

import torch
from torch import nn


class FixedRowLinear(nn.Module):
    """Run short inputs as one fixed-row GEMM and slice away padding rows."""

    def __init__(self, linear: nn.Linear, fixed_rows: int = 32):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("FixedRowLinear expects torch.nn.Linear")
        if int(fixed_rows) <= 0:
            raise ValueError("fixed_rows must be positive")
        self.linear = linear
        self.fixed_rows = int(fixed_rows)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rows = int(value.numel() // value.shape[-1])
        if rows <= 0 or rows > self.fixed_rows:
            return self.linear(value)
        flat = value.reshape(rows, value.shape[-1])
        if rows < self.fixed_rows:
            flat = torch.cat(
                [
                    flat,
                    flat.new_zeros(
                        (self.fixed_rows - rows, flat.shape[-1])
                    ),
                ],
                dim=0,
            )
        output = self.linear(flat)[:rows]
        return output.reshape(*value.shape[:-1], output.shape[-1])


def _wrap_linear(parent: nn.Module, attribute: str, fixed_rows: int) -> bool:
    linear = getattr(parent, attribute)
    if isinstance(linear, FixedRowLinear):
        linear.fixed_rows = int(fixed_rows)
        return False
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"{type(parent).__name__}.{attribute} is not torch.nn.Linear"
        )
    setattr(parent, attribute, FixedRowLinear(linear, fixed_rows=fixed_rows))
    return True


def enable_qshape_fixed_row_linears(
    causal_lm: nn.Module,
    fixed_rows: int = 32,
    include_lm_head: bool = True,
) -> int:
    """Wrap Qwen decoder MLP projections and optionally its LM head."""

    fixed_rows = int(fixed_rows)
    if fixed_rows <= 0:
        raise ValueError("fixed_rows must be positive")
    decoder = getattr(getattr(causal_lm, "model", None), "layers", None)
    if decoder is None:
        raise ValueError("causal LM does not expose model.layers")
    wrapped = 0
    for layer in decoder:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError("decoder layer does not expose mlp")
        for attribute in ("gate_proj", "up_proj", "down_proj"):
            wrapped += int(_wrap_linear(mlp, attribute, fixed_rows))
    if include_lm_head:
        wrapped += int(_wrap_linear(causal_lm, "lm_head", fixed_rows))
    return wrapped


def enable_qshape_fixed_row_attention(
    causal_lm: nn.Module,
    fixed_rows: int = 32,
    fixed_key_block: int = 0,
) -> int:
    """Set fixed short-query rows on every decoder SDPA module."""

    fixed_rows = int(fixed_rows)
    if fixed_rows <= 0:
        raise ValueError("fixed_rows must be positive")
    fixed_key_block = int(fixed_key_block)
    if fixed_key_block < 0:
        raise ValueError("fixed_key_block must be non-negative")
    decoder = getattr(getattr(causal_lm, "model", None), "layers", None)
    if decoder is None:
        raise ValueError("causal LM does not expose model.layers")
    configured = 0
    for layer in decoder:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise ValueError("decoder layer does not expose self_attn")
        attention.qshape_fixed_rows = fixed_rows
        attention.qshape_fixed_key_block = fixed_key_block
        configured += 1
    return configured


def enable_qshape_exact_root_attention(
    causal_lm: nn.Module,
    max_rows: int = 32,
) -> int:
    """Recompute the packed-tree root SDPA row with its physical q_len=1 shape."""

    max_rows = int(max_rows)
    if max_rows <= 1:
        raise ValueError("max_rows must be greater than one")
    decoder = getattr(getattr(causal_lm, "model", None), "layers", None)
    if decoder is None:
        raise ValueError("causal LM does not expose model.layers")
    configured = 0
    for layer in decoder:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise ValueError("decoder layer does not expose self_attn")
        attention.qshape_exact_root_attention_rows = max_rows
        configured += 1
    return configured
