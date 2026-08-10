"""Q-length-stable BF16 linear kernel for small speculative trees.

The CUDA BLAS kernels selected by ``torch.nn.Linear`` can change their
reduction order when the number of input rows changes.  For greedy speculative
verification, those BF16 rounding differences can flip near-tied tokens.  This
Triton kernel keeps a fixed row tile and K-reduction schedule so the first row
uses the same arithmetic for q_len=1 and packed q_len<=the tree budget.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qshape_stable_linear_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    rows,
    output_features: tl.constexpr,
    input_features: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(axis=0)
    column_block = tl.program_id(axis=1)
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction_offsets = tl.arange(0, BLOCK_K)

    input_pointers = (
        input_pointer
        + row_offsets[:, None] * input_features
        + reduction_offsets[None, :]
    )
    weight_pointers = (
        weight_pointer
        + column_offsets[:, None] * input_features
        + reduction_offsets[None, :]
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for reduction_start in range(0, input_features, BLOCK_K):
        input_values = tl.load(
            input_pointers,
            mask=(row_offsets[:, None] < rows)
            & (reduction_start + reduction_offsets[None, :] < input_features),
            other=0.0,
        )
        weight_values = tl.load(
            weight_pointers,
            mask=(column_offsets[:, None] < output_features)
            & (reduction_start + reduction_offsets[None, :] < input_features),
            other=0.0,
        )
        accumulator += tl.dot(input_values, tl.trans(weight_values))
        input_pointers += BLOCK_K
        weight_pointers += BLOCK_K

    output_offsets = (
        row_offsets[:, None] * output_features + column_offsets[None, :]
    )
    tl.store(
        output_pointer + output_offsets,
        accumulator,
        mask=(row_offsets[:, None] < rows)
        & (column_offsets[None, :] < output_features),
    )


def qshape_stable_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Apply ``value @ weight.T`` with a q-length-invariant row tile."""

    if not value.is_cuda or not weight.is_cuda:
        raise ValueError("qshape_stable_linear requires CUDA tensors")
    if value.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("qshape_stable_linear expects BF16 or FP16 inputs")
    if value.dtype != weight.dtype:
        raise ValueError("input and weight dtypes must match")
    if weight.ndim != 2 or value.shape[-1] != weight.shape[1]:
        raise ValueError("incompatible input and weight shapes")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")

    original_shape = value.shape[:-1]
    flat_value = value.reshape(-1, value.shape[-1]).contiguous()
    contiguous_weight = weight.contiguous()
    rows = int(flat_value.shape[0])
    output_features = int(contiguous_weight.shape[0])
    input_features = int(contiguous_weight.shape[1])
    output = torch.empty(
        (rows, output_features),
        dtype=value.dtype,
        device=value.device,
    )
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(output_features, block_n),
    )
    _qshape_stable_linear_kernel[grid](
        flat_value,
        contiguous_weight,
        output,
        rows,
        output_features,
        input_features,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*original_shape, output_features)
