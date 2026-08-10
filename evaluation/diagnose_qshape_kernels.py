"""Localize BF16 q_len-dependent numerical differences in Qwen2.5-VL.

This diagnostic feeds an identical first row through q_len=1 and packed
q_len=N executions.  It measures language-layer projections independently and
then compares raw SDPA with its default and math-only backends.
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from method.vispec.modeling_qwen2_5_vl_kv import (
    Qwen2_5_VLForConditionalGeneration,
)
from method.vispec.qshape_stable_linear import qshape_stable_linear


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _difference(reference, packed):
    reference = reference.detach().float()
    packed = packed.detach().float()
    difference = packed - reference
    return {
        "exact_equal": bool(torch.equal(reference, packed)),
        "max_abs": float(difference.abs().amax().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rms": float(difference.square().mean().sqrt().item()),
        "different_element_ratio": float(
            difference.ne(0).float().mean().item()
        ),
    }


def _time_call(function, value, warmup, repeats):
    for _ in range(warmup):
        function(value)
    _synchronize(value.device)
    started = time.perf_counter()
    for _ in range(repeats):
        function(value)
    _synchronize(value.device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def _compare_callable(name, function, feature_size, packed_length, dtype, device):
    generator = torch.Generator(device=device).manual_seed(17 + feature_size)
    root = torch.randn(
        (1, 1, feature_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    tail = torch.randn(
        (1, packed_length - 1, feature_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    packed_input = torch.cat([root, tail], dim=1)
    with torch.inference_mode():
        reference = function(root)[0, 0]
        packed = function(packed_input)[0, 0]
        result = {
            "name": name,
            "input_features": feature_size,
            "output_features": int(reference.numel()),
            "difference": _difference(reference, packed),
            "q_len_1_ms": _time_call(function, root, warmup=2, repeats=10),
            "q_len_packed_ms": _time_call(
                function, packed_input, warmup=2, repeats=10
            ),
        }
    return result


def _compare_module(name, module, feature_size, packed_length, dtype, device):
    return _compare_callable(
        name,
        module,
        feature_size,
        packed_length,
        dtype,
        device,
    )


def _compare_cross_bf16_reduction(
    name, module, feature_size, packed_length, dtype, device
):
    generator = torch.Generator(device=device).manual_seed(17 + feature_size)
    root = torch.randn(
        (1, 1, feature_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    tail = torch.randn(
        (1, packed_length - 1, feature_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    packed_input = torch.cat([root, tail], dim=1)
    with torch.inference_mode(), _bf16_reduction(True):
        reference = module(root)[0, 0]
        q_len_1_ms = _time_call(module, root, warmup=2, repeats=10)
    with torch.inference_mode(), _bf16_reduction(False):
        packed = module(packed_input)[0, 0]
        q_len_packed_ms = _time_call(
            module, packed_input, warmup=2, repeats=10
        )
    return {
        "name": name,
        "input_features": feature_size,
        "output_features": int(reference.numel()),
        "reference_allow_bf16_reduced_precision_reduction": True,
        "packed_allow_bf16_reduced_precision_reduction": False,
        "difference": _difference(reference, packed),
        "q_len_1_ms": q_len_1_ms,
        "q_len_packed_ms": q_len_packed_ms,
    }


def _compare_two_callables(
    name,
    reference_function,
    packed_function,
    feature_size,
    packed_length,
    dtype,
    device,
):
    generator = torch.Generator(device=device).manual_seed(17 + feature_size)
    root = torch.randn(
        (1, 1, feature_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    tail = torch.randn(
        (1, packed_length - 1, feature_size),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    packed_input = torch.cat([root, tail], dim=1)
    with torch.inference_mode():
        reference = reference_function(root)[0, 0]
        packed = packed_function(packed_input)[0, 0]
        return {
            "name": name,
            "input_features": feature_size,
            "output_features": int(reference.numel()),
            "difference": _difference(reference, packed),
            "q_len_1_ms": _time_call(
                reference_function, root, warmup=2, repeats=10
            ),
            "q_len_packed_ms": _time_call(
                packed_function, packed_input, warmup=2, repeats=10
            ),
        }


@contextmanager
def _bf16_reduction(enabled):
    backend = torch.backends.cuda.matmul
    old_value = backend.allow_bf16_reduced_precision_reduction
    backend.allow_bf16_reduced_precision_reduction = enabled
    try:
        yield
    finally:
        backend.allow_bf16_reduced_precision_reduction = old_value


@contextmanager
def _blas_backend(name):
    preference = torch.backends.cuda.preferred_blas_library
    old_value = preference()
    preference(name)
    try:
        yield
    finally:
        preference(old_value)


def _batched_gemv(linear, value):
    """Apply a linear layer as independent M=1 batched GEMMs.

    The matrix shape seen by the CUDA kernel is invariant to q_len; only the
    batch count changes.  This is a diagnostic fallback, not yet a production
    implementation.
    """

    flat = value.reshape(-1, value.shape[-1])
    weight = linear.weight.transpose(0, 1).unsqueeze(0).expand(
        flat.shape[0], -1, -1
    )
    output = torch.bmm(flat.unsqueeze(1), weight).squeeze(1)
    if linear.bias is not None:
        output = output + linear.bias
    return output.reshape(*value.shape[:-1], linear.out_features)


def _tokenwise(function, value):
    return torch.cat(
        [function(value[:, index : index + 1]) for index in range(value.shape[1])],
        dim=1,
    )


def _fp32_linear_function(linear):
    weight = linear.weight.detach().float()
    bias = None if linear.bias is None else linear.bias.detach().float()

    def run(value):
        return F.linear(value.float(), weight, bias).to(value.dtype)

    return run


def _fixed_row_linear(linear, value, fixed_rows=32):
    flat = value.reshape(-1, value.shape[-1])
    if flat.shape[0] > fixed_rows:
        return linear(value)
    if flat.shape[0] < fixed_rows:
        flat = torch.cat(
            [
                flat,
                flat.new_zeros((fixed_rows - flat.shape[0], flat.shape[1])),
            ],
            dim=0,
        )
    output = linear(flat)[: value.numel() // value.shape[-1]]
    return output.reshape(*value.shape[:-1], output.shape[-1])


def _sdpa_context(math_only):
    if not math_only:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(backends=[SDPBackend.MATH])
    except ImportError:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )


def _compare_sdpa(
    name,
    num_heads,
    head_dim,
    prefix_length,
    packed_length,
    dtype,
    device,
    math_only=False,
):
    generator = torch.Generator(device=device).manual_seed(2718)
    root_q = torch.randn(
        (1, num_heads, 1, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    tail_q = torch.randn(
        (1, num_heads, packed_length - 1, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    query = torch.cat([root_q, tail_q], dim=2)
    prefix_k = torch.randn(
        (1, num_heads, prefix_length, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    prefix_v = torch.randn(
        (1, num_heads, prefix_length, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    root_k = torch.randn(
        (1, num_heads, 1, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    root_v = torch.randn(
        (1, num_heads, 1, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    tail_k = torch.randn(
        (1, num_heads, packed_length - 1, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    tail_v = torch.randn(
        (1, num_heads, packed_length - 1, head_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    single_k = torch.cat([prefix_k, root_k], dim=2)
    single_v = torch.cat([prefix_v, root_v], dim=2)
    packed_k = torch.cat([prefix_k, root_k, tail_k], dim=2)
    packed_v = torch.cat([prefix_v, root_v, tail_v], dim=2)
    single_mask = torch.zeros(
        (1, 1, 1, prefix_length + 1), dtype=dtype, device=device
    )
    minimum = torch.finfo(dtype).min
    packed_mask = torch.full(
        (1, 1, packed_length, prefix_length + packed_length),
        minimum,
        dtype=dtype,
        device=device,
    )
    packed_mask[:, :, :, :prefix_length] = 0
    for index in range(packed_length):
        packed_mask[:, :, index, prefix_length + index] = 0

    def run_single(unused):
        return F.scaled_dot_product_attention(
            root_q,
            single_k,
            single_v,
            attn_mask=single_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    def run_packed(unused):
        return F.scaled_dot_product_attention(
            query,
            packed_k,
            packed_v,
            attn_mask=packed_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    with torch.inference_mode(), _sdpa_context(math_only):
        reference = run_single(root_q)[0, :, 0]
        packed = run_packed(query)[0, :, 0]
        result = {
            "name": name,
            "prefix_length": prefix_length,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "math_only": math_only,
            "difference": _difference(reference, packed),
            "q_len_1_ms": _time_call(
                run_single, root_q, warmup=2, repeats=10
            ),
            "q_len_packed_ms": _time_call(
                run_packed, query, warmup=2, repeats=10
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--packed-length", type=int, default=31)
    parser.add_argument("--prefix-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
        attn_implementation="sdpa",
    ).eval()
    dtype = model.dtype
    layer = model.model.layers[0]
    attention = layer.self_attn
    mlp = layer.mlp
    hidden_size = int(model.config.hidden_size)
    intermediate_size = int(model.config.intermediate_size)

    modules = [
        ("input_layernorm", layer.input_layernorm, hidden_size),
        ("q_proj", attention.q_proj, hidden_size),
        ("k_proj", attention.k_proj, hidden_size),
        ("v_proj", attention.v_proj, hidden_size),
        ("o_proj", attention.o_proj, hidden_size),
        ("post_attention_layernorm", layer.post_attention_layernorm, hidden_size),
        ("gate_proj", mlp.gate_proj, hidden_size),
        ("up_proj", mlp.up_proj, hidden_size),
        ("down_proj", mlp.down_proj, intermediate_size),
        ("mlp", mlp, hidden_size),
        ("lm_head", model.lm_head, hidden_size),
    ]
    results = [
        _compare_module(
            name,
            module,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for name, module, feature_size in modules
    ]

    sensitive_modules = [
        ("gate_proj", mlp.gate_proj, hidden_size),
        ("up_proj", mlp.up_proj, hidden_size),
        ("down_proj", mlp.down_proj, intermediate_size),
        ("mlp", mlp, hidden_size),
        ("lm_head", model.lm_head, hidden_size),
    ]
    with _bf16_reduction(False):
        results.extend(
            _compare_module(
                f"{name}_bf16_full_reduction",
                module,
                feature_size,
                args.packed_length,
                dtype,
                device,
            )
            for name, module, feature_size in sensitive_modules
        )
    results.extend(
        _compare_cross_bf16_reduction(
            f"{name}_packed_bf16_full_vs_q1_native",
            module,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for name, module, feature_size in sensitive_modules
    )

    with _blas_backend("cublaslt"):
        results.extend(
            _compare_module(
                f"{name}_cublaslt",
                module,
                feature_size,
                args.packed_length,
                dtype,
                device,
            )
            for name, module, feature_size in sensitive_modules
        )

    with _blas_backend("cublaslt"), _bf16_reduction(False):
        results.extend(
            _compare_module(
                f"{name}_cublaslt_bf16_full_reduction",
                module,
                feature_size,
                args.packed_length,
                dtype,
                device,
            )
            for name, module, feature_size in sensitive_modules
        )

    gemv_functions = [
        (
            "gate_proj_batched_gemv",
            lambda value: _batched_gemv(mlp.gate_proj, value),
            hidden_size,
        ),
        (
            "up_proj_batched_gemv",
            lambda value: _batched_gemv(mlp.up_proj, value),
            hidden_size,
        ),
        (
            "down_proj_batched_gemv",
            lambda value: _batched_gemv(mlp.down_proj, value),
            intermediate_size,
        ),
        (
            "mlp_batched_gemv",
            lambda value: _batched_gemv(
                mlp.down_proj,
                mlp.act_fn(_batched_gemv(mlp.gate_proj, value))
                * _batched_gemv(mlp.up_proj, value),
            ),
            hidden_size,
        ),
        (
            "lm_head_batched_gemv",
            lambda value: _batched_gemv(model.lm_head, value),
            hidden_size,
        ),
    ]
    results.extend(
        _compare_callable(
            name,
            function,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for name, function, feature_size in gemv_functions
    )

    fp32_gate = _fp32_linear_function(mlp.gate_proj)
    fp32_up = _fp32_linear_function(mlp.up_proj)
    fp32_down = _fp32_linear_function(mlp.down_proj)
    fp32_lm_head = _fp32_linear_function(model.lm_head)

    def fp32_mlp(value):
        return fp32_down(mlp.act_fn(fp32_gate(value)) * fp32_up(value))

    fp32_functions = [
        ("gate_proj_fp32_accum_bf16_output", fp32_gate, hidden_size),
        ("up_proj_fp32_accum_bf16_output", fp32_up, hidden_size),
        (
            "down_proj_fp32_accum_bf16_output",
            fp32_down,
            intermediate_size,
        ),
        ("mlp_fp32_accum_bf16_output", fp32_mlp, hidden_size),
        ("lm_head_fp32_accum_bf16_output", fp32_lm_head, hidden_size),
        (
            "mlp_tokenwise_native",
            lambda value: _tokenwise(mlp, value),
            hidden_size,
        ),
        (
            "lm_head_tokenwise_native",
            lambda value: _tokenwise(model.lm_head, value),
            hidden_size,
        ),
    ]
    results.extend(
        _compare_callable(
            name,
            function,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for name, function, feature_size in fp32_functions
    )

    stable_gate = lambda value: qshape_stable_linear(
        value, mlp.gate_proj.weight, mlp.gate_proj.bias
    )
    stable_up = lambda value: qshape_stable_linear(
        value, mlp.up_proj.weight, mlp.up_proj.bias
    )
    stable_down = lambda value: qshape_stable_linear(
        value, mlp.down_proj.weight, mlp.down_proj.bias
    )
    stable_lm_head = lambda value: qshape_stable_linear(
        value, model.lm_head.weight, model.lm_head.bias
    )

    def stable_mlp(value):
        return stable_down(mlp.act_fn(stable_gate(value)) * stable_up(value))

    stable_functions = [
        ("gate_proj_qshape_stable", stable_gate, hidden_size),
        ("up_proj_qshape_stable", stable_up, hidden_size),
        ("down_proj_qshape_stable", stable_down, intermediate_size),
        ("mlp_qshape_stable", stable_mlp, hidden_size),
        ("lm_head_qshape_stable", stable_lm_head, hidden_size),
    ]
    results.extend(
        _compare_callable(
            name,
            function,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for name, function, feature_size in stable_functions
    )
    results.extend(
        _compare_two_callables(
            f"{name}_packed_vs_q1_native",
            module,
            function,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for (name, function, feature_size), (_, module, _) in zip(
            stable_functions,
            sensitive_modules,
        )
    )

    padded_gate = lambda value: _fixed_row_linear(mlp.gate_proj, value)
    padded_up = lambda value: _fixed_row_linear(mlp.up_proj, value)
    padded_down = lambda value: _fixed_row_linear(mlp.down_proj, value)
    padded_lm_head = lambda value: _fixed_row_linear(model.lm_head, value)

    def padded_mlp(value):
        return padded_down(mlp.act_fn(padded_gate(value)) * padded_up(value))

    padded_functions = [
        ("gate_proj_fixed_row32", padded_gate, hidden_size),
        ("up_proj_fixed_row32", padded_up, hidden_size),
        ("down_proj_fixed_row32", padded_down, intermediate_size),
        ("mlp_fixed_row32", padded_mlp, hidden_size),
        ("lm_head_fixed_row32", padded_lm_head, hidden_size),
    ]
    results.extend(
        _compare_callable(
            name,
            function,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for name, function, feature_size in padded_functions
    )
    results.extend(
        _compare_two_callables(
            f"{name}_packed_vs_q1_native",
            module,
            function,
            feature_size,
            args.packed_length,
            dtype,
            device,
        )
        for (name, function, feature_size), (_, module, _) in zip(
            padded_functions,
            sensitive_modules,
        )
    )
    results.extend(
        [
            _compare_sdpa(
                "sdpa_default",
                int(model.config.num_attention_heads),
                hidden_size // int(model.config.num_attention_heads),
                args.prefix_length,
                args.packed_length,
                dtype,
                device,
                math_only=False,
            ),
            _compare_sdpa(
                "sdpa_math",
                int(model.config.num_attention_heads),
                hidden_size // int(model.config.num_attention_heads),
                args.prefix_length,
                args.packed_length,
                dtype,
                device,
                math_only=True,
            ),
        ]
    )
    payload = {
        "model_path": str(Path(args.model_path).resolve()),
        "dtype": str(dtype),
        "device": str(device),
        "packed_length": args.packed_length,
        "prefix_length": args.prefix_length,
        "initial_blas_backend": str(
            torch.backends.cuda.preferred_blas_library()
        ),
        "initial_allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
