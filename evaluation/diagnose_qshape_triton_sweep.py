"""Tune the q-length-stable Triton linear kernel on Qwen2.5-VL shapes."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from method.vispec.qshape_stable_linear import qshape_stable_linear


CONFIGS = [
    (16, 64, 32, 4, 3),
    (16, 64, 64, 4, 3),
    (16, 128, 32, 4, 3),
    (16, 128, 32, 8, 3),
    (16, 128, 64, 8, 3),
    (32, 64, 32, 4, 3),
    (32, 32, 32, 4, 3),
    (32, 32, 64, 4, 3),
    (32, 32, 128, 4, 3),
    (32, 64, 64, 2, 3),
    (32, 64, 64, 4, 2),
    (32, 64, 64, 4, 3),
    (32, 64, 64, 4, 4),
    (32, 64, 64, 4, 5),
    (32, 64, 64, 8, 3),
    (32, 64, 128, 4, 3),
    (32, 64, 128, 8, 3),
    (32, 128, 64, 4, 3),
    (32, 128, 32, 8, 3),
    (32, 128, 64, 8, 3),
    (64, 64, 32, 8, 3),
]


def _time(function, value, warmup=3, repeats=20):
    for _ in range(warmup):
        function(value)
    torch.cuda.synchronize(value.device)
    started = time.perf_counter()
    for _ in range(repeats):
        function(value)
    torch.cuda.synchronize(value.device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def _difference(reference, packed):
    difference = packed.float() - reference.float()
    return {
        "exact_equal": bool(torch.equal(reference, packed)),
        "max_abs": float(difference.abs().amax().item()),
        "different_element_ratio": float(
            difference.ne(0).float().mean().item()
        ),
    }


def _benchmark_shape(name, input_features, output_features, packed_length, device):
    generator = torch.Generator(device=device).manual_seed(
        input_features + output_features
    )
    root = torch.randn(
        (1, 1, input_features),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    tail = torch.randn(
        (1, packed_length - 1, input_features),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    packed = torch.cat([root, tail], dim=1)
    weight = torch.randn(
        (output_features, input_features),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    def native(value):
        return torch.nn.functional.linear(value, weight)

    with torch.inference_mode():
        native_reference = native(root)[0, 0]
        native_packed = native(packed)[0, 0]
        result = {
            "name": name,
            "input_features": input_features,
            "output_features": output_features,
            "native": {
                "difference": _difference(native_reference, native_packed),
                "q_len_1_ms": _time(native, root),
                "q_len_packed_ms": _time(native, packed),
            },
            "configs": [],
        }
        for block_m, block_n, block_k, num_warps, num_stages in CONFIGS:
            config = {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }

            def stable(value, config=config):
                return qshape_stable_linear(value, weight, **config)

            try:
                stable_reference = stable(root)[0, 0]
                stable_packed = stable(packed)[0, 0]
                config.update(
                    {
                        "difference": _difference(
                            stable_reference, stable_packed
                        ),
                        "native_q1_difference": _difference(
                            native_reference, stable_reference
                        ),
                        "q_len_1_ms": _time(stable, root),
                        "q_len_packed_ms": _time(stable, packed),
                    }
                )
            except Exception as error:
                config["error"] = repr(error)
            result["configs"].append(config)
    del root, tail, packed, weight
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--packed-length", type=int, default=31)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--intermediate-size", type=int, default=18944)
    parser.add_argument("--vocab-size", type=int, default=152064)
    args = parser.parse_args()

    device = torch.device(args.device)
    results = [
        _benchmark_shape(
            "gate_or_up",
            args.hidden_size,
            args.intermediate_size,
            args.packed_length,
            device,
        ),
        _benchmark_shape(
            "down",
            args.intermediate_size,
            args.hidden_size,
            args.packed_length,
            device,
        ),
        _benchmark_shape(
            "lm_head",
            args.hidden_size,
            args.vocab_size,
            args.packed_length,
            device,
        ),
    ]
    payload = {
        "device": str(device),
        "dtype": str(torch.bfloat16),
        "packed_length": args.packed_length,
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
