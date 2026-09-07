"""Evaluate Qwen EAGLE2 on the fixed NewDREAM benchmark manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMSPEC_ROOT = PROJECT_ROOT / "MMSpec"
if str(MMSPEC_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSPEC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from method.eagle2.kv_cache import initialize_past_key_values  # noqa: E402
from method.eagle2.spec_model import SpecModel  # noqa: E402
from method.eagle2.utils import prepare_logits_processor, reset_tree_mode  # noqa: E402
from new_dream.evaluation import eval_llava_fixed_multi as fixed_eval  # noqa: E402


REPO_MMSPEC_TEST = PROJECT_ROOT / "MMSpec" / "dataset" / "MMSpec" / "test"
if not fixed_eval.MMSPEC_ROOT.exists() and REPO_MMSPEC_TEST.exists():
    fixed_eval.MMSPEC_ROOT = REPO_MMSPEC_TEST


MappingLike = dict[str, Any]
EAGLE2_DEFAULT_DATASETS = [
    "MMT-Bench",
    "SEEDBench",
    "ScienceQA",
    "OCRBench",
    "ChartQA",
    "MathVista",
    "MMSpec",
    "TextVQA",
    "MME_Benchmark",
]


def write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summary_artifact_paths(dataset_dir: Path) -> MappingLike:
    result_path = str(dataset_dir / "results.jsonl")
    return {
        "result_path": result_path,
        "jsonl_path": result_path,
        "summary_path": str(dataset_dir / "summary.json"),
    }


def synchronize_if_cuda(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device_map(device_map: str, device: str):
    if device_map == "single":
        return {"": device}
    return device_map


def resolve_torch_dtype(value: str):
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def validate_attention_backend(value: str) -> None:
    if value == "flash_attention_2":
        raise ValueError(
            "flash_attention_2 is not implemented by the EAGLE2 custom "
            "Qwen2.5-VL manual-KV target; use sdpa or eager"
        )


def uses_manual_kv_cache(base_model) -> bool:
    config = getattr(base_model, "config", None)
    architectures = getattr(config, "architectures", []) or []
    return "Qwen2_5_VLForConditionalGeneration" in architectures


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    raw_ids = token_ids if isinstance(token_ids, (list, tuple, set)) else [token_ids]
    normalized = []
    for token_id in raw_ids:
        if token_id is None:
            continue
        if isinstance(token_id, torch.Tensor):
            if token_id.numel() == 1:
                token_id = token_id.item()
            else:
                normalized.extend(int(value) for value in token_id.view(-1).tolist())
                continue
        try:
            normalized.append(int(token_id))
        except (TypeError, ValueError):
            continue
    return normalized


def collect_stop_token_ids(tokenizer, model):
    stop_ids = set(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        stop_ids.update(_normalize_token_ids(getattr(generation_config, "eos_token_id", None)))
    config = getattr(model, "config", None)
    if config is not None:
        stop_ids.update(_normalize_token_ids(getattr(config, "eos_token_id", None)))
    stop_ids.update(_normalize_token_ids(getattr(tokenizer, "eod_id", None)))
    return stop_ids


def generated_tokens_exact_match(
    naive_ids: torch.Tensor | None,
    output_ids: torch.Tensor | None,
    input_len: int,
    stop_token_ids: set[int] | None = None,
) -> bool:
    if naive_ids is None or output_ids is None:
        return False
    naive_generated = canonical_generated_token_ids(naive_ids, input_len, stop_token_ids)
    output_generated = canonical_generated_token_ids(output_ids, input_len, stop_token_ids)
    return naive_generated == output_generated


def canonical_generated_token_ids(
    output_ids: torch.Tensor | None,
    input_len: int,
    stop_token_ids: set[int] | None = None,
) -> list[int]:
    if output_ids is None:
        return []
    generated = [int(value) for value in output_ids[0, input_len:].detach().cpu().tolist()]
    if stop_token_ids:
        for index, token_id in enumerate(generated):
            if token_id in stop_token_ids:
                return generated[: index + 1]
    return generated


def token_comparison_details(
    naive_ids: torch.Tensor | None,
    output_ids: torch.Tensor | None,
    input_len: int,
    stop_token_ids: set[int] | None = None,
) -> MappingLike:
    naive = canonical_generated_token_ids(naive_ids, input_len, stop_token_ids)
    eagle = canonical_generated_token_ids(output_ids, input_len, stop_token_ids)
    common_length = min(len(naive), len(eagle))
    mismatch_index = next(
        (index for index in range(common_length) if naive[index] != eagle[index]),
        None,
    )
    if mismatch_index is None and len(naive) != len(eagle):
        mismatch_index = common_length
    window_start = max(0, (mismatch_index or 0) - 4)
    window_end = (mismatch_index or 0) + 5
    return {
        "naive_effective_length": len(naive),
        "eagle_effective_length": len(eagle),
        "first_mismatch_index": mismatch_index,
        "naive_token_at_mismatch": (
            naive[mismatch_index] if mismatch_index is not None and mismatch_index < len(naive) else None
        ),
        "eagle_token_at_mismatch": (
            eagle[mismatch_index] if mismatch_index is not None and mismatch_index < len(eagle) else None
        ),
        "naive_token_window": naive[window_start:window_end],
        "eagle_token_window": eagle[window_start:window_end],
        "token_window_start": window_start,
    }


def add_exact_match_summary(summary: MappingLike, records: list[MappingLike]) -> MappingLike:
    result = dict(summary)
    compared = [
        bool(row["output_token_exact_match"])
        for row in records
        if not row.get("error") and "output_token_exact_match" in row
    ]
    mismatch_count = sum(1 for matched in compared if not matched)
    result.update(
        {
            "output_token_compared_count": len(compared),
            "output_token_mismatch_count": mismatch_count,
            "output_token_exact_match_ratio": (
                sum(1 for matched in compared if matched) / len(compared) if compared else None
            ),
            "output_token_match_definition": (
                "exact equality of generated token IDs after the prompt prefix and through "
                "the first stop token between EAGLE2 and the paired greedy custom-target baseline"
            ),
        }
    )
    return result


def resolve_manifest_path(
    manifest_dir: Path,
    dataset: str,
    seed: int,
    sample_num: int,
) -> Path:
    return manifest_dir / f"{fixed_eval.dataset_slug(dataset)}_seed{seed}_n{sample_num}.jsonl"


def parse_requested_datasets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(EAGLE2_DEFAULT_DATASETS)
    return fixed_eval.parse_datasets(value)


def _select_source_indices(dataset: str, base_dataset, indices: list[int]):
    if dataset == "MMSpec":
        return [base_dataset[index] for index in indices]
    return base_dataset.select(indices)


def load_or_create_fixed_manifest(dataset: str, base_dataset, args):
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_manifest_path(manifest_dir, dataset, args.seed, args.sample_num)

    if manifest_path.exists() and not args.regenerate_manifest:
        manifest = fixed_eval.read_jsonl(manifest_path)
        expected_count = min(args.sample_num, len(base_dataset))
        if len(manifest) != expected_count:
            raise ValueError(
                f"Manifest {manifest_path} has {len(manifest)} rows; expected {expected_count}"
            )
        indices = [int(row["source_index"]) for row in manifest]
        return _select_source_indices(dataset, base_dataset, indices), manifest, manifest_path

    indexed_dataset = (
        fixed_eval.add_fixed_index_to_rows(base_dataset) if dataset == "MMSpec" else base_dataset
    )
    shuffled = indexed_dataset.shuffle(seed=args.seed)
    selected = shuffled.select(range(min(args.sample_num, len(shuffled))))
    manifest = [
        fixed_eval.manifest_record(dataset, position, selected[position])
        for position in range(len(selected))
    ]
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    if dataset == "MMSpec":
        selected = [selected[position] for position in range(len(selected))]
    return selected, manifest, manifest_path


def decode_final(model: SpecModel, output_ids: torch.Tensor | None, input_len: int) -> str:
    if output_ids is None:
        return ""
    decode_ids = output_ids[0, input_len:].tolist()
    stop_ids = collect_stop_token_ids(model.tokenizer, model.base_model)
    stop_positions = [index for index, token_id in enumerate(decode_ids) if token_id in stop_ids]
    if stop_positions:
        decode_ids = decode_ids[: stop_positions[0] + 1]
    return model.tokenizer.decode(
        decode_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )


def _shared_manual_cache(model: SpecModel):
    reset_tree_mode(model)
    if not hasattr(model, "past_key_values"):
        (
            model.past_key_values,
            model.past_key_values_data,
            model.current_length_data,
        ) = initialize_past_key_values(model.base_model)
    model.current_length_data.zero_()
    return model.past_key_values


@torch.inference_mode()
def baseline_generate(model: SpecModel, inputs, args):
    device = model.base_model.device
    input_ids = inputs.input_ids.to(device)
    input_len = input_ids.shape[1]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    vision_kwargs = {}
    for key in ("pixel_values", "image_sizes", "image_grid_thw", "video_grid_thw"):
        value = inputs.get(key)
        if value is not None:
            vision_kwargs[key] = value.to(device) if hasattr(value, "to") else value

    logits_processor = None
    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=0,
        )

    stop_token_ids = collect_stop_token_ids(model.tokenizer, model.base_model)
    final_ids = input_ids
    generated = 0
    manual_kv_cache = uses_manual_kv_cache(model.base_model)
    past_key_values = _shared_manual_cache(model) if manual_kv_cache else None

    synchronize_if_cuda(device)
    start_time = time.perf_counter()
    outputs = model.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        **vision_kwargs,
    )
    if not manual_kv_cache:
        past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    while generated < args.max_new_token:
        if logits_processor is not None:
            next_logits = logits_processor(None, logits)
            next_token = torch.multinomial(torch.nn.functional.softmax(next_logits, dim=-1), 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        final_ids = torch.cat([final_ids, next_token], dim=-1)
        generated += 1
        if int(next_token.item()) in stop_token_ids:
            break

        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
        outputs = model.base_model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if not manual_kv_cache:
            past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    synchronize_if_cuda(device)
    return final_ids, generated, time.perf_counter() - start_time, input_len


@torch.inference_mode()
def run_one_sample(model: SpecModel, inputs, args):
    input_len = inputs.input_ids.shape[1]
    torch.manual_seed(args.seed)
    naive_ids, naive_steps, naive_time, _ = baseline_generate(model, inputs, args)

    torch.manual_seed(args.seed)
    synchronize_if_cuda(model.base_model.device)
    start_time = time.perf_counter()
    output_ids, new_tokens, _, acceptance_lengths = model.specgenerate(
        **inputs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_token,
        log=True,
        return_acceptance_len=True,
    )
    synchronize_if_cuda(model.base_model.device)
    spec_time = time.perf_counter() - start_time

    spec_steps = len(acceptance_lengths)
    avg_accept = sum(acceptance_lengths) / spec_steps if spec_steps else 0.0
    tokens_per_iteration = float(new_tokens) / float(spec_steps) if spec_steps else 0.0
    speedup = naive_time / spec_time if spec_time > 0 else 0.0
    stop_token_ids = collect_stop_token_ids(model.tokenizer, model.base_model)
    raw_token_match = generated_tokens_exact_match(naive_ids, output_ids, input_len)
    token_match = generated_tokens_exact_match(
        naive_ids,
        output_ids,
        input_len,
        stop_token_ids=stop_token_ids,
    )
    naive_effective_tokens = canonical_generated_token_ids(
        naive_ids,
        input_len,
        stop_token_ids,
    )
    output_effective_tokens = canonical_generated_token_ids(
        output_ids,
        input_len,
        stop_token_ids,
    )

    record = {
        "average_accept_length": f"{avg_accept:.2f}",
        "average_tokens_per_iteration": f"{tokens_per_iteration:.2f}",
        "speedup": speedup,
        "naive_time": naive_time,
        "spec_time": spec_time,
        "naive_step_count": int(naive_steps),
        "spec_iteration_count": int(spec_steps),
        "generated_token_count": int(new_tokens),
        "naive_generated_token_count": len(naive_effective_tokens),
        "effective_generated_token_count": len(output_effective_tokens),
        "output_token_raw_exact_match": raw_token_match,
        "output_token_exact_match": token_match,
    }
    if not token_match:
        record["output_token_mismatch_details"] = token_comparison_details(
            naive_ids,
            output_ids,
            input_len,
            stop_token_ids,
        )
    if not args.no_save_decoded_output:
        record["decoded_output"] = "" if args.skip_final_decode else decode_final(model, output_ids, input_len)
        record["naive_decoded_output"] = (
            "" if args.skip_final_decode else decode_final(model, naive_ids, input_len)
        )
    return record


def evaluate_dataset(model, dataset, selected, manifest, manifest_path, args, run_dir: Path):
    dataset_dir = run_dir / fixed_eval.dataset_slug(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = summary_artifact_paths(dataset_dir)
    result_path = Path(artifact_paths["result_path"])
    summary_path = Path(artifact_paths["summary_path"])
    if args.overwrite and result_path.exists():
        result_path.unlink()

    completed = {}
    if result_path.exists():
        for row in fixed_eval.read_jsonl(result_path):
            if not row.get("error"):
                completed[int(row["sample_position"])] = row

    records = []
    for sample_position in range(len(selected)):
        row_manifest = manifest[sample_position]
        if sample_position in completed:
            records.append(completed[sample_position])
            continue

        row = selected[sample_position]
        record = {
            "dataset": dataset,
            "question_id": sample_position,
            "sample_position": sample_position,
            "source_index": row_manifest["source_index"],
        }
        for key in (
            "idx",
            "pid",
            "id",
            "imgname",
            "dataset_name",
            "type",
            "category",
            "topic",
            "image_id",
            "image_path",
            "question_index",
        ):
            if key in row_manifest:
                record[key] = row_manifest[key]

        try:
            inputs = fixed_eval.build_inputs_for_record(model, dataset, row)
            record.update(run_one_sample(model, inputs, args))
        except Exception as exc:
            record["error"] = repr(exc)

        fixed_eval.append_jsonl(result_path, record)
        records.append(record)
        print(
            "record_metrics: "
            + json.dumps(
                {
                    "dataset": dataset,
                    "sample_position": sample_position,
                    "average_accept_length": record.get("average_accept_length"),
                    "average_tokens_per_iteration": record.get("average_tokens_per_iteration"),
                    "speedup": record.get("speedup"),
                    "output_token_exact_match": record.get("output_token_exact_match"),
                    "error": record.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if record.get("error") and args.stop_on_error:
            raise RuntimeError(record["error"])

    summary = add_exact_match_summary(fixed_eval.summarize_records(records), records)
    summary.update(
        {
            "dataset": dataset,
            **artifact_paths,
            "manifest_path": str(manifest_path),
            "sample_selection": f"fixed manifest, seed={args.seed}, first {args.sample_num}",
            "config": evaluator_config(args),
        }
    )
    if dataset == "MMSpec":
        write_json(dataset_dir / "by_topic_summary.json", fixed_eval.grouped_summaries(records, "topic"))
        write_json(
            dataset_dir / "by_category_summary.json",
            fixed_eval.grouped_summaries(records, "category"),
        )
        summary["by_topic_summary_path"] = str(dataset_dir / "by_topic_summary.json")
        summary["by_category_summary_path"] = str(dataset_dir / "by_category_summary.json")
    if dataset == "MME_Benchmark":
        write_json(
            dataset_dir / "by_category_summary.json",
            fixed_eval.grouped_summaries(records, "category"),
        )
        summary["by_category_summary_path"] = str(dataset_dir / "by_category_summary.json")

    write_json(summary_path, summary)
    return summary


def evaluator_config(args):
    return {
        "method": "eagle2",
        "base_model_path": args.base_model_path,
        "spec_model_path": args.spec_model_path,
        "top_k": args.top_k,
        "depth": args.depth,
        "total_token": args.total_token,
        "threshold": args.threshold,
        "max_new_token": args.max_new_token,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "compact_eval": args.quiet and args.skip_final_decode and args.no_save_decoded_output,
        "primary_speedup_policy": "newdream_global_sum_time_ratio",
        "baseline_policy": "paired greedy decoding with the same EAGLE2 custom target model",
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/mnt/data/wdy/spec_vlm/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--spec-model-path",
        type=str,
        default="Cloudriver/EAGLE-Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--max-new-token", type=int, default=200)
    parser.add_argument("--total-token", type=int, default=40)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", dest="top_k", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--device-map", type=str, default="single")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--sample-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/data/wdy/spec_vlm/outputs/eval/eagle2/qwen25vl7b-fixed-multi",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="/mnt/data/wdy/spec_vlm/result/manifests/dream_baseline_seed42",
    )
    parser.add_argument("--run-tag", type=str, default="qwen25vl-eagle2-fixed")
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-final-decode", action="store_true")
    parser.add_argument("--no-save-decoded-output", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_attention_backend(args.attn_implementation)
    datasets = parse_requested_datasets(args.datasets)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_tag={args.run_tag}")
    print(f"output_dir={run_dir}")
    print(f"manifest_dir={args.manifest_dir}")
    print(f"datasets={','.join(datasets)}")
    for key, value in evaluator_config(args).items():
        print(f"{key}={value}")
    print(f"sample_num={args.sample_num}")
    print(f"seed={args.seed}")
    print(f"device={args.device}")
    print(f"device_map={args.device_map}")
    print(f"warmup_samples={args.warmup_samples}")
    print(f"mmspec_root={fixed_eval.MMSPEC_ROOT}")

    selected_by_dataset = {}
    manifest_info = {}
    for dataset in datasets:
        base_dataset = fixed_eval.load_base_dataset(dataset)
        selected, manifest, manifest_path = load_or_create_fixed_manifest(dataset, base_dataset, args)
        selected_by_dataset[dataset] = (selected, manifest)
        manifest_info[dataset] = {
            "manifest_path": str(manifest_path),
            "selected_count": len(selected),
            "base_count": len(base_dataset),
        }
        print(
            f"dataset_ready dataset={dataset} base_count={len(base_dataset)} "
            f"selected_count={len(selected)} manifest={manifest_path}",
            flush=True,
        )

    if args.dry_run:
        write_json(run_dir / "manifest_summary.json", manifest_info)
        print("dry_run_complete")
        return

    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        device_map=resolve_device_map(args.device_map, args.device),
    )
    model.processor = fixed_eval.load_processor_for_model(args.base_model_path)
    model.eval()

    if args.warmup_samples > 0:
        warmup_dataset = datasets[0]
        warmup_selected, _ = selected_by_dataset[warmup_dataset]
        for warmup_index in range(min(args.warmup_samples, len(warmup_selected))):
            print(
                f"warmup_start dataset={warmup_dataset} sample_position={warmup_index}",
                flush=True,
            )
            inputs = fixed_eval.build_inputs_for_record(
                model,
                warmup_dataset,
                warmup_selected[warmup_index],
            )
            warmup_record = run_one_sample(model, inputs, args)
            print(
                "warmup_complete: "
                + json.dumps(
                    {
                        "output_token_exact_match": warmup_record["output_token_exact_match"],
                        "average_accept_length": warmup_record["average_accept_length"],
                    }
                ),
                flush=True,
            )

    all_summary = {
        "run_tag": args.run_tag,
        "output_dir": str(run_dir),
        "manifest_dir": args.manifest_dir,
        "datasets": {},
        "config": {
            **evaluator_config(args),
            "device_map": args.device_map,
            "sample_num": args.sample_num,
            "seed": args.seed,
            "device": args.device,
            "qwen_processor_min_pixels": fixed_eval.QWEN_MIN_PIXELS,
            "qwen_processor_max_pixels": fixed_eval.QWEN_MAX_PIXELS,
            "qwen_image_factor": fixed_eval.QWEN_IMAGE_FACTOR,
        },
    }

    for dataset in datasets:
        selected, manifest = selected_by_dataset[dataset]
        manifest_path = Path(manifest_info[dataset]["manifest_path"])
        print(f"start_dataset={dataset}", flush=True)
        summary = evaluate_dataset(model, dataset, selected, manifest, manifest_path, args, run_dir)
        all_summary["datasets"][dataset] = summary
        write_json(run_dir / "summary.json", all_summary)
        print(
            "dataset_summary: "
            + json.dumps(
                {
                    "dataset": dataset,
                    "num_records": summary.get("num_records"),
                    "num_error_records": summary.get("num_error_records"),
                    "avg_speedup": summary.get("avg_speedup"),
                    "avg_accept_length": summary.get("avg_accept_length"),
                    "avg_tokens_per_iteration": summary.get("avg_tokens_per_iteration"),
                    "output_token_exact_match_ratio": summary.get(
                        "output_token_exact_match_ratio"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_json(run_dir / "summary.json", all_summary)
    print(f"final_summary={run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
