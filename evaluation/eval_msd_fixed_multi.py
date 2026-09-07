"""Evaluate Qwen MSD on the fixed NewDREAM benchmark manifests."""

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

from evaluation import eval_eagle2_fixed_multi as eagle_fixed  # noqa: E402
from method.msd.ea_model import EaModel  # noqa: E402
from method.msd.utils import temp_cache  # noqa: E402
from new_dream.evaluation import eval_llava_fixed_multi as fixed_eval  # noqa: E402


REPO_MMSPEC_TEST = PROJECT_ROOT / "MMSpec" / "dataset" / "MMSpec" / "test"
if not fixed_eval.MMSPEC_ROOT.exists() and REPO_MMSPEC_TEST.exists():
    fixed_eval.MMSPEC_ROOT = REPO_MMSPEC_TEST


MappingLike = dict[str, Any]
MSD_DEFAULT_DATASETS = list(eagle_fixed.EAGLE2_DEFAULT_DATASETS)


def write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_requested_datasets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MSD_DEFAULT_DATASETS)
    return fixed_eval.parse_datasets(value)


def max_scored_draft_tokens(top_k: int, depth: int) -> int:
    return top_k + depth * top_k * top_k


def validate_tree_budget(top_k: int, depth: int, total_token: int) -> None:
    if top_k <= 0 or depth < 0 or total_token <= 1:
        raise ValueError("top_k, depth, and total_token must define a non-empty MSD tree")
    score_pool = max_scored_draft_tokens(top_k, depth)
    draft_budget = total_token - 1
    if draft_budget > score_pool:
        raise ValueError(
            f"MSD draft budget {draft_budget} exceeds MSD score pool {score_pool} "
            f"for top_k={top_k}, depth={depth}"
        )


def validate_attention_backend(value: str) -> None:
    if value == "flash_attention_2":
        raise ValueError(
            "flash_attention_2 is not implemented by the MSD custom Qwen2.5-VL "
            "manual-KV target; use eager"
        )


def acceptance_stats_from_counter_delta(
    *,
    accept_before: float,
    iterations_before: int,
    accept_after: float,
    iterations_after: int,
    generated_tokens: int,
) -> MappingLike:
    spec_iterations = int(iterations_after) - int(iterations_before)
    accepted_tokens = float(accept_after) - float(accept_before)
    if spec_iterations <= 0:
        return {
            "spec_iteration_count": 0,
            "average_accept_length": 0.0,
            "average_tokens_per_iteration": 0.0,
        }
    return {
        "spec_iteration_count": spec_iterations,
        "average_accept_length": accepted_tokens / spec_iterations,
        "average_tokens_per_iteration": float(generated_tokens) / spec_iterations,
    }


def unwrap_loaded_model(loaded, base_model_path: str):
    if not isinstance(loaded, tuple):
        model = loaded
    elif len(loaded) == 4:
        _, model, _, _ = loaded
    elif len(loaded) == 2:
        model, _ = loaded
    else:
        model = loaded[0]
    model.processor = fixed_eval.load_processor_for_model(base_model_path)
    return model


def synchronize_if_cuda(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_msd_inputs(model: EaModel, inputs):
    device = model.base_model.device
    input_ids = inputs.input_ids.to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    attention_mask = inputs.get("attention_mask")
    inputs_embeds, _ = model.get_inputs_embeds(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    return input_ids, inputs_embeds


def _generation_max_length(model: EaModel, input_len: int, max_new_token: int) -> int:
    tree_tokens = int(getattr(model.ea_layer, "total_tokens", 0))
    return max(2048, input_len + max_new_token + tree_tokens + 10)


def _token_comparison_details(
    naive_ids: torch.Tensor,
    output_ids: torch.Tensor,
    input_len: int,
    stop_token_ids: set[int],
) -> MappingLike:
    details = eagle_fixed.token_comparison_details(
        naive_ids,
        output_ids,
        input_len,
        stop_token_ids,
    )
    details["msd_effective_length"] = details.pop("eagle_effective_length")
    details["msd_token_at_mismatch"] = details.pop("eagle_token_at_mismatch")
    details["msd_token_window"] = details.pop("eagle_token_window")
    return details


@torch.inference_mode()
def run_one_sample(model: EaModel, inputs, args):
    input_ids, inputs_embeds = _prepare_msd_inputs(model, inputs)
    input_len = int(input_ids.shape[1])
    max_length = _generation_max_length(model, input_len, args.max_new_token)

    torch.manual_seed(args.seed)
    temp_cache.use_msd = False
    synchronize_if_cuda(model.base_model.device)
    naive_start = time.perf_counter()
    naive_ids, naive_new_tokens, _ = model.naivegenerate(
        input_ids,
        inputs_embeds=inputs_embeds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_token,
        max_length=max_length,
        log=True,
    )
    synchronize_if_cuda(model.base_model.device)
    naive_time = time.perf_counter() - naive_start

    accept_before = float(model.acclen)
    iterations_before = int(model.accnum)
    torch.manual_seed(args.seed)
    temp_cache.use_msd = True
    synchronize_if_cuda(model.base_model.device)
    spec_start = time.perf_counter()
    output_ids, new_tokens, _ = model.msdgenerate(
        input_ids,
        inputs_embeds=inputs_embeds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_token,
        max_length=max_length,
        log=True,
    )
    synchronize_if_cuda(model.base_model.device)
    spec_time = time.perf_counter() - spec_start

    accept_stats = acceptance_stats_from_counter_delta(
        accept_before=accept_before,
        iterations_before=iterations_before,
        accept_after=float(model.acclen),
        iterations_after=int(model.accnum),
        generated_tokens=int(new_tokens),
    )
    stop_token_ids = eagle_fixed.collect_stop_token_ids(model.tokenizer, model.base_model)
    raw_token_match = eagle_fixed.generated_tokens_exact_match(
        naive_ids,
        output_ids,
        input_len,
    )
    token_match = eagle_fixed.generated_tokens_exact_match(
        naive_ids,
        output_ids,
        input_len,
        stop_token_ids,
    )
    naive_effective_tokens = eagle_fixed.canonical_generated_token_ids(
        naive_ids,
        input_len,
        stop_token_ids,
    )
    output_effective_tokens = eagle_fixed.canonical_generated_token_ids(
        output_ids,
        input_len,
        stop_token_ids,
    )

    record = {
        "average_accept_length": f"{accept_stats['average_accept_length']:.2f}",
        "average_tokens_per_iteration": f"{accept_stats['average_tokens_per_iteration']:.2f}",
        "speedup": naive_time / spec_time if spec_time > 0 else 0.0,
        "naive_time": naive_time,
        "spec_time": spec_time,
        "naive_step_count": int(naive_new_tokens),
        "spec_iteration_count": int(accept_stats["spec_iteration_count"]),
        "generated_token_count": int(new_tokens),
        "naive_generated_token_count": len(naive_effective_tokens),
        "effective_generated_token_count": len(output_effective_tokens),
        "output_token_raw_exact_match": raw_token_match,
        "output_token_exact_match": token_match,
    }
    if not token_match:
        record["output_token_mismatch_details"] = _token_comparison_details(
            naive_ids,
            output_ids,
            input_len,
            stop_token_ids,
        )
    if not args.no_save_decoded_output:
        record["decoded_output"] = (
            "" if args.skip_final_decode else eagle_fixed.decode_final(model, output_ids, input_len)
        )
        record["naive_decoded_output"] = (
            "" if args.skip_final_decode else eagle_fixed.decode_final(model, naive_ids, input_len)
        )
    return record


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
                "the first stop token between MSD and the paired naive MSD custom target"
            ),
        }
    )
    return result


def evaluator_config(args) -> MappingLike:
    return {
        "method": "msd",
        "base_model_path": args.base_model_path,
        "msd_model_path": args.msd_model_path,
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
        "baseline_policy": "paired naive decoding with the same MSD custom target model",
    }


def evaluate_dataset(model, dataset, selected, manifest, manifest_path, args, run_dir: Path):
    dataset_dir = run_dir / fixed_eval.dataset_slug(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = eagle_fixed.summary_artifact_paths(dataset_dir)
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


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/mnt/data/wdy/spec_vlm/models/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument(
        "--msd-model-path",
        type=str,
        default="Cloudriver/MSD-Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--max-new-token", type=int, default=200)
    parser.add_argument("--total-token", type=int, default=40)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", dest="top_k", type=int, default=4)
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
        default="/mnt/data/wdy/spec_vlm/outputs/eval/msd/qwen25vl7b-fixed-multi",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="/mnt/data/wdy/spec_vlm/result/manifests/dream_baseline_seed42",
    )
    parser.add_argument("--run-tag", type=str, default="qwen25vl-msd-fixed")
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
    validate_tree_budget(args.top_k, args.depth, args.total_token)
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
        selected, manifest, manifest_path = eagle_fixed.load_or_create_fixed_manifest(
            dataset,
            base_dataset,
            args,
        )
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

    temp_cache.use_msd = True
    loaded = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.msd_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
        torch_dtype=eagle_fixed.resolve_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        device_map=eagle_fixed.resolve_device_map(args.device_map, args.device),
    )
    model = unwrap_loaded_model(loaded, args.base_model_path)
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
