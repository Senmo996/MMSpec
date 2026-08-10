"""Evaluate Lookahead decoding on the fixed NewDream/DREAM benchmark manifests."""

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

from method.lookahead.spec_model_lookahead import (  # noqa: E402
    SpecModel,
    _collect_stop_token_ids,
)
from method.vispec.kv_cache import initialize_past_key_values  # noqa: E402
from method.vispec.utils import prepare_logits_processor  # noqa: E402
from new_dream.evaluation import eval_llava_fixed_multi as fixed_eval  # noqa: E402


REPO_MMSPEC_TEST = PROJECT_ROOT / "MMSpec" / "dataset" / "MMSpec" / "test"
if not fixed_eval.MMSPEC_ROOT.exists() and REPO_MMSPEC_TEST.exists():
    fixed_eval.MMSPEC_ROOT = REPO_MMSPEC_TEST


def write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


MappingLike = dict[str, Any]


def synchronize_if_cuda(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device_map(device_map: str, device: str):
    if device_map == "single":
        return {"": device}
    return device_map


def uses_manual_kv_cache(base_model) -> bool:
    config = getattr(base_model, "config", None)
    architectures = getattr(config, "architectures", []) or []
    return "Qwen2_5_VLForConditionalGeneration" in architectures


def decode_final(model: SpecModel, output_ids: torch.Tensor | None, input_len: int) -> str:
    if output_ids is None:
        return ""
    decode_ids = output_ids[0, input_len:].tolist()
    eos_id = model.tokenizer.eos_token_id
    if eos_id in decode_ids:
        decode_ids = decode_ids[: decode_ids.index(eos_id) + 1]
    return model.tokenizer.decode(
        decode_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )


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

    stop_token_ids = _collect_stop_token_ids(model.tokenizer, model.base_model)
    final_ids = input_ids
    generated = 0
    past_key_values = None
    manual_kv_cache = uses_manual_kv_cache(model.base_model)
    if manual_kv_cache:
        past_key_values, _, _ = initialize_past_key_values(model.base_model)

    synchronize_if_cuda(device)
    start_time = time.time()

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
    return final_ids, generated, time.time() - start_time, input_len


@torch.inference_mode()
def run_one_sample(model: SpecModel, inputs, args):
    input_len = inputs.input_ids.shape[1]
    naive_ids, naive_steps, naive_time, _ = baseline_generate(model, inputs, args)

    synchronize_if_cuda(model.base_model.device)
    start_time = time.time()
    output_ids, new_tokens, spec_idx, acceptance_lengths = model.specgenerate(
        **inputs,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_token,
        log=True,
        return_acceptance_len=True,
        decoding_length=args.decoding_length,
        branch_length=args.branch_length,
    )
    synchronize_if_cuda(model.base_model.device)
    spec_time = time.time() - start_time

    spec_steps = int(spec_idx) + 1 if spec_idx is not None else len(acceptance_lengths)
    if spec_steps <= 0:
        spec_steps = len(acceptance_lengths)
    avg_accept = sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
    tokens_per_iteration = float(new_tokens) / float(spec_steps) if spec_steps else 0.0
    speedup = naive_time / spec_time if spec_time > 0 else 0.0

    record = {
        "average_accept_length": f"{avg_accept:.2f}",
        "average_tokens_per_iteration": f"{tokens_per_iteration:.2f}",
        "speedup": speedup,
        "naive_time": naive_time,
        "spec_time": spec_time,
        "naive_step_count": naive_steps,
        "spec_iteration_count": spec_steps,
        "generated_token_count": int(new_tokens),
    }
    if not args.no_save_decoded_output:
        record["decoded_output"] = "" if args.skip_final_decode else decode_final(model, output_ids, input_len)
        record["naive_decoded_output"] = "" if args.skip_final_decode else decode_final(model, naive_ids, input_len)
    return record


def evaluate_dataset(model, dataset, selected, manifest, manifest_path, args, run_dir: Path):
    dataset_dir = run_dir / fixed_eval.dataset_slug(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    result_path = dataset_dir / "results.jsonl"
    summary_path = dataset_dir / "summary.json"
    if args.overwrite and result_path.exists():
        result_path.unlink()

    completed = {}
    if result_path.exists():
        for row in fixed_eval.read_jsonl(result_path):
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
            if args.stop_on_error:
                fixed_eval.append_jsonl(result_path, record)
                raise

        fixed_eval.append_jsonl(result_path, record)
        records.append(record)
        if args.quiet:
            print(
                "record_metrics: "
                + json.dumps(
                    {
                        "dataset": dataset,
                        "sample_position": sample_position,
                        "average_accept_length": record.get("average_accept_length"),
                        "average_tokens_per_iteration": record.get("average_tokens_per_iteration"),
                        "speedup": record.get("speedup"),
                        "error": record.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            print("record: " + json.dumps(record, ensure_ascii=False), flush=True)

    summary = fixed_eval.summarize_records(records)
    summary.update(
        {
            "dataset": dataset,
            "result_path": str(result_path),
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
            "sample_selection": (
                "all samples in source order"
                if dataset == "MMSpec"
                else f"fixed manifest, seed={args.seed}, first {args.sample_num}"
            ),
            "config": {
                "method": "lookahead",
                "base_model_path": args.base_model_path,
                "decoding_length": args.decoding_length,
                "branch_length": args.branch_length,
                "max_new_token": args.max_new_token,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "torch_dtype": "auto",
                "compact_eval": args.quiet and args.skip_final_decode and args.no_save_decoded_output,
                "primary_speedup_policy": "newdream_global_sum_time_ratio",
            },
        }
    )
    if dataset == "MMSpec":
        by_topic = fixed_eval.grouped_summaries(records, "topic")
        by_category = fixed_eval.grouped_summaries(records, "category")
        write_json(dataset_dir / "by_topic_summary.json", by_topic)
        write_json(dataset_dir / "by_category_summary.json", by_category)
        summary["by_topic_summary_path"] = str(dataset_dir / "by_topic_summary.json")
        summary["by_category_summary_path"] = str(dataset_dir / "by_category_summary.json")
    if dataset == "MME_Benchmark":
        by_category = fixed_eval.grouped_summaries(records, "category")
        write_json(dataset_dir / "by_category_summary.json", by_category)
        summary["by_category_summary_path"] = str(dataset_dir / "by_category_summary.json")

    write_json(summary_path, summary)
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="/mnt/data/wdy/spec_vlm/models/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--max-new-token", type=int, default=200)
    parser.add_argument("--decoding-length", type=int, default=64)
    parser.add_argument("--branch-length", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--device-map", type=str, default="single")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--sample-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="/mnt/data/wdy/spec_vlm/result/lookahead_fixed_eval")
    parser.add_argument("--manifest-dir", type=str, default="/mnt/data/wdy/spec_vlm/result/manifests/dream_baseline_seed42")
    parser.add_argument("--run-tag", type=str, default="lookahead-fixed")
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
    datasets = fixed_eval.parse_datasets(args.datasets)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_tag={args.run_tag}")
    print(f"output_dir={run_dir}")
    print(f"manifest_dir={args.manifest_dir}")
    print(f"datasets={','.join(datasets)}")
    print(f"sample_num={args.sample_num}")
    print(f"device={args.device}")
    print(f"device_map={args.device_map}")
    print(f"decoding_length={args.decoding_length}")
    print(f"branch_length={args.branch_length}")
    print(f"max_new_token={args.max_new_token}")
    print(f"temperature={args.temperature}")
    print(f"top_p={args.top_p}")
    print(f"attn_implementation={args.attn_implementation}")
    print("primary_speedup_policy=newdream_global_sum_time_ratio")
    print(f"mmspec_root={fixed_eval.MMSPEC_ROOT}")

    selected_by_dataset = {}
    manifest_info = {}
    for dataset in datasets:
        base_dataset = fixed_eval.load_base_dataset(dataset)
        selected, manifest, manifest_path = fixed_eval.load_or_create_manifest(dataset, base_dataset, args)
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
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        device_map=resolve_device_map(args.device_map, args.device),
    )
    model.processor = fixed_eval.load_processor_for_model(args.base_model_path)
    model.eval()

    all_summary = {
        "run_tag": args.run_tag,
        "output_dir": str(run_dir),
        "manifest_dir": args.manifest_dir,
        "datasets": {},
        "config": {
            "method": "lookahead",
            "base_model_path": args.base_model_path,
            "decoding_length": args.decoding_length,
            "branch_length": args.branch_length,
            "max_new_token": args.max_new_token,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "attn_implementation": args.attn_implementation,
            "device_map": args.device_map,
            "sample_num": args.sample_num,
            "seed": args.seed,
            "device": args.device,
            "primary_speedup_policy": "newdream_global_sum_time_ratio",
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
                    "num_valid_records": summary.get("num_valid_records"),
                    "num_error_records": summary.get("num_error_records"),
                    "avg_speedup": summary.get("avg_speedup"),
                    "avg_accept_length": summary.get("avg_accept_length"),
                    "avg_tokens_per_iteration": summary.get("avg_tokens_per_iteration"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_json(run_dir / "summary.json", all_summary)
    print(f"final_summary={run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
