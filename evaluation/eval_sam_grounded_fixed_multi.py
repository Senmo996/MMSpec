"""Interleaved recycling evaluation on the project's fixed image benchmarks.

The data loading and seed-42 manifests are shared with the existing fixed
multi-dataset evaluator. Target-only and candidate decoding reuse one loaded
model and rotate execution order per sample to limit timing bias.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MMSPEC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MMSPEC_ROOT.parent
for path in (str(MMSPEC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.eval_sam_grounded_mmspec import (  # noqa: E402
    _parse_policies,
    _policy_kwargs,
    _token_hash,
)
from evaluation.summarize_training_free import summarize  # noqa: E402
from evaluation.selective_reuse_protocol import (  # noqa: E402
    apply_teacher_forced_pixel_probes,
    deterministic_cluster_split,
)
from evaluation.selective_reuse_content_ablation_protocol import (  # noqa: E402
    apply_teacher_forced_content_ablation_probes,
)
from evaluation.selective_reuse_counterfactual_bank_protocol import (  # noqa: E402
    apply_teacher_forced_counterfactual_bank_probes,
    build_matched_wrong_image_pairs,
    resize_wrong_image,
)
from evaluation.time_breakdown import build_time_breakdown_tracker  # noqa: E402
from evaluation.utils import (  # noqa: E402
    load_existing_ids,
    reorg_answer_file,
    save_result,
)
from method.sam_grounded.recycling_model import RecyclingSpecModel  # noqa: E402
from method.sam_grounded.spec_model import SpecModel  # noqa: E402
from method.sam_grounded.tree_recycling_model import (  # noqa: E402
    TreeRecyclingSpecModel,
)
from new_dream.evaluation.eval_llava_fixed_multi import (  # noqa: E402
    build_inputs_for_record,
    dataset_slug,
    load_base_dataset,
    load_image_for_record,
    load_or_create_manifest,
    load_processor_for_model,
    parse_datasets,
)


DEFAULT_FIXED_DATASETS = (
    "MMT-Bench,SEEDBench,ScienceQA,OCRBench,ChartQA,MathVista,"
    "TextVQA,MME_Benchmark"
)

DRAFT_MODEL_CLASSES = {
    "sam": SpecModel,
    "recycling": RecyclingSpecModel,
    "tree-recycling": TreeRecyclingSpecModel,
}


def _image_cluster_identity(dataset, row, manifest_row):
    """Return a stable image-level cluster key when metadata permits it."""

    for source in (manifest_row or {}, row):
        for key in ("image_id", "image_path", "imgname"):
            value = source.get(key)
            if value not in (None, ""):
                return f"{dataset}:{key}:{value}"
    value = row.get("image")
    if isinstance(value, str):
        return f"{dataset}:image:{hashlib.sha1(value.encode('utf-8')).hexdigest()}"
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            digest = hashlib.sha1(value["bytes"]).hexdigest()
            return f"{dataset}:bytes:{digest}"
        if value.get("path"):
            return f"{dataset}:path:{value['path']}"
    source_index = (manifest_row or {}).get(
        "source_index", row.get("_fixed_index", "unknown")
    )
    return f"{dataset}:source_index:{source_index}"


def _parse_sample_positions(value):
    positions = []
    for raw_position in str(value).split(","):
        raw_position = raw_position.strip()
        if not raw_position:
            continue
        position = int(raw_position)
        if position < 0:
            raise argparse.ArgumentTypeError(
                "sample positions must be non-negative"
            )
        positions.append(position)
    return tuple(sorted(set(positions)))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _policy_summary_row(policy, summary_path, payload):
    return {
        "policy": policy,
        "tokens_per_second": payload.get("tokens_per_second", 0.0),
        "avg_sample_speedup": payload.get("avg_sample_speedup", 0.0),
        "median_sample_speedup": payload.get("median_sample_speedup", 0.0),
        "sample_speedup_gt_1_ratio": payload.get(
            "sample_speedup_gt_1_ratio", 0.0
        ),
        "output_hash_match_ratio": payload.get("output_hash_match_ratio"),
        "avg_accept_length": payload.get("avg_accept_length", 0.0),
        "avg_tokens_per_iteration": payload.get(
            "avg_tokens_per_iteration", 0.0
        ),
        "avg_draft_accept_ratio": payload.get(
            "avg_draft_accept_ratio", 0.0
        ),
        "summary_path": str(Path(summary_path).resolve()),
    }


def _summarize_dataset(dataset, dataset_dir, policies, manifest_path):
    reference_path = dataset_dir / "target" / "results.jsonl"
    summaries = {}
    rows = []
    for policy in policies:
        result_path = dataset_dir / policy / "results.jsonl"
        summary_path = dataset_dir / policy / "summary.json"
        payload = summarize(result_path, reference_path)
        payload.update(
            {
                "dataset": dataset,
                "manifest_path": str(Path(manifest_path).resolve()),
            }
        )
        _write_json(summary_path, payload)
        summaries[policy] = payload
        rows.append(_policy_summary_row(policy, summary_path, payload))
    rows.sort(key=lambda row: row["avg_sample_speedup"], reverse=True)
    comparison = {"reference_policy": "target", "ranked_policies": rows}
    _write_json(dataset_dir / "comparison.json", comparison)
    return summaries


def _write_run_summary(output_root, dataset_results, candidate_policy, args):
    candidate_rows = [
        payloads[candidate_policy]
        for payloads in dataset_results.values()
        if candidate_policy in payloads
    ]
    turn_count = sum(int(row.get("num_turns", 0)) for row in candidate_rows)
    weighted_speedup = (
        sum(
            float(row.get("avg_sample_speedup", 0.0))
            * int(row.get("num_turns", 0))
            for row in candidate_rows
        )
        / turn_count
        if turn_count
        else 0.0
    )
    total_tokens = sum(int(row.get("total_new_tokens", 0)) for row in candidate_rows)
    total_time = sum(float(row.get("total_wall_time", 0.0)) for row in candidate_rows)
    payload = {
        "candidate_policy": candidate_policy,
        "completed_datasets": len(dataset_results),
        "expected_datasets": len(args.datasets),
        "sample_num_per_dataset": args.sample_num,
        "num_turns": turn_count,
        "total_new_tokens": total_tokens,
        "total_wall_time": total_time,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
        "weighted_avg_sample_speedup": weighted_speedup,
        "datasets": {
            dataset: {
                policy: {
                    "num_records": summary.get("num_records"),
                    "tokens_per_second": summary.get("tokens_per_second"),
                    "avg_sample_speedup": summary.get("avg_sample_speedup"),
                    "median_sample_speedup": summary.get(
                        "median_sample_speedup"
                    ),
                    "sample_speedup_gt_1_ratio": summary.get(
                        "sample_speedup_gt_1_ratio"
                    ),
                    "output_hash_match_ratio": summary.get(
                        "output_hash_match_ratio"
                    ),
                    "avg_accept_length": summary.get("avg_accept_length"),
                    "avg_tokens_per_iteration": summary.get(
                        "avg_tokens_per_iteration"
                    ),
                    "jsonl_path": summary.get("jsonl_path"),
                }
                for policy, summary in summaries.items()
            }
            for dataset, summaries in dataset_results.items()
        },
        "config": {
            "draft_engine": args.draft_engine,
            "base_model_path": args.base_model_path,
            "datasets": args.datasets,
            "sample_num": args.sample_num,
            "seed": args.seed,
            "max_new_token": args.max_new_token,
            "policies": args.policies,
            "matrix_top_k": args.matrix_top_k,
            "tree_node_budget": args.tree_node_budget,
            "attn_implementation": args.attn_implementation,
            "limit": args.limit,
            "policy_trace_positions": args.policy_trace_positions,
            "candidate_trace_diagnostics": (
                args.candidate_trace_diagnostics
            ),
            "selective_reuse_diagnostics": (
                args.selective_reuse_diagnostics
            ),
            "selective_reuse_probe_mode": args.selective_reuse_probe_mode,
            "visual_probe_batch_size": args.visual_probe_batch_size,
            "discovery_fraction": args.discovery_fraction,
            "split_seed": args.split_seed,
            "diagnostic_timing_valid": False,
        },
    }
    _write_json(Path(output_root) / "summary.json", payload)


def summarize_existing(args):
    """Rebuild summaries from completed JSONL files without loading a model."""

    output_root = Path(args.output_root)
    dataset_results = {}
    for dataset in args.datasets:
        slug = dataset_slug(dataset)
        dataset_dir = output_root / slug
        manifest_path = (
            Path(args.manifest_dir)
            / f"{slug}_seed{args.seed}_n{args.sample_num}.jsonl"
        )
        for policy in args.policies:
            result_path = dataset_dir / policy / "results.jsonl"
            if not result_path.is_file():
                raise FileNotFoundError(
                    f"missing completed result for summary-only mode: {result_path}"
                )
        summaries = _summarize_dataset(
            dataset, dataset_dir, args.policies, manifest_path
        )
        dataset_results[dataset] = summaries
    _write_run_summary(
        output_root, dataset_results, args.candidate_policy, args
    )
    print(
        f"Rebuilt summaries for {len(dataset_results)} datasets under "
        f"{output_root}",
        flush=True,
    )


@torch.inference_mode()
def evaluate(args):
    if args.disable_bf16_reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    model_kwargs = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = DRAFT_MODEL_CLASSES[args.draft_engine].from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=max(args.total_token, args.max_draft_tokens),
        **model_kwargs,
    )
    model.processor = load_processor_for_model(args.base_model_path)
    model.eval()
    tracker = build_time_breakdown_tracker(model)

    print(f"Loaded model: {args.base_model_path}")
    print(f"Datasets: {','.join(args.datasets)}")
    print(f"Policies: {','.join(args.policies)}")
    print(f"Samples per dataset: {args.sample_num}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"attention_implementation={args.attn_implementation}")
    print(f"policy_trace_enabled={not args.no_policy_trace}")
    print(f"policy_trace_positions={args.policy_trace_positions}")
    print(
        "candidate_trace_diagnostics="
        f"{args.candidate_trace_diagnostics}"
    )
    print(
        "selective_reuse_diagnostics="
        f"{args.selective_reuse_diagnostics}"
    )
    print(f"selective_reuse_probe_mode={args.selective_reuse_probe_mode}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_results = {}
    warmed_up = False

    for dataset in args.datasets:
        print(f"Loading dataset={dataset}", flush=True)
        base_dataset = load_base_dataset(dataset)
        selected, manifest, manifest_path = load_or_create_manifest(
            dataset, base_dataset, args
        )
        if len(manifest) != args.sample_num:
            raise ValueError(
                f"{dataset} manifest has {len(manifest)} rows; "
                f"expected exactly {args.sample_num}"
            )
        if args.limit is not None:
            limit = min(int(args.limit), len(selected))
            selected = selected.select(range(limit))
            manifest = manifest[:limit]
        print(
            f"dataset={dataset} selected_count={len(selected)} "
            f"manifest={manifest_path}",
            flush=True,
        )

        image_cluster_ids = [
            _image_cluster_identity(dataset, selected[index], manifest[index])
            for index in range(len(selected))
        ]
        wrong_image_pairs = None
        wrong_image_source_rows = base_dataset
        if (
            args.selective_reuse_diagnostics
            and args.selective_reuse_probe_mode
            == "teacher-forced-counterfactual-bank"
        ):
            if args.counterfactual_wrong_image_pool == "selected":
                wrong_image_source_rows = selected
            wrong_image_pairs = build_matched_wrong_image_pairs(
                selected,
                manifest,
                seed=args.split_seed,
                name=dataset,
                source_rows=wrong_image_source_rows,
            )
            fallback_ratio = sum(
                bool(pair["used_category_fallback"])
                for pair in wrong_image_pairs
            ) / len(wrong_image_pairs)
            print(
                f"dataset={dataset} wrong_image_category_fallback_ratio="
                f"{fallback_ratio:.6f}",
                flush=True,
            )
        analysis_splits = ["all"] * len(selected)
        if args.selective_reuse_diagnostics:
            analysis_splits = deterministic_cluster_split(
                image_cluster_ids,
                discovery_fraction=args.discovery_fraction,
                seed=args.split_seed,
                stratum=dataset,
            )
        evaluation_positions = sorted(
            range(len(selected)),
            key=lambda index: (
                0 if analysis_splits[index] == "discovery" else 1,
                index,
            ),
        )
        if args.selective_reuse_diagnostics:
            print(
                f"dataset={dataset} analysis_split_counts="
                + json.dumps(
                    {
                        split: analysis_splits.count(split)
                        for split in ("discovery", "heldout")
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        dataset_dir = output_root / dataset_slug(dataset)
        answer_files = {
            policy: dataset_dir / policy / "results.jsonl"
            for policy in args.policies
        }
        existing_ids = {
            policy: load_existing_ids(str(path))
            for policy, path in answer_files.items()
        }
        for path in answer_files.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not warmed_up and len(selected) > 0:
            inputs = build_inputs_for_record(model, dataset, selected[0])
            print("Warming up policies...", flush=True)
            for policy in args.policies:
                for warmup_index in range(args.warmup_runs):
                    torch.manual_seed(warmup_index)
                    model.specgenerate(
                        **inputs,
                        temperature=args.temperature,
                        max_new_tokens=min(
                            args.max_new_token, args.warmup_tokens
                        ),
                        log=True,
                        **_policy_kwargs(args, policy),
                    )
            reset_persistent_cache = getattr(
                model, "reset_persistent_recycling_cache", None
            )
            if callable(reset_persistent_cache):
                reset_persistent_cache()
                print("Cleared persistent caches after warmup", flush=True)
            warmed_up = True
            print("Warmup done", flush=True)

        progress = tqdm(
            evaluation_positions, desc=f"Evaluating {dataset} interleaved"
        )
        active_split = None
        for run_position, sample_position in enumerate(progress):
            row = selected[sample_position]
            manifest_row = manifest[sample_position]
            analysis_split = analysis_splits[sample_position]
            if analysis_split != active_split:
                reset_persistent_cache = getattr(
                    model, "reset_persistent_recycling_cache", None
                )
                if callable(reset_persistent_cache):
                    reset_persistent_cache()
                    print(
                        f"Cleared persistent caches before dataset={dataset} "
                        f"split={analysis_split}",
                        flush=True,
                    )
                active_split = analysis_split
            question_id = (
                f"{dataset_slug(dataset)}:{sample_position:04d}"
            )
            policy_order = list(args.policies)
            if args.rotate_policy_order:
                offset = run_position % len(policy_order)
                policy_order = policy_order[offset:] + policy_order[:offset]
            pending = [
                policy
                for policy in policy_order
                if question_id not in existing_ids[policy]
            ]
            if not pending:
                continue
            inputs = build_inputs_for_record(model, dataset, row)
            input_len = int(inputs.input_ids.shape[1])
            wrong_inputs = None
            wrong_image_pair = None
            if wrong_image_pairs is not None:
                wrong_image_pair = wrong_image_pairs[sample_position]
                wrong_row = wrong_image_source_rows[
                    wrong_image_pair["source_position"]
                ]
                target_image = load_image_for_record(dataset, row)
                wrong_image = resize_wrong_image(
                    load_image_for_record(dataset, wrong_row), target_image
                )
                wrong_inputs = build_inputs_for_record(
                    model,
                    dataset,
                    row,
                    image_override=wrong_image,
                )

            for policy in pending:
                collect_policy_trace = bool(
                    not args.no_policy_trace
                    and policy != "target"
                    and (
                        args.policy_trace_positions is None
                        or sample_position in args.policy_trace_positions
                    )
                )
                torch.manual_seed(0)
                torch.cuda.synchronize()
                tracker.reset()
                started = time.perf_counter()
                result = model.specgenerate(
                    **inputs,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_token,
                    log=True,
                    return_acceptance_len=True,
                    return_policy_trace=collect_policy_trace,
                    **_policy_kwargs(args, policy),
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                draft_time, target_time = tracker.snapshot()
                if collect_policy_trace:
                    output_ids, n_new, idx, accepted, trace = result
                else:
                    output_ids, n_new, idx, accepted = result
                    trace = []
                if (
                    trace
                    and args.selective_reuse_diagnostics
                    and args.selective_reuse_probe_mode
                    in (
                        "teacher-forced-pixel",
                        "teacher-forced-content-ablation",
                        "teacher-forced-counterfactual-bank",
                    )
                ):
                    if args.selective_reuse_probe_mode == "teacher-forced-pixel":
                        probe_metadata = apply_teacher_forced_pixel_probes(
                            model.base_model,
                            inputs,
                            output_ids,
                            prompt_length=input_len,
                            trace=trace,
                            num_regions=args.cover_num_probes,
                            top_k=args.matrix_top_k,
                            view_batch_size=args.visual_probe_batch_size,
                        )
                    elif (
                        args.selective_reuse_probe_mode
                        == "teacher-forced-content-ablation"
                    ):
                        probe_metadata = apply_teacher_forced_content_ablation_probes(
                            model.base_model,
                            inputs,
                            output_ids,
                            prompt_length=input_len,
                            trace=trace,
                            top_k=args.matrix_top_k,
                            view_batch_size=args.visual_probe_batch_size,
                        )
                    else:
                        if wrong_inputs is None or wrong_image_pair is None:
                            raise RuntimeError(
                                "counterfactual-bank probe lacks a wrong-image pair"
                            )
                        probe_metadata = (
                            apply_teacher_forced_counterfactual_bank_probes(
                                model.base_model,
                                inputs,
                                wrong_inputs,
                                output_ids,
                                prompt_length=input_len,
                                trace=trace,
                                wrong_image_pair=wrong_image_pair,
                                view_batch_size=args.visual_probe_batch_size,
                            )
                        )
                else:
                    probe_metadata = None
                choice = {
                    "index": 0,
                    "output_hashes": [_token_hash(output_ids, input_len)],
                    "idxs": [int(idx)],
                    "new_tokens": [int(n_new)],
                    "wall_time": [float(elapsed)],
                    "acceptance_length": [accepted],
                    "draft_time": [float(draft_time)],
                    "target_time": [float(target_time)],
                }
                if collect_policy_trace:
                    choice["policy_trace"] = [trace]
                    if probe_metadata is not None:
                        choice["selective_probe_metadata"] = [probe_metadata]
                if args.save_token_ids:
                    choice["output_token_ids"] = [
                        output_ids[0, input_len:].detach().cpu().tolist()
                    ]
                sample_metadata = {
                    "id": question_id,
                    "topic": dataset,
                    "source_index": int(manifest_row["source_index"]),
                    "category": manifest_row.get(
                        "category", manifest_row.get("type", "default")
                    ),
                    "image_cluster_id": image_cluster_ids[sample_position],
                    "analysis_split": analysis_split,
                    "analysis_split_seed": int(args.split_seed),
                }
                save_result(
                    str(answer_files[policy]),
                    sample_metadata,
                    f"{args.model_id}-{args.draft_engine}-{policy}",
                    [choice],
                )
                existing_ids[policy].add(question_id)

        for answer_file in answer_files.values():
            reorg_answer_file(str(answer_file))
        summaries = _summarize_dataset(
            dataset, dataset_dir, args.policies, manifest_path
        )
        dataset_results[dataset] = summaries
        _write_run_summary(
            output_root, dataset_results, args.candidate_policy, args
        )
        print(f"Completed dataset={dataset}", flush=True)
        del selected, manifest, base_dataset
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Results saved under {output_root}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Interleaved recycling evaluation on fixed image benchmarks"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--spec-model-path", default="")
    parser.add_argument("--model-id", default="training-free-grounded")
    parser.add_argument(
        "--draft-engine",
        choices=tuple(DRAFT_MODEL_CLASSES),
        default="tree-recycling",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--datasets", default=DEFAULT_FIXED_DATASETS)
    parser.add_argument("--sample-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild summaries from existing JSONL files without loading a model.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--policies",
        type=_parse_policies,
        default=_parse_policies(
            "target,context-score-trigram-deeper-wide-plus4"
        ),
    )
    parser.add_argument(
        "--candidate-policy",
        default="context-score-trigram-deeper-wide-plus4",
    )
    parser.add_argument("--total-token", type=int, default=64)
    parser.add_argument("--max-new-token", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-draft-tokens", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=40)
    parser.add_argument("--matrix-top-k", type=int, default=4)
    parser.add_argument("--tree-fixed-width", type=int, default=2)
    parser.add_argument("--tree-fixed-depth", type=int, default=4)
    parser.add_argument("--tree-broad-width", type=int, default=4)
    parser.add_argument("--tree-shallow-depth", type=int, default=3)
    parser.add_argument("--tree-node-budget", type=int, default=63)
    parser.add_argument("--cover-num-probes", type=int, default=4)
    parser.add_argument("--cover-anchor-fraction", type=float, default=0.5)
    parser.add_argument(
        "--cover-probe-visual-threshold", type=float, default=0.75
    )
    parser.add_argument("--cover-min-jsd", type=float, default=0.02)
    parser.add_argument("--visual-lexical-pool-size", type=int, default=64)
    parser.add_argument("--visual-lexical-width", type=int, default=4)
    parser.add_argument("--hst-token-weight", type=float, default=0.5)
    parser.add_argument("--hst-neighbors", type=int, default=16)
    parser.add_argument("--hst-temperature", type=float, default=0.05)
    parser.add_argument("--hst-transport-mode", default="delta_full")
    parser.add_argument("--hst-source-scope", default="all_text")
    parser.add_argument("--hst-visual-weight", type=float, default=0.0)
    parser.add_argument("--hst-min-confidence", type=float, default=0.39)
    parser.add_argument("--hst-online-update", type=int, default=1)
    parser.add_argument("--hst-trace-diagnostics", action="store_true")
    parser.add_argument(
        "--verification-trace-diagnostics", action="store_true"
    )
    parser.add_argument(
        "--verification-layer-diagnostics", action="store_true"
    )
    parser.add_argument(
        "--verification-margin-threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--verification-compact-path-repair", action="store_true"
    )
    parser.add_argument(
        "--verification-compact-root-margin-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument("--visual-threshold", type=float, default=0.55)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--visual-gamma", type=float, default=1.0)
    parser.add_argument("--visual-weight", type=float, default=0.7)
    parser.add_argument("--acceptance-weight", type=float, default=0.3)
    parser.add_argument("--acceptance-ema-decay", type=float, default=0.8)
    parser.add_argument("--grounding-layer", type=int, default=-1)
    parser.add_argument("--confidence-margin-scale", type=float, default=5.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--disable-bf16-reduced-precision-reduction", action="store_true"
    )
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument(
        "--no-policy-trace",
        action="store_true",
        help="Disable per-iteration policy traces for compact formal timing.",
    )
    parser.add_argument(
        "--policy-trace-positions",
        type=_parse_sample_positions,
        help=(
            "Optional comma-separated manifest positions to trace; target "
            "rows remain compact."
        ),
    )
    parser.add_argument(
        "--candidate-trace-diagnostics",
        action="store_true",
        help=(
            "Include candidate paths, source rows, and accepted token IDs in "
            "selected policy traces."
        ),
    )
    parser.add_argument(
        "--selective-reuse-diagnostics",
        action="store_true",
        help=(
            "Record four-region counterfactual visual sensitivity and "
            "U-only/GC-only shadow-tree outcomes without changing decoding."
        ),
    )
    parser.add_argument(
        "--selective-reuse-probe-mode",
        choices=(
            "packed-attention",
            "teacher-forced-pixel",
            "teacher-forced-content-ablation",
            "teacher-forced-counterfactual-bank",
        ),
        default="packed-attention",
    )
    parser.add_argument("--visual-probe-batch-size", type=int, default=1)
    parser.add_argument(
        "--counterfactual-wrong-image-pool",
        choices=("full", "selected"),
        default="full",
        help=(
            "Choose wrong-image controls from the full dataset or only the "
            "selected target set."
        ),
    )
    parser.add_argument("--discovery-fraction", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--enable-repeat-guard", action="store_true")
    parser.add_argument(
        "--no-rotate-policy-order",
        dest="rotate_policy_order",
        action="store_false",
    )
    parser.set_defaults(rotate_policy_order=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.datasets = parse_datasets(args.datasets)
    if "MMSpec" in args.datasets:
        parser.error("This entry point is for non-MMSpec fixed datasets")
    if len(args.policies) < 2 or "target" not in args.policies:
        parser.error("--policies must include target and at least one candidate")
    if args.candidate_policy not in args.policies:
        parser.error("--candidate-policy must be present in --policies")
    if args.sample_num <= 0:
        parser.error("--sample-num must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.no_policy_trace and args.policy_trace_positions is not None:
        parser.error("--policy-trace-positions conflicts with --no-policy-trace")
    if args.no_policy_trace and args.candidate_trace_diagnostics:
        parser.error(
            "--candidate-trace-diagnostics conflicts with --no-policy-trace"
        )
    if args.no_policy_trace and args.selective_reuse_diagnostics:
        parser.error(
            "--selective-reuse-diagnostics conflicts with --no-policy-trace"
        )
    if args.selective_reuse_diagnostics and args.draft_engine != "tree-recycling":
        parser.error(
            "--selective-reuse-diagnostics requires --draft-engine tree-recycling"
        )
    if args.visual_probe_batch_size <= 0:
        parser.error("--visual-probe-batch-size must be positive")
    if not 0.0 < args.discovery_fraction < 1.0:
        parser.error("--discovery-fraction must lie strictly between 0 and 1")
    if args.summary_only:
        summarize_existing(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
