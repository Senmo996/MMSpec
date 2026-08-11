"""Interleaved GWTR evaluation on the project's fixed image benchmarks.

The data loading and seed-42 manifests are shared with the existing fixed
multi-dataset evaluator.  Target-only and GWTR decoding reuse one loaded model
and rotate execution order per sample to limit timing bias.
"""

import argparse
import gc
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
from evaluation.time_breakdown import build_time_breakdown_tracker  # noqa: E402
from evaluation.utils import (  # noqa: E402
    load_existing_ids,
    reorg_answer_file,
    save_result,
)
from method.sam_grounded.tree_recycling_model import (  # noqa: E402
    TreeRecyclingSpecModel,
)
from new_dream.evaluation.eval_llava_fixed_multi import (  # noqa: E402
    build_inputs_for_record,
    dataset_slug,
    load_base_dataset,
    load_or_create_manifest,
    load_processor_for_model,
    parse_datasets,
)


DEFAULT_FIXED_DATASETS = (
    "MMT-Bench,SEEDBench,ScienceQA,OCRBench,ChartQA,MathVista,"
    "TextVQA,MME_Benchmark"
)


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
            "base_model_path": args.base_model_path,
            "datasets": args.datasets,
            "sample_num": args.sample_num,
            "seed": args.seed,
            "max_new_token": args.max_new_token,
            "policies": args.policies,
            "matrix_top_k": args.matrix_top_k,
            "tree_node_budget": args.tree_node_budget,
            "attn_implementation": args.attn_implementation,
        },
    }
    _write_json(Path(output_root) / "summary.json", payload)


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
    model = TreeRecyclingSpecModel.from_pretrained(
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
        del base_dataset
        gc.collect()

        if args.limit is not None:
            limit = min(int(args.limit), len(selected))
            selected = selected.select(range(limit))
            manifest = manifest[:limit]
        print(
            f"dataset={dataset} selected_count={len(selected)} "
            f"manifest={manifest_path}",
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
            warmed_up = True
            print("Warmup done", flush=True)

        progress = tqdm(
            range(len(selected)), desc=f"Evaluating {dataset} interleaved"
        )
        for sample_position in progress:
            row = selected[sample_position]
            manifest_row = manifest[sample_position]
            question_id = (
                f"{dataset_slug(dataset)}:{sample_position:04d}"
            )
            policy_order = list(args.policies)
            if args.rotate_policy_order:
                offset = sample_position % len(policy_order)
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

            for policy in pending:
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
                    return_policy_trace=True,
                    **_policy_kwargs(args, policy),
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                draft_time, target_time = tracker.snapshot()
                output_ids, n_new, idx, accepted, trace = result
                choice = {
                    "index": 0,
                    "output_hashes": [_token_hash(output_ids, input_len)],
                    "idxs": [int(idx)],
                    "new_tokens": [int(n_new)],
                    "wall_time": [float(elapsed)],
                    "acceptance_length": [accepted],
                    "policy_trace": [trace],
                    "draft_time": [float(draft_time)],
                    "target_time": [float(target_time)],
                }
                if args.save_token_ids:
                    choice["output_token_ids"] = [
                        output_ids[0, input_len:].detach().cpu().tolist()
                    ]
                sample_metadata = {
                    "id": question_id,
                    "topic": dataset,
                    "category": manifest_row.get(
                        "category", manifest_row.get("type", "default")
                    ),
                }
                save_result(
                    str(answer_files[policy]),
                    sample_metadata,
                    f"{args.model_id}-tree-recycling-{policy}",
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
        del selected, manifest
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Results saved under {output_root}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Interleaved GWTR evaluation on fixed image benchmarks"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--spec-model-path", default="")
    parser.add_argument("--model-id", default="training-free-grounded")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--datasets", default=DEFAULT_FIXED_DATASETS)
    parser.add_argument("--sample-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate-manifest", action="store_true")
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
    args.draft_engine = "tree-recycling"
    evaluate(args)


if __name__ == "__main__":
    main()
