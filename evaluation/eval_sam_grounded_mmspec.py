"""Interleaved evaluation for training-free multimodal SAM policies.

The same loaded target model is reused for all policies, and policy order is
rotated per sample to reduce timing bias from transient GPU load.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.time_breakdown import build_time_breakdown_tracker
from evaluation.utils import (
    build_prompt,
    get_common_args,
    get_num_turns,
    load_existing_ids,
    load_mmspec_data,
    process_output,
    reorg_answer_file,
    run_sanity_check,
    save_result,
)
from method.sam_grounded.spec_model import SpecModel
from method.sam_grounded.recycling_model import RecyclingSpecModel
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel


def _token_hash(output_ids: torch.Tensor, input_len: int) -> str:
    token_ids = output_ids[0, input_len:].detach().cpu().tolist()
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_policies(value: str):
    policies = []
    for policy in value.split(","):
        policy = policy.strip()
        if policy and policy not in policies:
            policies.append(policy)
    # The policy registry lives in the controller rather than the base model.
    from method.sam_grounded.controller import GroundedDraftController

    normalized = {
        policy[len("cover-") :] if policy.startswith("cover-") else policy
        for policy in policies
    }
    unknown = normalized - GroundedDraftController.POLICIES
    if not policies or unknown:
        raise argparse.ArgumentTypeError(
            f"Invalid policies {sorted(unknown)}; choose from "
            f"{sorted(GroundedDraftController.POLICIES)}"
        )
    return policies


def _select_topic_indices(data, samples_per_topic: int, topic_offset: int = 0):
    if samples_per_topic < 0:
        raise ValueError("samples_per_topic must be non-negative")
    if topic_offset < 0:
        raise ValueError("topic_offset must be non-negative")
    topic_counts = {}
    selected_indices = []
    stop = topic_offset + samples_per_topic
    for index, sample in enumerate(data):
        topic = sample.get("topic", "unknown")
        count = topic_counts.get(topic, 0)
        if topic_offset <= count < stop:
            selected_indices.append(index)
        topic_counts[topic] = count + 1
    return selected_indices


def _policy_kwargs(args, policy):
    values = {
        "draft_policy": policy,
        "min_draft_tokens": args.min_draft_tokens,
        "max_draft_tokens": args.max_draft_tokens,
        "visual_threshold": args.visual_threshold,
        "confidence_threshold": args.confidence_threshold,
        "visual_gamma": args.visual_gamma,
        "visual_weight": args.visual_weight,
        "acceptance_weight": args.acceptance_weight,
        "acceptance_ema_decay": args.acceptance_ema_decay,
        "grounding_layer": args.grounding_layer,
        "confidence_margin_scale": args.confidence_margin_scale,
        "disable_repeat_guard": not args.enable_repeat_guard,
    }
    if args.draft_engine in ("recycling", "tree-recycling"):
        values["matrix_top_k"] = args.matrix_top_k
    if args.draft_engine == "tree-recycling":
        values.update(
            {
                "tree_fixed_width": args.tree_fixed_width,
                "tree_fixed_depth": args.tree_fixed_depth,
                "tree_broad_width": args.tree_broad_width,
                "tree_shallow_depth": args.tree_shallow_depth,
                "tree_node_budget": args.tree_node_budget,
                "cover_num_probes": args.cover_num_probes,
                "cover_anchor_fraction": args.cover_anchor_fraction,
                "cover_probe_visual_threshold": args.cover_probe_visual_threshold,
                "cover_min_jsd": args.cover_min_jsd,
                "visual_lexical_pool_size": args.visual_lexical_pool_size,
                "visual_lexical_width": args.visual_lexical_width,
                "hst_token_weight": args.hst_token_weight,
                "hst_neighbors": args.hst_neighbors,
                "hst_temperature": args.hst_temperature,
                "hst_transport_mode": args.hst_transport_mode,
                "hst_source_scope": args.hst_source_scope,
                "hst_visual_weight": args.hst_visual_weight,
                "hst_min_confidence": args.hst_min_confidence,
                "hst_online_update": bool(args.hst_online_update),
                "hst_trace_diagnostics": args.hst_trace_diagnostics,
                "verification_trace_diagnostics": (
                    args.verification_trace_diagnostics
                ),
                "verification_layer_diagnostics": (
                    args.verification_layer_diagnostics
                ),
                "verification_margin_threshold": (
                    args.verification_margin_threshold
                ),
                "verification_compact_path_repair": (
                    args.verification_compact_path_repair
                ),
                "verification_compact_root_margin_threshold": (
                    args.verification_compact_root_margin_threshold
                ),
            }
        )
    return values


@torch.inference_mode()
def evaluate(args):
    if args.disable_bf16_reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model_load_kwargs = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_load_kwargs["attn_implementation"] = args.attn_implementation

    model_classes = {
        "sam": SpecModel,
        "recycling": RecyclingSpecModel,
        "tree-recycling": TreeRecyclingSpecModel,
    }
    model_class = model_classes[args.draft_engine]
    model = model_class.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=max(args.total_token, args.max_draft_tokens),
        **model_load_kwargs,
    )
    qshape_wrapped_linears = 0
    if args.qshape_fixed_rows > 0:
        from method.vispec.qshape_fixed_row import (
            enable_qshape_fixed_row_linears,
        )

        qshape_wrapped_linears = enable_qshape_fixed_row_linears(
            model.base_model,
            fixed_rows=args.qshape_fixed_rows,
            include_lm_head=True,
        )
    qshape_attention_layers = 0
    if args.qshape_attention_fixed_rows > 0:
        from method.vispec.qshape_fixed_row import (
            enable_qshape_fixed_row_attention,
        )

        qshape_attention_layers = enable_qshape_fixed_row_attention(
            model.base_model,
            fixed_rows=args.qshape_attention_fixed_rows,
            fixed_key_block=args.qshape_attention_key_block,
        )
    qshape_exact_root_attention_layers = 0
    if args.qshape_exact_root_attention_rows > 1:
        from method.vispec.qshape_fixed_row import (
            enable_qshape_exact_root_attention,
        )

        qshape_exact_root_attention_layers = (
            enable_qshape_exact_root_attention(
                model.base_model,
                max_rows=args.qshape_exact_root_attention_rows,
            )
        )
    tokenizer = model.get_tokenizer()
    model.eval()
    tracker = build_time_breakdown_tracker(model)

    data = load_mmspec_data(args.data_folder)
    if args.samples_per_topic is not None:
        selected_indices = _select_topic_indices(
            data,
            args.samples_per_topic,
            args.topic_offset,
        )
        data = data.select(selected_indices)
    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))
    print(f"Loaded {len(data)} samples")
    print(f"Policies: {','.join(args.policies)}")
    print(f"Draft engine: {args.draft_engine}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"attention_implementation={args.attn_implementation}")
    print(
        "allow_bf16_reduced_precision_reduction="
        f"{torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}"
    )
    print(f"qshape_fixed_rows={args.qshape_fixed_rows}")
    print(f"qshape_wrapped_linears={qshape_wrapped_linears}")
    print(
        f"qshape_attention_fixed_rows={args.qshape_attention_fixed_rows}"
    )
    print(f"qshape_attention_key_block={args.qshape_attention_key_block}")
    print(f"qshape_attention_layers={qshape_attention_layers}")
    print(
        "qshape_exact_root_attention_rows="
        f"{args.qshape_exact_root_attention_rows}"
    )
    print(
        "qshape_exact_root_attention_layers="
        f"{qshape_exact_root_attention_layers}"
    )
    print(f"policy_trace_enabled={not args.no_policy_trace}")

    if args.sanity:
        run_sanity_check(args, model, tokenizer, data)
        return

    answer_files = {}
    existing_ids = {}
    for policy in args.policies:
        answer_path = Path(args.output_root) / policy / "results.jsonl"
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        answer_files[policy] = str(answer_path)
        existing_ids[policy] = load_existing_ids(str(answer_path))
        print(
            f"policy={policy} output={answer_path} "
            f"resume_records={len(existing_ids[policy])}"
        )

    print("Warming up policies...")
    for policy in args.policies:
        for warmup_idx in range(args.warmup_runs):
            torch.manual_seed(warmup_idx)
            model_inputs = build_prompt(data[0], args)
            model.specgenerate(
                **model_inputs,
                temperature=args.temperature,
                max_new_tokens=min(args.max_new_token, args.warmup_tokens),
                log=True,
                **_policy_kwargs(args, policy),
            )
    reset_persistent_cache = getattr(
        model, "reset_persistent_recycling_cache", None
    )
    if callable(reset_persistent_cache):
        reset_persistent_cache()
    print("Warmup done")

    for sample_index, sample in enumerate(tqdm(data, desc="Evaluating interleaved policies")):
        if args.rotate_policy_order:
            offset = sample_index % len(args.policies)
            policy_order = args.policies[offset:] + args.policies[:offset]
        else:
            policy_order = args.policies

        for policy in policy_order:
            if sample["id"] in existing_ids[policy]:
                continue
            choices = []
            num_turns = get_num_turns(sample)

            for choice_index in range(args.num_choices):
                torch.manual_seed(choice_index)
                conversation_history = []
                decoded_turns = []
                output_hashes = []
                output_token_ids = []
                idxs = []
                new_tokens = []
                wall_times = []
                acceptance_lengths = []
                policy_traces = []
                draft_times = []
                target_times = []

                for turn_index in range(num_turns):
                    model_inputs = build_prompt(
                        sample,
                        args,
                        turn_idx=turn_index,
                        conversation_history=(
                            conversation_history if turn_index > 0 else None
                        ),
                    )
                    input_len = int(model_inputs["input_ids"].shape[1])
                    torch.cuda.synchronize()
                    tracker.reset()
                    started = time.perf_counter()
                    result = model.specgenerate(
                        **model_inputs,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_token,
                        log=True,
                        return_acceptance_len=True,
                        return_policy_trace=not args.no_policy_trace,
                        **_policy_kwargs(args, policy),
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    draft_time, target_time = tracker.snapshot()

                    if args.no_policy_trace:
                        output_ids, n_new, idx, accepted = result
                        trace = []
                    else:
                        output_ids, n_new, idx, accepted, trace = result
                    output_hashes.append(_token_hash(output_ids, input_len))
                    if args.save_token_ids:
                        output_token_ids.append(
                            output_ids[0, input_len:].detach().cpu().tolist()
                        )
                    needs_decode = args.save_decoded_output or num_turns > 1
                    decoded = (
                        process_output(output_ids, tokenizer, input_len)
                        if needs_decode
                        else ""
                    )
                    if args.save_decoded_output:
                        decoded_turns.append(decoded)

                    turns = sample.get("turns", [sample.get("prompt", "")])
                    user_message = (
                        turns[turn_index] if turn_index < len(turns) else turns[0]
                    )
                    conversation_history.append((user_message, decoded))

                    idxs.append(int(idx))
                    new_tokens.append(int(n_new))
                    wall_times.append(float(elapsed))
                    acceptance_lengths.append(accepted)
                    if not args.no_policy_trace:
                        policy_traces.append(trace)
                    draft_times.append(float(draft_time))
                    target_times.append(float(target_time))

                choice = {
                    "index": choice_index,
                    "output_hashes": output_hashes,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_times,
                    "acceptance_length": acceptance_lengths,
                    "draft_time": draft_times,
                    "target_time": target_times,
                }
                if not args.no_policy_trace:
                    choice["policy_trace"] = policy_traces
                if args.save_decoded_output:
                    choice["turns"] = decoded_turns
                if args.save_token_ids:
                    choice["output_token_ids"] = output_token_ids
                choices.append(choice)

            save_result(
                answer_files[policy],
                sample,
                f"{args.model_id}-{args.draft_engine}-{policy}",
                choices,
            )

    for answer_file in answer_files.values():
        reorg_answer_file(answer_file)
    print(f"Results saved under {args.output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Interleaved training-free multimodal SAM evaluation"
    )
    parser = get_common_args(parser)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--draft-engine",
        choices=["sam", "recycling", "tree-recycling"],
        default="sam",
    )
    parser.add_argument(
        "--policies",
        type=_parse_policies,
        default=_parse_policies(
            "target,fixed,short,visual-hard,visual-anchor,visual-soft,visual-accept"
        ),
    )
    parser.add_argument("--total-token", type=int, default=40)
    parser.add_argument("--min-draft-tokens", type=int, default=2)
    parser.add_argument("--max-draft-tokens", type=int, default=40)
    parser.add_argument("--matrix-top-k", type=int, default=4)
    parser.add_argument("--tree-fixed-width", type=int, default=2)
    parser.add_argument("--tree-fixed-depth", type=int, default=4)
    parser.add_argument("--tree-broad-width", type=int, default=4)
    parser.add_argument("--tree-shallow-depth", type=int, default=2)
    parser.add_argument("--tree-node-budget", type=int, default=31)
    parser.add_argument("--cover-num-probes", type=int, default=4)
    parser.add_argument("--cover-anchor-fraction", type=float, default=0.5)
    parser.add_argument("--cover-probe-visual-threshold", type=float, default=0.75)
    parser.add_argument("--cover-min-jsd", type=float, default=0.02)
    parser.add_argument("--visual-lexical-pool-size", type=int, default=64)
    parser.add_argument("--visual-lexical-width", type=int, default=4)
    parser.add_argument("--hst-token-weight", type=float, default=0.5)
    parser.add_argument("--hst-neighbors", type=int, default=16)
    parser.add_argument("--hst-temperature", type=float, default=0.05)
    parser.add_argument(
        "--hst-transport-mode",
        choices=["delta_half", "delta_full", "post_state"],
        default="delta_full",
    )
    parser.add_argument(
        "--hst-source-scope",
        choices=["all_text", "post_visual_text"],
        default="all_text",
    )
    parser.add_argument("--hst-visual-weight", type=float, default=0.0)
    parser.add_argument("--hst-min-confidence", type=float, default=0.39)
    parser.add_argument(
        "--hst-online-update", type=int, choices=[0, 1], default=1
    )
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
        "--verification-compact-path-repair",
        action="store_true",
        help=(
            "Re-run each accepted tree path as one compact causal forward "
            "before committing its KV rows."
        ),
    )
    parser.add_argument(
        "--verification-compact-root-margin-threshold",
        type=float,
        default=0.0,
        help=(
            "Recheck the packed root with q_len=1 when its top-two margin "
            "falls below this threshold."
        ),
    )
    parser.add_argument(
        "--disable-bf16-reduced-precision-reduction",
        action="store_true",
        help=(
            "Accumulate BF16 GEMM reductions without CUDA's reduced-precision "
            "fast path. This reduces q_len-dependent verifier drift."
        ),
    )
    parser.add_argument(
        "--qshape-fixed-rows",
        type=int,
        default=0,
        help=(
            "Pad short decoder MLP/LM-head inputs to this fixed row count; "
            "zero disables shape canonicalization."
        ),
    )
    parser.add_argument(
        "--qshape-attention-fixed-rows",
        type=int,
        default=0,
        help=(
            "Pad short SDPA queries and ephemeral K/V rows to this count; "
            "zero disables attention shape canonicalization."
        ),
    )
    parser.add_argument(
        "--qshape-attention-key-block",
        type=int,
        default=0,
        help=(
            "Round ephemeral attention K/V length up to this block when "
            "fixed-row attention is enabled; zero disables key bucketing."
        ),
    )
    parser.add_argument(
        "--qshape-exact-root-attention-rows",
        type=int,
        default=0,
        help=(
            "For packed calls up to this row count, recompute only the root "
            "SDPA row against the physical prefix-plus-root key tensor; zero "
            "disables exact-root attention."
        ),
    )
    parser.add_argument("--visual-threshold", type=float, default=0.55)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--visual-gamma", type=float, default=1.0)
    parser.add_argument("--visual-weight", type=float, default=0.7)
    parser.add_argument("--acceptance-weight", type=float, default=0.3)
    parser.add_argument("--acceptance-ema-decay", type=float, default=0.8)
    parser.add_argument("--grounding-layer", type=int, default=-1)
    parser.add_argument("--confidence-margin-scale", type=float, default=5.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--samples-per-topic",
        type=int,
        help="Select the first N samples from every MMSpec topic before --max-samples.",
    )
    parser.add_argument(
        "--topic-offset",
        type=int,
        default=0,
        help="Skip the first N samples in each topic before balanced selection.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--save-decoded-output", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument(
        "--no-policy-trace",
        action="store_true",
        help=(
            "Disable per-iteration policy-trace construction and storage for "
            "compact, production-like latency evaluation."
        ),
    )
    parser.add_argument(
        "--enable-repeat-guard",
        action="store_true",
        help="Enable SAM's non-standard repeated-ngram early-stop heuristic.",
    )
    parser.add_argument(
        "--no-rotate-policy-order",
        dest="rotate_policy_order",
        action="store_false",
    )
    parser.set_defaults(rotate_policy_order=True)
    args = parser.parse_args()
    if args.visual_lexical_pool_size <= 0:
        parser.error("--visual-lexical-pool-size must be positive")
    if args.visual_lexical_width <= 0:
        parser.error("--visual-lexical-width must be positive")
    if args.hst_neighbors <= 0 or args.hst_temperature <= 0.0:
        parser.error("HST neighbors and temperature must be positive")
    if not 0.0 <= args.hst_token_weight <= 1.0:
        parser.error("--hst-token-weight must be in [0, 1]")
    if not 0.0 <= args.hst_visual_weight <= 1.0:
        parser.error("--hst-visual-weight must be in [0, 1]")
    if not 0.0 <= args.hst_min_confidence <= 1.0:
        parser.error("--hst-min-confidence must be in [0, 1]")
    if args.verification_margin_threshold < 0.0:
        parser.error("--verification-margin-threshold must be non-negative")
    if args.verification_compact_root_margin_threshold < 0.0:
        parser.error(
            "--verification-compact-root-margin-threshold must be "
            "non-negative"
        )
    if args.qshape_fixed_rows < 0:
        parser.error("--qshape-fixed-rows must be non-negative")
    if args.qshape_attention_fixed_rows < 0:
        parser.error("--qshape-attention-fixed-rows must be non-negative")
    if args.qshape_attention_key_block < 0:
        parser.error("--qshape-attention-key-block must be non-negative")
    if args.qshape_exact_root_attention_rows < 0:
        parser.error(
            "--qshape-exact-root-attention-rows must be non-negative"
        )
    if args.qshape_exact_root_attention_rows == 1:
        parser.error(
            "--qshape-exact-root-attention-rows must be zero or greater "
            "than one"
        )
    if (
        args.qshape_attention_key_block > 0
        and args.qshape_attention_fixed_rows <= 0
    ):
        parser.error(
            "--qshape-attention-key-block requires positive "
            "--qshape-attention-fixed-rows"
        )
    if (
        args.verification_layer_diagnostics
        and args.verification_margin_threshold <= 0.0
    ):
        parser.error(
            "--verification-layer-diagnostics requires a positive "
            "--verification-margin-threshold"
        )
    if (
        args.verification_compact_path_repair
        and args.verification_margin_threshold > 0.0
    ):
        parser.error(
            "--verification-compact-path-repair cannot be combined with "
            "positive --verification-margin-threshold"
        )
    if args.no_policy_trace and (
        args.hst_trace_diagnostics
        or args.verification_trace_diagnostics
        or args.verification_layer_diagnostics
    ):
        parser.error(
            "trace diagnostics require policy traces; remove "
            "--no-policy-trace"
        )
    args.model = args.base_model_path
    evaluate(args)


if __name__ == "__main__":
    main()
