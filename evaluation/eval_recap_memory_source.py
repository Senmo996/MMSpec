"""Fixed-prefix, past-only memory-source intervention; not a speed benchmark.

Only the image used to produce historical prediction rows changes. Both banks
share keys, source positions, prompt, and the true-image target trajectory.
Reuse the production root-candidate builder to compare U/C/G, backoff and
fusion. This diagnostic excludes persistent memory and uncommitted branches.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

MMSPEC_ROOT = Path(__file__).resolve().parents[1]
for directory in (MMSPEC_ROOT.parent, MMSPEC_ROOT):
    sys.path.insert(0, str(directory))

from evaluation.selective_reuse_counterfactual_bank_protocol import (
    build_matched_wrong_image_pairs,
    resize_wrong_image,
)
from evaluation.selective_reuse_protocol import _selected_qwen_logits
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel
from new_dream.evaluation.eval_llava_fixed_multi import (
    build_inputs_for_record,
    dataset_slug,
    load_base_dataset,
    load_image_for_record,
    load_or_create_manifest,
    load_processor_for_model,
    parse_datasets,
)

MODES = ("U", "C", "G", "backoff", "uniform", "fusion")
VIEWS = ("matched", "wrong_image")
PROTOCOL = "recap_past_only_image_conditioned_memory_v1"


def root_candidates(tokens, position, banks, mode, top_k, blocked_token_id):
    orders = {"U": (1,), "C": (2,), "G": (3,)}.get(mode, (1, 2, 3))
    rows, scores = {}, {}
    for order in orders:
        key = tuple(tokens[max(0, position + 1 - order):position + 1])
        entry = banks[order].get(key)
        if entry is not None:
            if entry[0] >= position:
                raise ValueError("memory row must precede the current query")
            builder_key = key[0] if order == 1 else key
            rows[order] = {builder_key: entry[1]}
            scores[order] = {builder_key: entry[2]}
    flat, _, _, _ = TreeRecyclingSpecModel._build_tree(
        root_token=tokens[position],
        transitions=torch.empty(0, dtype=torch.long),
        transition_valid=torch.empty(0, dtype=torch.bool),
        width=top_k, depth=1, node_budget=top_k,
        blocked_token_id=blocked_token_id,
        host_transitions=rows.get(1, {}),
        host_transition_scores=scores.get(1, {}),
        root_previous_token=tokens[position - 1] if position >= 1 else None,
        root_previous_previous_token=tokens[position - 2] if position >= 2 else None,
        host_context_transitions=rows.get(2),
        host_context_transition_scores=scores.get(2),
        host_trigram_transitions=rows.get(3),
        host_trigram_transition_scores=scores.get(3),
        context_candidate_mode={"uniform": "fusion_uniform", "fusion": "fusion"}.get(mode, "strict"),
        score_priority_layout=True,
    )
    return [int(token) for token in flat[1:]]


def replay(tokens, prompt_length, rows_by_view, top_k, blocked_token_id):
    """Query before update: no current/future logits can enter a candidate bank."""
    if prompt_length < 1 or len(tokens) <= prompt_length:
        raise ValueError("expected a nonempty prompt and at least one output token")
    prompt_positions, _ = TreeRecyclingSpecModel._select_prompt_transition_rows(
        tokens[:prompt_length]
    )
    banks = {view: {order: {} for order in (1, 2, 3)} for view in VIEWS}
    for view in VIEWS:
        for position in prompt_positions:
            if tokens[position] != blocked_token_id:
                ids, scores = rows_by_view[view][position]
                banks[view][1][(tokens[position],)] = (position, ids, scores)

    states = []
    # The first generated token comes directly from prefill, not recycling.
    for position in range(prompt_length, len(tokens) - 1):
        candidates = {
            mode: {
                view: root_candidates(tokens, position, banks[view], mode, top_k, blocked_token_id)
                for view in VIEWS
            }
            for mode in MODES
        }
        source_positions = {}
        for order in (1, 2, 3):
            key = tuple(tokens[max(0, position + 1 - order):position + 1])
            first = banks["matched"][order].get(key)
            second = banks["wrong_image"][order].get(key)
            if (None if first is None else first[0]) != (None if second is None else second[0]):
                raise ValueError("intervention banks have different source positions")
            source_positions[str(order)] = None if first is None else first[0]
        states.append({
            "position": position,
            "generated_offset": position + 1 - prompt_length,
            "context_token_ids": tokens[max(0, position - 2):position + 1],
            "reference_token_id": int(rows_by_view["matched"][position][0][0]),
            "trajectory_token_id": int(tokens[position + 1]),
            "source_positions": source_positions,
            "candidates": candidates,
        })
        for view in VIEWS:
            ids, scores = rows_by_view[view][position]
            for order in (1, 2, 3):
                if position + 1 >= order:
                    key = tuple(tokens[position + 1 - order:position + 1])
                    banks[view][order][key] = (position, ids, scores)
    return states


def summarize(records, top_k, bootstrap_resamples, seed):
    by_dataset = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    results = {}
    for dataset, samples in by_dataset.items():
        modes = {}
        for mode in MODES:
            totals = defaultdict(int)
            # Pair by target image: all repeated questions stay in one cluster.
            cluster_counts = defaultdict(lambda: np.zeros(2, dtype=float))
            for sample in samples:
                cluster = sample["pair"]["target_image_identity"]
                for state in sample["states"]:
                    target = state["reference_token_id"]
                    totals["num_states"] += 1
                    for view in VIEWS:
                        candidates = state["candidates"][mode][view]
                        totals[view + "_available"] += bool(candidates)
                        totals[view + "_hits"] += target in candidates
                        totals[view + "_candidate_count"] += len(candidates)
                    budget = min(len(state["candidates"][mode][v]) for v in VIEWS)
                    if budget:
                        delta = int(target in state["candidates"][mode]["matched"][:budget]) - int(
                            target in state["candidates"][mode]["wrong_image"][:budget]
                        )
                        totals["matched_budget_states"] += 1
                        totals["matched_budget_delta_sum"] += delta
                        cluster_counts[cluster] += (delta, 1)
            count = totals["num_states"]
            row = {"counts": dict(totals)}
            for view in VIEWS:
                available = totals[view + "_available"]
                row[view] = {
                    "query_coverage": available / count if count else None,
                    "recall_at_k_all_states": totals[view + "_hits"] / count if count else None,
                    "conditional_recall_at_k": totals[view + "_hits"] / available if available else None,
                    "avg_candidate_count": totals[view + "_candidate_count"] / count if count else None,
                }
            matched_count = totals["matched_budget_states"]
            row["matched_minus_wrong_recall_equal_budget"] = (
                totals["matched_budget_delta_sum"] / matched_count if matched_count else None
            )
            row["num_eligible_image_clusters"] = len(cluster_counts)
            row["paired_image_cluster_bootstrap_95_ci"] = None
            if len(cluster_counts) >= 2:
                values = np.array(list(cluster_counts.values()))
                rng = np.random.default_rng(seed)
                draws = np.empty(bootstrap_resamples)
                for index in range(bootstrap_resamples):
                    sampled = values[rng.integers(len(values), size=len(values))].sum(axis=0)
                    draws[index] = sampled[0] / sampled[1]
                row["paired_image_cluster_bootstrap_95_ci"] = np.quantile(draws, [0.025, 0.975]).tolist()
            modes[mode] = row
        all_states = [state for sample in samples for state in sample["states"]]
        results[dataset] = {
            "num_records": len(samples),
            "num_states": len(all_states),
            "reference_matches_native_trajectory_ratio": (
                sum(s["reference_token_id"] == s["trajectory_token_id"] for s in all_states) / len(all_states)
                if all_states else None
            ),
            "wrong_image_category_fallback_ratio": sum(s["pair"]["used_category_fallback"] for s in samples) / len(samples),
            "modes": modes,
        }
    return {
        "protocol": PROTOCOL,
        "num_records": len(records),
        "num_states": sum(len(record["states"]) for record in records),
        "top_k": top_k,
        "benchmark_aggregation": "per dataset, state weighted; no pooled headline",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": seed,
        "bootstrap_scope": "paired target-image clusters conditional on fixed donor pairing; exploratory intervals",
        "diagnostic_only": True,
        "formal_speedup_evaluated": False,
        "datasets": results,
    }


@torch.inference_mode()
def evaluate(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "results.jsonl"
    if output_path.exists() or (output_root / "run_config.json").exists():
        raise FileExistsError("use a fresh output directory; this diagnostic does not silently resume")
    data_payloads, pairs_manifest, manifest_hashes = [], [], {}
    manifest_args = SimpleNamespace(manifest_dir=args.manifest_dir, seed=args.seed,
                                    sample_num=args.source_pool_size, regenerate_manifest=False)
    for dataset in args.datasets:
        path = Path(args.manifest_dir) / f"{dataset_slug(dataset)}_seed{args.seed}_n{args.source_pool_size}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        selected, manifest, _ = load_or_create_manifest(dataset, load_base_dataset(dataset), manifest_args)
        if len(selected) != args.source_pool_size:
            raise ValueError(f"unexpected manifest size: {dataset}")
        manifest_hashes[dataset] = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs = build_matched_wrong_image_pairs(selected, manifest, seed=args.seed, name=dataset)
        chosen = pairs[:args.samples_per_benchmark]
        pairs_manifest.extend({"dataset": dataset, **pair} for pair in chosen)
        data_payloads.append((dataset, selected, manifest, chosen))
    config = {**vars(args), "protocol": PROTOCOL, "manifest_sha256": manifest_hashes,
              "modes": MODES, "treatments": VIEWS,
              "scope": "cold request; first-occurrence prompt U seeding; committed-prefix updates only; root recall, no speculative timing"}
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_root / "pairing_manifest.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs_manifest))
    print(f"protocol={PROTOCOL} output={output_path} planned_records={len(pairs_manifest)}", flush=True)
    print("loading_model=" + args.base_model_path, flush=True)
    model = TreeRecyclingSpecModel.from_pretrained(
        base_model_path=args.base_model_path, spec_model_path="", total_token=112,
        torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa",
    )
    model.processor = load_processor_for_model(args.base_model_path)
    model.eval()
    if model.base_model.config.model_type != "qwen2_5_vl":
        raise ValueError("this intervention currently supports Qwen2.5-VL only")
    blocked_token_id = int(model.base_model.config.image_token_id)
    records = []
    with output_path.open("x", encoding="utf-8") as handle:
        for dataset, selected, manifest, pairs in data_payloads:
            for pair in pairs:
                started = time.perf_counter()
                index = pair["target_position"]
                row = selected[index]
                true_image = load_image_for_record(dataset, row)
                wrong_image = resize_wrong_image(
                    load_image_for_record(dataset, selected[pair["source_position"]]), true_image
                )
                if true_image.convert("RGB").tobytes() == wrong_image.convert("RGB").tobytes():
                    raise ValueError("paired images have identical content")
                inputs = build_inputs_for_record(model, dataset, row)
                wrong_inputs = build_inputs_for_record(model, dataset, row, image_override=wrong_image)
                for key in ("input_ids", "attention_mask", "image_grid_thw"):
                    if not torch.equal(inputs[key], wrong_inputs[key]):
                        raise ValueError(f"intervention changed {key}")
                if inputs["pixel_values"].shape != wrong_inputs["pixel_values"].shape:
                    raise ValueError("intervention changed pixel tensor shape")
                torch.manual_seed(args.seed)
                sequence = model.specgenerate(**inputs, temperature=0.0, max_new_tokens=args.max_new_token,
                                              draft_policy="target", return_policy_trace=False)
                tokens = sequence[0].tolist()
                prompt_length = int(inputs["input_ids"].shape[1])
                if not 0 < len(tokens) - prompt_length <= args.max_new_token:
                    raise ValueError("invalid target generation length")
                prompt_positions, _ = model._select_prompt_transition_rows(tokens[:prompt_length])
                positions = sorted(set(prompt_positions) | set(range(prompt_length, len(tokens) - 1)))
                logit_positions = torch.tensor(positions, device=sequence.device)
                rows_by_view = {}
                previous_rope_deltas = getattr(model.base_model, "rope_deltas", None)
                try:
                    for view, visual_inputs in (("matched", inputs), ("wrong_image", wrong_inputs)):
                        logits = _selected_qwen_logits(
                            model.base_model, input_ids=sequence, attention_mask=torch.ones_like(sequence),
                            pixel_values=visual_inputs["pixel_values"], image_grid_thw=visual_inputs["image_grid_thw"],
                            logit_positions=logit_positions,
                        )[0].float()
                        values, ids = torch.topk(logits, k=args.top_k, dim=-1)
                        probabilities = torch.exp(values - torch.logsumexp(logits, dim=-1, keepdim=True))
                        rows_by_view[view] = dict(zip(positions, zip(ids.cpu().tolist(), probabilities.cpu().tolist())))
                        del logits, values, ids, probabilities
                finally:
                    model.base_model.rope_deltas = previous_rope_deltas
                states = replay(tokens, prompt_length, rows_by_view, args.top_k, blocked_token_id)
                record = {
                    "protocol": PROTOCOL, "dataset": dataset,
                    "question_id": f"{dataset_slug(dataset)}:{manifest[index]['source_index']}",
                    "pair": pair, "prompt_length": prompt_length,
                    "num_generated_tokens": len(tokens) - prompt_length,
                    "output_token_sha256": hashlib.sha256(json.dumps(tokens[prompt_length:]).encode()).hexdigest(),
                    "input_ids_and_visual_grid_identical": True,
                    "memory_query_before_update": True,
                    "states": states, "diagnostic_wall_seconds": time.perf_counter() - started,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"record={len(records)}/{len(pairs_manifest)} dataset={dataset} source_index={manifest[index]['source_index']} tokens={record['num_generated_tokens']} states={len(states)} elapsed={record['diagnostic_wall_seconds']:.2f}s", flush=True)
            summary = summarize(records, args.top_k, args.bootstrap_resamples, args.seed)
            summary["jsonl_path"] = str(output_path)
            summary["complete"] = len(records) == len(pairs_manifest)
            (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not any(record["states"] for record in records):
        raise ValueError("no diagnostic states were produced")
    print("completed_summary=" + str(output_root / "summary.json"), flush=True)


def self_test():
    tokens = [1, 2, 1, 2, 1, 2, 1]
    matched = {p: ([9, 8], [0.7, 0.2]) for p in range(len(tokens))}
    wrong = {p: ([7, 6], [0.6, 0.3]) for p in range(len(tokens))}
    states = replay(tokens, 2, {"matched": matched, "wrong_image": wrong}, 2, -1)
    assert states[0]["candidates"]["U"] == {"matched": [9, 8], "wrong_image": [7, 6]}
    assert states[0]["candidates"]["C"]["matched"] == []
    assert states[-1]["candidates"]["C"]["matched"] == [9, 8]
    assert states[-1]["candidates"]["G"]["matched"] == [9, 8]
    changed = dict(matched)
    changed[2] = ([5, 4], [0.8, 0.1])
    other = replay(tokens, 2, {"matched": changed, "wrong_image": wrong}, 2, -1)
    assert other[0]["candidates"] == states[0]["candidates"]  # current logits cannot leak
    for state in states:
        assert all(p is None or p < state["position"] for p in state["source_positions"].values())
        assert all(len(c) <= 2 for rows in state["candidates"].values() for c in rows.values())
    banks = {1: {(1,): (0, [7, 6], [0.8, 0.1])},
             2: {(2, 1): (1, [9, 8], [0.8, 0.1])}, 3: {}}
    assert root_candidates([2, 1, 2, 1], 3, banks, "backoff", 2, -1) == [9, 8]
    assert root_candidates([2, 1, 2, 1], 3, banks, "fusion", 2, -1) == [9, 7]
    sample = {"dataset": "test", "pair": {"target_image_identity": "image1", "used_category_fallback": False}, "states": states}
    summary = summarize([sample], 2, 10, 42)
    assert summary["datasets"]["test"]["modes"]["U"]["matched_minus_wrong_recall_equal_budget"] == 1.0
    assert summary["num_states"] == 4
    print("self_test=passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", default=str(MMSPEC_ROOT.parent / "models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--manifest-dir", default=str(MMSPEC_ROOT.parent / "result/manifests/dream_baseline_seed42"))
    parser.add_argument("--output-root")
    parser.add_argument("--datasets", default="ChartQA,TextVQA,OCRBench,MMT-Bench")
    parser.add_argument("--source-pool-size", type=int, default=100)
    parser.add_argument("--samples-per-benchmark", type=int, default=100)
    parser.add_argument("--max-new-token", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.output_root or not 0 < args.samples_per_benchmark <= args.source_pool_size or args.source_pool_size < 2:
        parser.error("require output-root and 0 < samples-per-benchmark <= source-pool-size, with pool >= 2")
    if args.top_k < 1 or args.max_new_token < 2 or args.bootstrap_resamples < 1:
        parser.error("require top-k >= 1, max-new-token >= 2, bootstrap-resamples >= 1")
    args.datasets = parse_datasets(args.datasets)
    if not args.datasets or "MMSpec" in args.datasets or len(set(args.datasets)) != len(args.datasets):
        parser.error("choose distinct fixed benchmarks; this first diagnostic excludes multi-turn MMSpec")
    evaluate(args)


if __name__ == "__main__":
    main()
