"""Past-only, same-state tree-path coverage on saved native greedy trajectories.

This is not packed-tree verification or a speed benchmark. Reuse the previous
memory-source experiment's trajectories, and the production tree builder.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

import eval_recap_memory_source as source

Model = source.TreeRecyclingSpecModel
MODES = ("U", "backoff", "uniform", "fusion")
PROTOCOL = "recap_past_only_same_trajectory_path_coverage_v1"


def build_paths(tokens, position, rows, scores, mode, width, depth, budget, blocked):
    flat, _, _, paths = Model._build_tree(
        root_token=tokens[position],
        transitions=torch.empty(0, dtype=torch.long),
        transition_valid=torch.empty(0, dtype=torch.bool),
        width=width, depth=depth, node_budget=budget, blocked_token_id=blocked,
        host_transitions=rows[1], host_transition_scores=scores[1],
        root_previous_token=tokens[position - 1] if position >= 1 else None,
        root_previous_previous_token=tokens[position - 2] if position >= 2 else None,
        host_context_transitions=rows[2] if mode != "U" else None,
        host_context_transition_scores=scores[2] if mode != "U" else None,
        host_trigram_transitions=rows[3] if mode != "U" else None,
        host_trigram_transition_scores=scores[3] if mode != "U" else None,
        context_candidate_mode={"uniform": "fusion_uniform", "fusion": "fusion"}.get(mode, "strict"),
        score_priority_layout=True,
    )
    if len(paths) != len(flat) - 1 or len(paths) > budget:
        raise ValueError("unexpected production tree node count")
    if any(path[-1] != i or any(a >= b for a, b in zip(path, path[1:]))
           for i, path in enumerate(paths, 1)):
        raise ValueError("tree must contain one ancestor-ordered path per node")
    return Model._tree_path_token_ids(flat, paths)


def replay_paths(tokens, prompt_length, prediction_rows, width, depth, node_cap, blocked):
    """Build from past rows only; look at the future solely to score frozen trees."""
    rows, scores = ({order: {} for order in (1, 2, 3)} for _ in range(2))
    prompt_positions, _ = Model._select_prompt_transition_rows(tokens[:prompt_length])
    for position in prompt_positions:
        if tokens[position] != blocked:
            rows[1][tokens[position]], scores[1][tokens[position]] = prediction_rows[position]
    states = []
    for position in range(prompt_length, len(tokens) - 1):
        cap_paths = {
            mode: build_paths(tokens, position, rows, scores, mode, width, depth, node_cap, blocked)
            for mode in MODES
        }
        budget = min(len(paths) for paths in cap_paths.values())
        # Packed order is DFS, NOT score selection order. Rebuild; do not slice.
        equal_paths = {
            mode: paths if len(paths) == budget else build_paths(
                tokens, position, rows, scores, mode, width, depth, budget, blocked
            )
            for mode, paths in cap_paths.items()
        }
        if any(len(paths) != budget for paths in equal_paths.values()):
            raise ValueError("failed to match actual candidate-node counts")
        future = tokens[position + 1:position + 1 + depth]
        agreement_prefix = 0
        for offset, token in enumerate(future):
            if prediction_rows[position + offset][0][0] != token:
                break
            agreement_prefix += 1
        methods = {}
        for mode in MODES:
            length = Model._longest_matching_path(equal_paths[mode], future)
            methods[mode] = {
                "nodes_at_cap": len(cap_paths[mode]),
                "match_length_at_cap": Model._longest_matching_path(cap_paths[mode], future),
                "matched_nodes": len(equal_paths[mode]),
                "match_length": length,
                "tf_supported_match_length": min(length, agreement_prefix),
                "matched_tree_sha256": hashlib.sha256(json.dumps(equal_paths[mode]).encode()).hexdigest(),
            }
        states.append({
            "position": position, "generated_offset": position + 1 - prompt_length,
            "available_future_tokens": len(tokens) - position - 1,
            "matched_node_budget": budget,
            "tf_native_first_token_agrees": prediction_rows[position][0][0] == future[0],
            "tf_native_agreement_prefix_length": agreement_prefix,
            "modes": methods,
        })
        # No current or uncommitted branch logits enter any of the trees above.
        for order in (1, 2, 3):
            if position + 1 >= order:
                key = tokens[position] if order == 1 else tuple(tokens[position + 1 - order:position + 1])
                rows[order][key], scores[order][key] = prediction_rows[position]
    return states


def summarize(records, depth, node_cap, resamples, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)
    datasets = {}
    for dataset, samples in grouped.items():
        states = [state for sample in samples for state in sample["states"]]
        if not states:
            raise ValueError(f"no diagnostic states for {dataset}")
        methods = {}
        for mode in MODES:
            lengths = np.array([s["modes"][mode]["match_length"] for s in states])
            methods[mode] = {
                "avg_match_length_equal_nodes": float(lengths.mean()),
                "median_match_length_equal_nodes": float(np.median(lengths)),
                "p90_match_length_equal_nodes": float(np.quantile(lengths, 0.9)),
                "max_match_length_equal_nodes": int(lengths.max()),
                "first_token_hit_ratio": float((lengths >= 1).mean()),
                "at_least_two_tokens_ratio": float((lengths >= 2).mean()),
                "avg_tokens_beyond_first": float(np.maximum(lengths - 1, 0).mean()),
                "avg_tf_supported_match_length": float(np.mean([s["modes"][mode]["tf_supported_match_length"] for s in states])),
                "avg_match_length_at_node_cap": float(np.mean([s["modes"][mode]["match_length_at_cap"] for s in states])),
                "avg_nodes_at_cap": float(np.mean([s["modes"][mode]["nodes_at_cap"] for s in states])),
            }
        comparisons = {}
        for left, right in (("backoff", "U"), ("fusion", "U"), ("uniform", "backoff"), ("fusion", "backoff")):
            clusters = defaultdict(lambda: np.zeros(7))
            for sample in samples:
                counts = clusters[sample["target_image_identity"]]
                for state in sample["states"]:
                    a, b = (state["modes"][m] for m in (left, right))
                    delta = a["match_length"] - b["match_length"]
                    common = a["match_length"] >= 1 and b["match_length"] >= 1
                    counts += (
                        1, delta,
                        max(a["match_length"] - 1, 0) - max(b["match_length"] - 1, 0),
                        a["tf_supported_match_length"] - b["tf_supported_match_length"],
                        common, delta if common else 0,
                        int(a["match_length"] >= 2) - int(b["match_length"] >= 2),
                    )
            values = np.array(list(clusters.values()))
            total = values.sum(axis=0)
            rng = np.random.default_rng(seed)
            draws = np.array([values[rng.integers(len(values), size=len(values))].sum(axis=0) for _ in range(resamples)])
            contrasts = {}
            for name, numerator, denominator in (
                ("match_length", 1, 0), ("tokens_beyond_first", 2, 0),
                ("tf_supported_match_length", 3, 0),
                ("match_length_on_common_first_hit", 5, 4),
                ("at_least_two_tokens_ratio", 6, 0),
            ):
                valid = draws[:, denominator] > 0
                contrasts[name] = {
                    "delta": float(total[numerator] / total[denominator]) if total[denominator] else None,
                    "num_states": int(total[denominator]),
                    "paired_image_cluster_bootstrap_95_ci": np.quantile(
                        draws[valid, numerator] / draws[valid, denominator], [0.025, 0.975]
                    ).tolist() if len(values) >= 2 and valid.any() else None,
                    "valid_bootstrap_draws": int(valid.sum()),
                }
            comparisons[f"{left}_minus_{right}"] = contrasts
        datasets[dataset] = {
            "num_records": len(samples), "num_states": len(states),
            "num_image_clusters": len({s["target_image_identity"] for s in samples}),
            "avg_matched_node_budget": float(np.mean([s["matched_node_budget"] for s in states])),
            "positive_matched_budget_ratio": sum(s["matched_node_budget"] > 0 for s in states) / len(states),
            "tf_native_first_token_agreement_ratio": sum(s["tf_native_first_token_agrees"] for s in states) / len(states),
            "recomputed_tf_matches_source_reference_ratio": sum(s["num_source_reference_agreements"] for s in samples) / len(states),
            "future_shorter_than_depth_ratio": sum(s["available_future_tokens"] < depth for s in states) / len(states),
            "modes": methods, "paired_comparisons": comparisons,
        }
    return {
        "protocol": PROTOCOL, "num_records": len(records),
        "num_states": sum(len(r["states"]) for r in records),
        "depth": depth, "node_cap": node_cap,
        "diagnostic_only": True, "formal_speedup_evaluated": False,
        "packed_tree_verification_evaluated": False,
        "metric": "longest tree prefix matching the saved native greedy continuation; excludes root and correction",
        "budget_matching": "minimum actual candidate count across four cap trees; rebuild every larger tree at that budget",
        "tf_supported_metric": "matched native path clipped at first TF/native disagreement; not a coherent alternative rollout",
        "aggregation": "all states including empty banks, state weighted within each benchmark; no pooled headline",
        "bootstrap_scope": "paired target-image clusters; exploratory, conditional on saved trajectories",
        "bootstrap_resamples": resamples, "seed": seed, "datasets": datasets,
    }


def restore_generated(record):
    if not record["states"]:
        raise ValueError("cannot restore a one-token trajectory from this source format")
    generated = [record["states"][0]["context_token_ids"][-1]]
    generated.extend(state["trajectory_token_id"] for state in record["states"])
    if len(generated) != record["num_generated_tokens"] or hashlib.sha256(
        json.dumps(generated).encode()
    ).hexdigest() != record["output_token_sha256"]:
        raise ValueError("saved trajectory reconstruction failed its original hash")
    return generated


@torch.inference_mode()
def evaluate(args):
    source_root, output_root = Path(args.source_root), Path(args.output_root)
    if (output_root / "run_config.json").exists() or (output_root / "results.jsonl").exists():
        raise FileExistsError("use a fresh output directory")
    previous = json.loads((source_root / "run_config.json").read_text())
    previous_summary = json.loads((source_root / "summary.json").read_text())
    if previous["protocol"] != source.PROTOCOL or not previous_summary.get("complete"):
        raise ValueError("require a complete compatible memory-source run")
    source_jsonl = source_root / "results.jsonl"
    source_records = [json.loads(line) for line in source_jsonl.read_text().splitlines() if line.strip()]
    if len(source_records) != previous_summary["num_records"] or len({r["question_id"] for r in source_records}) != len(source_records):
        raise ValueError("source record count or uniqueness check failed")
    manifest_args = SimpleNamespace(manifest_dir=previous["manifest_dir"], seed=previous["seed"],
                                    sample_num=previous["source_pool_size"], regenerate_manifest=False)
    payloads = []
    for dataset in previous["datasets"]:
        manifest_path = Path(previous["manifest_dir"]) / f"{source.dataset_slug(dataset)}_seed{previous['seed']}_n{previous['source_pool_size']}.jsonl"
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != previous["manifest_sha256"][dataset]:
            raise ValueError(f"input manifest changed: {dataset}")
        selected, manifest, _ = source.load_or_create_manifest(dataset, source.load_base_dataset(dataset), manifest_args)
        chosen = [r for r in source_records if r["dataset"] == dataset][:args.samples_per_benchmark]
        if len(chosen) != args.samples_per_benchmark:
            raise ValueError(f"insufficient source records: {dataset}")
        for record in chosen:
            restore_generated(record)
        payloads.append((dataset, selected, manifest, chosen))
    config = {**vars(args), "protocol": PROTOCOL, "source_config": previous,
              "source_jsonl_sha256": hashlib.sha256(source_jsonl.read_bytes()).hexdigest(),
              "modes": MODES, "memory": "correct-image, cold request, prompt U seeding; past committed rows only",
              "score_weights": "production defaults; ranked available sources, not fixed weights for absent orders",
              "code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                  Path(__file__), source.MMSPEC_ROOT / "method/sam_grounded/tree_recycling_model.py",
                  source.MMSPEC_ROOT / "evaluation/selective_reuse_protocol.py",
                  source.MMSPEC_ROOT.parent / "new_dream/evaluation/eval_llava_fixed_multi.py")}}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    planned = sum(len(item[3]) for item in payloads)
    print(f"protocol={PROTOCOL} output_root={output_root} planned_records={planned}", flush=True)
    print(f"loading_model={previous['base_model_path']}", flush=True)
    model = Model.from_pretrained(base_model_path=previous["base_model_path"], spec_model_path="", total_token=112,
                                 torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa")
    model.processor = source.load_processor_for_model(previous["base_model_path"])
    model.eval()
    if model.base_model.config.model_type != "qwen2_5_vl":
        raise ValueError("this replay currently supports Qwen2.5-VL only")
    blocked = int(model.base_model.config.image_token_id)
    records = []
    output_path = output_root / "results.jsonl"
    with output_path.open("x", encoding="utf-8") as handle:
        for dataset, selected, manifest, chosen in payloads:
            for saved in chosen:
                started = time.perf_counter()
                index = saved["pair"]["target_position"]
                if saved["question_id"] != f"{source.dataset_slug(dataset)}:{manifest[index]['source_index']}":
                    raise ValueError("source question/manifest identity mismatch")
                inputs = source.build_inputs_for_record(model, dataset, selected[index])
                prompt = inputs["input_ids"][0].tolist()
                tokens = prompt + restore_generated(saved)
                if len(prompt) != saved["prompt_length"]:
                    raise ValueError("prompt length changed since source run")
                for offset, state in enumerate(saved["states"], len(prompt)):
                    if state["position"] != offset or state["context_token_ids"] != tokens[max(0, offset - 2):offset + 1]:
                        raise ValueError("restored query position/context mismatch")
                sequence = torch.tensor([tokens], device=inputs["input_ids"].device)
                prompt_positions, _ = model._select_prompt_transition_rows(prompt)
                positions = sorted(set(prompt_positions) | set(range(len(prompt), len(tokens) - 1)))
                old_rope = getattr(model.base_model, "rope_deltas", None)
                try:
                    logits = source._selected_qwen_logits(
                        model.base_model, input_ids=sequence, attention_mask=torch.ones_like(sequence),
                        pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
                        logit_positions=torch.tensor(positions, device=sequence.device),
                    )[0].float()
                    values, ids = torch.topk(logits, k=previous["top_k"], dim=-1)
                    probabilities = torch.exp(values - torch.logsumexp(logits, dim=-1, keepdim=True))
                    prediction_rows = dict(zip(positions, zip(ids.cpu().tolist(), probabilities.cpu().tolist())))
                    del logits, values, ids, probabilities
                finally:
                    model.base_model.rope_deltas = old_rope
                states = replay_paths(tokens, len(prompt), prediction_rows, previous["top_k"], args.depth, args.node_cap, blocked)
                record = {
                    "protocol": PROTOCOL, "dataset": dataset, "question_id": saved["question_id"],
                    "target_image_identity": saved["pair"]["target_image_identity"],
                    "output_token_sha256": saved["output_token_sha256"], "trajectory_hash_verified": True,
                    "prompt_length": len(prompt), "num_generated_tokens": saved["num_generated_tokens"],
                    "input_token_ids": tokens, "blocked_token_id": blocked,
                    # Save compact rows so later audits need no further GPU forwards.
                    "prediction_rows": [[p, *prediction_rows[p]] for p in positions],
                    "num_source_reference_agreements": sum(prediction_rows[s["position"]][0][0] == s["reference_token_id"] for s in saved["states"]),
                    "states": states, "diagnostic_wall_seconds": time.perf_counter() - started,
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                records.append(record)
                print(f"record={len(records)}/{planned} dataset={dataset} question_id={record['question_id']} states={len(states)} elapsed={record['diagnostic_wall_seconds']:.2f}s", flush=True)
            summary = summarize(records, args.depth, args.node_cap, args.bootstrap_resamples, previous["seed"])
            summary.update(jsonl_path=str(output_path), complete=len(records) == planned)
            (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"completed_summary={output_root / 'summary.json'}", flush=True)


def self_test():
    tokens = [1, 2, 3, 4, 1, 2, 3, 4, 5]
    rows = {p: ([{1: 2, 2: 3, 3: 4, 4: 5, 5: 6}[token], 90 + token], [0.8, 0.1]) for p, token in enumerate(tokens)}
    states = replay_paths(tokens, 4, rows, 2, 4, 5, -1)
    assert states[0]["modes"]["U"]["match_length"] == 4
    changed = {p: ([88, 89], [0.8, 0.1]) if p >= 4 else row for p, row in rows.items()}
    other = replay_paths(tokens, 4, changed, 2, 4, 5, -1)
    assert [states[0]["modes"][m]["matched_tree_sha256"] for m in MODES] == [other[0]["modes"][m]["matched_tree_sha256"] for m in MODES]
    assert other[0]["modes"]["U"]["tf_supported_match_length"] == 0
    empty = replay_paths([1, 7, 8], 1, {0: ([9], [1.0]), 1: ([8], [1.0])}, 1, 4, 5, -1)
    assert empty[0]["matched_node_budget"] == 0
    assert Model._longest_matching_path([[9, 2, 3], [2, 7, 4]], [2, 3, 4]) == 1
    bank_rows = {1: {1: [2, 8], 2: [3, 9], 3: [4]}, 2: {(4, 1): [7, 6]}, 3: {}}
    bank_scores = {1: {1: [0.6, 0.4], 2: [0.2, 0.1], 3: [0.9]}, 2: {(4, 1): [0.6, 0.4]}, 3: {}}
    capped = build_paths([4, 1], 1, bank_rows, bank_scores, "U", 2, 4, 5, -1)
    rebuilt = build_paths([4, 1], 1, bank_rows, bank_scores, "U", 2, 4, 2, -1)
    assert rebuilt == [[2], [8]] and rebuilt != capped[:2]  # catches DFS slicing
    assert len(build_paths([4, 1], 1, bank_rows, bank_scores, "backoff", 2, 4, 5, -1)) == 2
    for state in states + other + empty:
        for row in state["modes"].values():
            assert row["matched_nodes"] == state["matched_node_budget"]
            assert 0 <= row["tf_supported_match_length"] <= row["match_length"] <= min(4, state["available_future_tokens"])
    sample = {"dataset": "test", "target_image_identity": "image1", "num_source_reference_agreements": len(states), "states": states}
    summary = summarize([sample], 4, 5, 10, 42)
    assert summary["num_states"] == 4
    assert summary["datasets"]["test"]["paired_comparisons"]["fusion_minus_U"]["match_length"]["delta"] == 0
    assert not summary["packed_tree_verification_evaluated"]
    print("self_test=passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--samples-per-benchmark", type=int, default=100)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--node-cap", type=int, default=63)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.source_root or not args.output_root or min(args.samples_per_benchmark, args.depth, args.node_cap, args.bootstrap_resamples) < 1:
            parser.error("require source-root, output-root, and positive counts")
        evaluate(args)
