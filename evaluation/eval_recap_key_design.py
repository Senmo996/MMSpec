"""Causal, fixed-pool lexical versus representation-key retrieval diagnostic.

This is root Recall@K on saved target trajectories, NOT online acceleration.
At root x_t, dense queries use h_{t-1}, never h_t. A stored prediction row at
s is addressed by h_{s-1}; token-filtered variants additionally require x_s=x_t.
All methods use the union of rows currently retained by the lexical U/C/G maps.
No current/future prediction row, persistent cross-request bank, or speculative
branch representation is available to retrieval.
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
import torch.nn.functional as F

MMSPEC_ROOT = Path(__file__).resolve().parents[1]
for directory in (MMSPEC_ROOT.parent, MMSPEC_ROOT):
    sys.path.insert(0, str(directory))

PROTOCOL = "recap_causal_fixed_pool_key_design_v1"
MODES = (
    "lexical_backoff", "lexical_fusion", "dense_token_top1",
    "dense_token_fusion", "dense_fusion", "dense_centered_token_fusion",
)
WEIGHTS = (0.70, 0.20, 0.10)


def normalize_rows(serialized_rows, top_k):
    rows = {}
    for position, ids, probabilities in serialized_rows:
        values = np.asarray(probabilities, dtype=np.float64)
        if (position in rows or len(ids) != top_k or len(set(ids)) != len(ids)
                or len(values) != len(ids) or not np.isfinite(values).all()
                or (values < 0).any() or values.sum() <= 0):
            raise ValueError("invalid saved prediction row")
        rows[int(position)] = ([int(x) for x in ids], (values / values.sum()).tolist())
    return rows


def fuse_rows(source_positions, rows, top_k, blocked_token_id):
    """Production specificity-rank weights, also used for similarity-rank fusion."""
    if not source_positions:
        return []
    if len(source_positions) == 1:
        return [x for x in rows[source_positions[0]][0][:top_k] if x != blocked_token_id]
    if len(source_positions) > len(WEIGHTS):
        raise ValueError("at most three source rows are supported")
    scores, first_seen = {}, {}
    for weight, position in zip(WEIGHTS, source_positions):
        for token, probability in zip(*rows[position]):
            first_seen.setdefault(token, len(first_seen))
            scores[token] = scores.get(token, 0.0) + weight * probability
    selected = sorted(scores, key=lambda token: (-scores[token], first_seen[token]))[:top_k]
    return [x for x in selected if x != blocked_token_id]


def first_prompt_positions(tokens):
    seen, positions = set(), []
    for position, token in enumerate(tokens):
        if token not in seen:
            seen.add(token)
            positions.append(position)
    return positions


def dense_rank(query_key, pool_positions, positions_to_index, features, tokens,
               root_token, same_token, centered):
    eligible = [p for p in pool_positions if not same_token or tokens[p] == root_token]
    if not eligible:
        return [], []
    indices = torch.tensor([positions_to_index[p] for p in eligible], dtype=torch.long,
                           device=features.device)
    keys = features.index_select(0, indices)
    query = query_key
    if centered:
        # Fit on the eligible PAST memory only; never on the full trajectory.
        mean = keys.mean(dim=0)
        keys, query = keys - mean, query - mean
    similarities = F.normalize(keys, dim=-1) @ F.normalize(query, dim=-1)
    values = similarities.cpu().tolist()
    # Prefer the most recent donor for exact similarity ties.
    ranked = sorted(zip(eligible, values), key=lambda item: (-item[1], -item[0]))
    return [p for p, _ in ranked], [float(v) for _, v in ranked]


def replay_keys(tokens, prompt_length, prediction_rows, feature_positions,
                lagged_features, top_k, blocked_token_id, verify_production=False):
    """feature_positions[i]=s maps lagged_features[i] to h_{s-1}, not h_s."""
    if not 0 < prompt_length < len(tokens):
        raise ValueError("invalid prompt length")
    if (len(feature_positions) != len(set(feature_positions))
            or len(feature_positions) != lagged_features.shape[0]
            or set(feature_positions) != set(prediction_rows)
            or not torch.isfinite(lagged_features).all()):
        raise ValueError("features must align one-to-one with prediction rows")
    features = lagged_features.float()
    index = {p: i for i, p in enumerate(feature_positions)}
    banks = {order: {} for order in (1, 2, 3)}
    past_trigram_labels = defaultdict(set)
    for position in first_prompt_positions(tokens[:prompt_length]):
        if tokens[position] != blocked_token_id:
            banks[1][(tokens[position],)] = position
            if position >= 2:
                past_trigram_labels[tuple(tokens[position-2:position+1])].add(prediction_rows[position][0][0])
    states = []
    for position in range(prompt_length, len(tokens) - 1):
        lexical_sources = []
        for order in (3, 2, 1):
            if position + 1 >= order:
                key = tuple(tokens[position + 1 - order:position + 1])
                if key in banks[order]:
                    lexical_sources.append(banks[order][key])
        pool = sorted(set(p for bank in banks.values() for p in bank.values()))
        if any(p >= position for p in pool):
            raise ValueError("current/future prediction entered the common memory pool")
        query = features[index[position]]  # h_{position-1}, verified prefix only
        token_rank, token_sim = dense_rank(query, pool, index, features, tokens,
                                           tokens[position], True, False)
        all_rank, all_sim = dense_rank(query, pool, index, features, tokens,
                                       tokens[position], False, False)
        centered_rank, centered_sim = dense_rank(query, pool, index, features, tokens,
                                                 tokens[position], True, True)
        # Match the lexical fusion source-slot budget for token-filtered fusion.
        # Lexical slots can point to the same row; dense donors are distinct.
        slots = len(lexical_sources)
        selections = {
            "lexical_backoff": lexical_sources[:1],
            "lexical_fusion": lexical_sources,
            "dense_token_top1": token_rank[:1],
            "dense_token_fusion": token_rank[:slots],
            "dense_fusion": all_rank[:3],
            "dense_centered_token_fusion": centered_rank[:slots],
        }
        similarities = {
            "dense_token_top1": token_sim[:1], "dense_token_fusion": token_sim[:slots],
            "dense_fusion": all_sim[:3], "dense_centered_token_fusion": centered_sim[:slots],
        }
        methods = {}
        for mode, selected in selections.items():
            if not set(selected) <= set(pool):
                raise ValueError("retrieval escaped the shared active memory pool")
            candidates = fuse_rows(selected, prediction_rows, top_k, blocked_token_id)
            if len(candidates) > top_k or len(candidates) != len(set(candidates)):
                raise ValueError("invalid final candidate budget")
            methods[mode] = {
                "candidate_token_ids": candidates,
                "source_positions": selected,
                "source_key_positions": [p - 1 for p in selected],
                "num_source_slots": len(selected),
                "num_unique_source_rows": len(set(selected)),
                "source_similarities": similarities.get(mode),
            }
        if verify_production:
            from evaluation.eval_recap_memory_source import root_candidates
            production_banks = {
                order: {key: (p, *prediction_rows[p]) for key, p in bank.items()}
                for order, bank in banks.items()
            }
            for mode in ("backoff", "fusion"):
                expected = root_candidates(tokens, position, production_banks, mode,
                                           top_k, blocked_token_id)
                # At depth 1 only membership matters for recall.
                if set(expected) != set(methods["lexical_" + mode]["candidate_token_ids"]):
                    raise ValueError("lexical diagnostic differs from production root candidates")
        # Labels are read AFTER every candidate set has been fixed.
        reference = prediction_rows[position][0][0]
        native = tokens[position + 1]
        for result in methods.values():
            result.update(available=bool(result["candidate_token_ids"]),
                          hit=reference in result["candidate_token_ids"],
                          native_hit=native in result["candidate_token_ids"])
        # A prespecified diagnostic subgroup based on past prediction labels only.
        historical_aliasing = (position >= 2 and
                              len(past_trigram_labels[tuple(tokens[position-2:position+1])]) > 1)
        states.append({
            "position": position, "query_key_position": position - 1,
            "reference_token_id": reference, "native_token_id": native,
            "context_token_ids": tokens[max(0, position-2):position+1],
            "pool_positions": pool, "pool_size": len(pool),
            "same_token_pool_size": len(token_rank),
            "lexical_source_slot_budget": slots,
            "historical_trigram_aliasing": historical_aliasing,
            "modes": methods,
        })
        for order in (1, 2, 3):
            if position + 1 >= order and tokens[position] != blocked_token_id:
                banks[order][tuple(tokens[position+1-order:position+1])] = position
        if position >= 2 and tokens[position] != blocked_token_id:
            past_trigram_labels[tuple(tokens[position-2:position+1])].add(reference)
    return states


def summarize(records, planned, top_k, output_path):
    groups = defaultdict(list)
    for record in records:
        groups[record["dataset"]].append(record)
    datasets = {}
    for dataset, samples in groups.items():
        states = [s for sample in samples for s in sample["states"]]
        modes = {}
        for mode in MODES:
            available = sum(s["modes"][mode]["available"] for s in states)
            hits = sum(s["modes"][mode]["hit"] for s in states)
            modes[mode] = {
                "num_states": len(states), "num_available": available, "num_hits": hits,
                "query_coverage": available / len(states) if states else None,
                "recall_at_k_all_states": hits / len(states) if states else None,
                "conditional_recall_at_k": hits / available if available else None,
                "native_recall_at_k_all_states": np.mean([s["modes"][mode]["native_hit"] for s in states]).item() if states else None,
                "avg_candidate_count": np.mean([len(s["modes"][mode]["candidate_token_ids"]) for s in states]).item() if states else None,
                "avg_unique_source_rows": np.mean([s["modes"][mode]["num_unique_source_rows"] for s in states]).item() if states else None,
            }
        comparisons = {}
        pairs = [(mode, "lexical_fusion") for mode in MODES if mode not in ("lexical_fusion", "lexical_backoff")]
        pairs += [("dense_token_top1", "lexical_backoff")]
        for left, right in pairs:
            strata = {"all": states,
                      "jointly_covered": [s for s in states if s["modes"][left]["available"] and s["modes"][right]["available"]],
                      "historical_trigram_aliasing": [s for s in states if s["historical_trigram_aliasing"]]}
            comparison = {}
            for name, subset in strata.items():
                deltas, equal_deltas = [], []
                for s in subset:
                    a, b = (s["modes"][mode] for mode in (left, right))
                    deltas.append(int(a["hit"]) - int(b["hit"]))
                    budget = min(len(a["candidate_token_ids"]), len(b["candidate_token_ids"]))
                    if budget:
                        equal_deltas.append(int(s["reference_token_id"] in a["candidate_token_ids"][:budget])
                                            - int(s["reference_token_id"] in b["candidate_token_ids"][:budget]))
                comparison[name] = {
                    "num_states": len(subset),
                    "recall_delta": float(np.mean(deltas)) if deltas else None,
                    "equal_actual_budget_num_states": len(equal_deltas),
                    "equal_actual_budget_recall_delta": float(np.mean(equal_deltas)) if equal_deltas else None,
                }
            # Cluster sampling uncertainty is retained in raw artifacts, not paper tables.
            clusters = defaultdict(lambda: np.zeros(2, dtype=float))
            for sample in samples:
                for s in sample["states"]:
                    clusters[sample["target_image_identity"]] += (
                        int(s["modes"][left]["hit"]) - int(s["modes"][right]["hit"]), 1)
            values = np.array(list(clusters.values()))
            comparison["num_image_clusters"] = len(values)
            if len(values) > 1 and values[:, 1].sum() > 0:
                rng = np.random.default_rng(42)
                draws = values[rng.integers(len(values), size=(1000, len(values)))].sum(axis=1)
                valid = draws[:, 1] > 0
                comparison["paired_image_cluster_bootstrap_standard_error"] = float(
                    np.std(draws[valid, 0] / draws[valid, 1], ddof=1)) if valid.sum() > 1 else None
            comparisons[f"{left}_minus_{right}"] = comparison
        datasets[dataset] = {"num_records": len(samples), "num_states": len(states),
                             "modes": modes, "paired_comparisons": comparisons,
                             "reference_native_agreement_ratio": float(np.mean([
                                 s["reference_token_id"] == s["native_token_id"] for s in states])) if states else None}
    return {
        "protocol": PROTOCOL, "complete": len(records) == planned,
        "num_records": len(records), "planned_records": planned,
        "num_states": sum(len(r["states"]) for r in records), "top_k": top_k,
        "jsonl_path": str(output_path), "datasets": datasets,
        "diagnostic_only": True, "formal_speedup_evaluated": False,
        "accepted_prefix_length_evaluated": False,
        "query_key": "h[t-1]; source key h[s-1]; row at s predicts token s+1",
        "memory_scope": "cold request; identical active U/C/G row pool; no cross-request persistence",
        "aggregation": "state-weighted within each dataset; image-cluster paired sampling uncertainty",
        "new_gpu_generation": False,
        "development_only": True, "held_out_confirmation": False,
    }


@torch.inference_mode()
def extract_lagged_features(model, sequence, inputs, row_positions):
    """Reuse the established multimodal causal forward; capture only needed states."""
    from evaluation.selective_reuse_protocol import _selected_qwen_logits
    hidden_positions = sorted({p - 1 for p in row_positions if p > 0})
    captured = []

    def capture(_module, _arguments, output):
        hidden = output[0]
        indices = torch.tensor(hidden_positions, dtype=torch.long, device=hidden.device)
        captured.append(hidden[0].index_select(0, indices).detach())

    hook = model.base_model.model.register_forward_hook(capture)
    old_rope = getattr(model.base_model, "rope_deltas", None)
    try:
        # Only one vocabulary projection; cached prediction values stay unchanged.
        logits = _selected_qwen_logits(
            model.base_model, input_ids=sequence, attention_mask=torch.ones_like(sequence),
            pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
            logit_positions=torch.tensor([hidden_positions[-1]], device=sequence.device),
        )
        del logits
    finally:
        hook.remove()
        model.base_model.rope_deltas = old_rope
    if len(captured) != 1:
        raise ValueError("expected exactly one decoder forward")
    hidden = captured[0]
    by_position = dict(zip(hidden_positions, hidden.unbind(0)))
    empty_prefix = torch.zeros(hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
    return torch.stack([by_position[p-1] if p > 0 else empty_prefix for p in row_positions])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.inference_mode()
def evaluate(args):
    from evaluation import eval_recap_memory_source as source
    from evaluation.eval_recap_path_coverage import PROTOCOL as SOURCE_PROTOCOL

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    source_root, output_root = Path(args.source_root), Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("use a fresh output directory; no silent overwrite or resume")
    previous = json.loads((source_root / "run_config.json").read_text())
    previous_summary = json.loads((source_root / "summary.json").read_text())
    if previous["protocol"] != SOURCE_PROTOCOL or not previous_summary.get("complete"):
        raise ValueError("require a complete saved-row path-coverage run")
    original = previous["source_config"]
    source_jsonl = source_root / "results.jsonl"
    original_records = [json.loads(line) for line in source_jsonl.read_text().splitlines() if line.strip()]
    if len(original_records) != previous_summary["num_records"]:
        raise ValueError("source record count differs from summary")
    if len({r["question_id"] for r in original_records}) != len(original_records):
        raise ValueError("duplicate source question identity")
    payloads = []
    manifest_args = SimpleNamespace(manifest_dir=original["manifest_dir"], seed=original["seed"],
                                    sample_num=original["source_pool_size"], regenerate_manifest=False)
    for dataset in original["datasets"]:
        manifest_path = Path(original["manifest_dir"]) / f"{source.dataset_slug(dataset)}_seed{original['seed']}_n{original['source_pool_size']}.jsonl"
        if file_hash(manifest_path) != original["manifest_sha256"][dataset]:
            raise ValueError(f"source manifest changed: {dataset}")
        selected, manifest, _ = source.load_or_create_manifest(dataset, source.load_base_dataset(dataset), manifest_args)
        chosen = [r for r in original_records if r["dataset"] == dataset][:args.samples_per_benchmark]
        if len(chosen) != args.samples_per_benchmark:
            raise ValueError(f"insufficient source records: {dataset}")
        mapping = {f"{source.dataset_slug(dataset)}:{m['source_index']}": selected[i]
                   for i, m in enumerate(manifest)}
        payloads.append((dataset, mapping, chosen))
    planned = sum(len(x[2]) for x in payloads)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "features").mkdir()
    config = {
        **vars(args), "protocol": PROTOCOL, "source_config": original,
        "source_jsonl_sha256": file_hash(source_jsonl), "modes": MODES,
        "weights": WEIGHTS, "top_k": original["top_k"],
        "query_availability": "h[t-1] of verified prefix; not current root h[t]",
        "source_key": "h[s-1] paired with stored P(.|prefix through s); empty prefix uses zero vector",
        "pool": "deduplicated union of active U/C/G prediction rows, identical for all variants",
        "normalization": "L2 cosine; centered control subtracts mean of eligible past keys only",
        "source_slot_budget": "token-filtered fusion matches available lexical source slots; unrestricted dense fusion uses at most 3",
        "prediction_values": "unchanged cached target rows, renormalized on retained Top-K support",
        "not_evaluated": ["online throughput", "tree-path acceptance", "cross-request semantic memory", "DReSD replication"],
        "development_only": True, "held_out_confirmation": False,
        "code_sha256": {str(p): file_hash(p) for p in (
            Path(__file__), MMSPEC_ROOT / "evaluation/selective_reuse_protocol.py",
            MMSPEC_ROOT / "method/sam_grounded/tree_recycling_model.py",
            MMSPEC_ROOT.parent / "method/fovea/evaluation/eval_llava_fixed_multi.py")},
    }
    write_json(output_root / "run_config.json", config)
    print(f"protocol={PROTOCOL} planned_records={planned} output_root={output_root}", flush=True)
    print("loading_model=" + original["base_model_path"], flush=True)
    model = source.TreeRecyclingSpecModel.from_pretrained(
        base_model_path=original["base_model_path"], spec_model_path="", total_token=112,
        torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa")
    model.processor = source.load_processor_for_model(original["base_model_path"])
    model.eval()
    if model.base_model.config.model_type != "qwen2_5_vl":
        raise ValueError("this feature extractor currently supports the repository Qwen2.5-VL fork")
    records = []
    output_path = output_root / "results.jsonl"
    write_json(output_root / "summary.json", summarize(records, planned, original["top_k"], output_path))
    with output_path.open("x", encoding="utf-8") as handle:
        for dataset, data_by_id, chosen in payloads:
            for saved in chosen:
                started = time.perf_counter()
                tokens, prompt_length = saved["input_token_ids"], saved["prompt_length"]
                if hashlib.sha256(json.dumps(tokens[prompt_length:]).encode()).hexdigest() != saved["output_token_sha256"]:
                    raise ValueError("saved target trajectory integrity check failed")
                inputs = source.build_inputs_for_record(model, dataset, data_by_id[saved["question_id"]])
                if inputs["input_ids"][0].tolist() != tokens[:prompt_length]:
                    raise ValueError("current prompt differs from saved trajectory")
                prediction_rows = normalize_rows(saved["prediction_rows"], original["top_k"])
                positions = sorted(prediction_rows)
                sequence = torch.tensor([tokens], device=inputs["input_ids"].device)
                features = extract_lagged_features(model, sequence, inputs, positions)
                causal_check = None
                if args.check_prefix_causality and prompt_length in positions:
                    prefix_feature = extract_lagged_features(
                        model, sequence[:, :prompt_length], inputs, [prompt_length])[0]
                    full_feature = features[positions.index(prompt_length)]
                    cosine = float(F.cosine_similarity(prefix_feature.float(), full_feature.float(), dim=0))
                    if cosine < 0.999:
                        raise ValueError(f"prefix causality check failed: cosine={cosine}")
                    causal_check = {"prefix_length": prompt_length, "cosine_full_vs_truncated": cosine,
                                    "max_abs_difference": float((prefix_feature.float()-full_feature.float()).abs().max())}
                states = replay_keys(tokens, prompt_length, prediction_rows, positions, features,
                                     original["top_k"], saved["blocked_token_id"], args.verify_production)
                if len(states) != len(saved["states"]):
                    raise ValueError("query set changed relative to source replay")
                feature_path = output_root / "features" / f"{len(records):04d}-{saved['question_id'].replace(':', '-')}.pt"
                torch.save({
                    "question_id": saved["question_id"], "feature_positions": positions,
                    "lagged_hidden_states": features.cpu(), "input_token_ids": tokens,
                    "prompt_length": prompt_length, "prediction_rows": saved["prediction_rows"],
                    "blocked_token_id": saved["blocked_token_id"], "protocol": PROTOCOL,
                }, feature_path)
                record = {"protocol": PROTOCOL, "dataset": dataset, "question_id": saved["question_id"],
                          "target_image_identity": saved["target_image_identity"],
                          "num_generated_tokens": saved["num_generated_tokens"],
                          "num_states": len(states), "states": states,
                          "feature_path": str(feature_path), "feature_bytes": feature_path.stat().st_size,
                          "prefix_causality_check": causal_check,
                          "production_root_candidates_verified": args.verify_production,
                          "diagnostic_wall_seconds": time.perf_counter()-started}
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"record={len(records)}/{planned} dataset={dataset} question_id={saved['question_id']} states={len(states)} elapsed={record['diagnostic_wall_seconds']:.2f}s", flush=True)
                if len(records) % 10 == 0 or saved is chosen[-1]:
                    write_json(output_root / "summary.json", summarize(records, planned, original["top_k"], output_path))
                del features, sequence, inputs
    summary = summarize(records, planned, original["top_k"], output_path)
    if not summary["num_states"]:
        raise ValueError("no diagnostic states")
    write_json(output_root / "summary.json", summary)
    print(f"completed_summary={output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-benchmark", type=int, default=100)
    parser.add_argument("--check-prefix-causality", action="store_true")
    parser.add_argument("--verify-production", action="store_true")
    args = parser.parse_args()
    if args.samples_per_benchmark < 1:
        parser.error("samples-per-benchmark must be positive")
    evaluate(args)
