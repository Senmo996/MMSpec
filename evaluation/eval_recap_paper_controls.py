"""Frozen RECAP experiments 3/4/5: granularity, suffix, and request scope.

Uses the existing model, manifest loader, compact result format, and paired
summaries. These are follow-up controls on previously evaluated inputs, not a
new untouched test set. No method is selected from the results of this run.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for directory in (SCRIPT_DIR, SCRIPT_DIR.parent, PROJECT_ROOT):
    sys.path.insert(0, str(directory))

import torch
from evaluation.analyze_policy_pairs import compare
from evaluation.eval_sam_grounded_fixed_multi import (
    _image_cluster_identity, _summarize_dataset, _token_hash, _write_json,
    build_inputs_for_record, dataset_slug, load_base_dataset,
    load_or_create_manifest, load_processor_for_model, save_result,
)
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel

DATASETS = ("ChartQA", "TextVQA", "OCRBench", "MMT-Bench")
STRICT = "context-score-trigram-strict-depth10-node63-wide8"
FUSION = "context-score-trigram-fusion-depth10-node63-wide8"
UNIFORM = "context-score-trigram-fusion-uniform-depth10-node63-wide8"
PERSISTENT = "context-score-trigram-fusion-persistent-depth10-node63-wide8"
CONTROL_POLICIES = {
    "granularity": {
        "target": {"draft_policy": "target"},
        "U": {"draft_policy": STRICT, "recycling_max_order": 1},
        "UC": {"draft_policy": STRICT, "recycling_max_order": 2},
        "backoff": {"draft_policy": STRICT},
        "uniform": {"draft_policy": UNIFORM},
        "fusion": {"draft_policy": FUSION},
    },
    "suffix": {
        "target": {"draft_policy": "target"},
        "suffix_only": {"draft_policy": STRICT, "recycling_max_order": 0, "recycling_suffix_length": 4},
        "backoff": {"draft_policy": STRICT},
        "fusion": {"draft_policy": FUSION},
        "backoff_suffix": {"draft_policy": STRICT, "recycling_suffix_length": 4},
        "fusion_suffix": {"draft_policy": FUSION, "recycling_suffix_length": 4},
    },
    "scope": {
        "target": {"draft_policy": "target"},
        "reset": {"draft_policy": FUSION},
        "carry": {"draft_policy": PERSISTENT, "recycling_cache_namespace": "carry"},
        "task_reset": {"draft_policy": PERSISTENT, "recycling_cache_namespace": "task_reset"},
    },
}
COMPARISONS = {
    "granularity": (("UC", "U"), ("backoff", "UC"), ("fusion", "backoff"), ("uniform", "backoff"), ("fusion", "uniform")),
    "suffix": (("backoff_suffix", "backoff"), ("fusion_suffix", "fusion"), ("fusion_suffix", "backoff_suffix"), ("fusion_suffix", "suffix_only")),
    "scope": (("carry", "reset"), ("task_reset", "reset"), ("carry", "task_reset")),
}


def control_kwargs(experiment, name):
    return dict(
        matrix_top_k=8, tree_fixed_width=8, tree_fixed_depth=10,
        tree_broad_width=4, tree_shallow_depth=3, tree_node_budget=63,
        recycling_prompt_init="none" if experiment == "granularity" else "unigram",
        recycling_max_order=3, recycling_suffix_length=0,
        # Keep these frozen controls on their original suffix implementation.
        recycling_suffix_implementation="legacy",
        recycling_total_node_budget=63,
    ) | CONTROL_POLICIES[experiment][name]


def make_stream(samples, regime, seed):
    """Same inputs and within-task order; only the task interleaving changes."""
    buckets = {dataset: [s for s in samples if s["dataset"] == dataset] for dataset in DATASETS}
    rng = random.Random(seed)
    order = list(DATASETS)
    rng.shuffle(order)
    for dataset in DATASETS:
        rng.shuffle(buckets[dataset])
    if regime == "blocked":
        ordered = [s for dataset in order for s in buckets[dataset]]
    elif regime == "alternating":
        ordered = [buckets[dataset][i] for i in range(max(map(len, buckets.values())))
                   for dataset in order if i < len(buckets[dataset])]
    else:
        raise ValueError(f"unknown stream regime: {regime}")
    previous, age, episode = None, 0, -1
    result = []
    for position, sample in enumerate(ordered):
        switched = previous is not None and sample["dataset"] != previous
        if sample["dataset"] != previous:
            age, episode = 0, episode + 1
        result.append(sample | {"stream_position": position, "task_episode": episode,
                                "request_since_switch": age, "task_switch": switched})
        previous, age = sample["dataset"], age + 1
    return result


def recovery_bin(age):
    for upper, label in ((0, "1"), (4, "2-5"), (9, "6-10"), (24, "11-25")):
        if age <= upper:
            return label
    return "26+"


def shallow_stream_comparison(rows, left, right):
    """Descriptive only: requests sharing online memory are not IID draws."""
    ratios, matches, exact_ratios = [], [], []
    for row in rows:
        a, b, target = (row["policies"][name] for name in (left, right, "target"))
        ratio = (a["tokens"] / a["seconds"]) / (b["tokens"] / b["seconds"])
        ratios.append(ratio)
        matches.append(a["hash"] == b["hash"])
        if a["hash"] == b["hash"] == target["hash"]:
            exact_ratios.append(ratio)
    accepts = {name: [v for row in rows for v in row["policies"][name]["accepted"]] for name in (left, right)}
    return {"num_requests": len(rows), "mean_speed_ratio": statistics.fmean(ratios) if ratios else None,
            "median_speed_ratio": statistics.median(ratios) if ratios else None,
            "pair_output_match_ratio": statistics.fmean(matches) if matches else None,
            "both_match_target_count": len(exact_ratios),
            "exact_subset_mean_speed_ratio": statistics.fmean(exact_ratios) if exact_ratios else None,
            "mean_accept_length": {name: statistics.fmean(values) if values else None for name, values in accepts.items()},
            "mean_persistent_rows_at_start": {
                name: statistics.fmean(row["policies"][name]["counters"].get("persistent_rows_at_start", 0) for row in rows)
                if rows else None for name in (left, right)
            }}


def memory_bytes(value, seen=None):
    """Python object footprint of the actual host cache, excluding the model."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(memory_bytes(k, seen) + memory_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        total += sum(memory_bytes(v, seen) for v in value)
    return total


def run_online(model, samples, root, experiment, manifests, seed=42, regime="blocked"):
    policies = list(CONTROL_POLICIES[experiment])
    stream = make_stream(samples, regime, seed)
    root.mkdir(parents=True, exist_ok=False)
    sequence = [{k: s[k] for k in ("id", "dataset", "source_index", "stream_position",
                                 "task_episode", "request_since_switch", "task_switch")} for s in stream]
    _write_json(root / "stream.json", sequence)
    _write_json(root / "run_config.json", {"experiment": experiment, "seed": seed, "regime": regime,
        "controls": {name: control_kwargs(experiment, name) for name in policies},
        "max_new_tokens": 200, "tree_builder": "shared Python score-priority",
        "suffix_in_total_candidate_cap": True, "online_budget": "equal cap 63, real node counts saved; no fake padding",
        "first_token_from_prefill": True, "compact": True, "per_forward_timing_hooks": False,
        "lightweight_cpu_counters_timed": True, "decoded_text": False,
        "followup_on_previously_evaluated_inputs": True, "untouched_test_set": False,
        "output_equivalence_not_assumed": True, "scope_task_proxy": "benchmark identity, not semantic topic labels"})
    for dataset in DATASETS:
        for name in policies:
            (root / dataset_slug(dataset) / name).mkdir(parents=True)
    warm_inputs = build_inputs_for_record(model, stream[0]["dataset"], stream[0]["row"])
    for name in policies:
        torch.manual_seed(0)
        model.specgenerate(**warm_inputs, temperature=0, max_new_tokens=32,
                           **control_kwargs(experiment, name))
    model.reset_persistent_recycling_cache()
    print(f"stage=online experiment={experiment} regime={regime} seed={seed} requests={len(stream)} warmup_done=true", flush=True)
    records, memory_snapshots = [], []
    for sample in stream:
        if experiment != "scope":
            model.reset_persistent_recycling_cache()
        inputs = build_inputs_for_record(model, sample["dataset"], sample["row"])
        prompt_length = int(inputs.input_ids.shape[1])
        offset = sample["stream_position"] % len(policies)
        order = policies[offset:] + policies[:offset]
        compact = {k: sample[k] for k in ("id", "dataset", "stream_position", "task_episode", "request_since_switch")}
        compact["policies"] = {}
        for name in order:
            counters = {}
            torch.manual_seed(0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if experiment == "scope" and name == "task_reset" and sample["task_switch"]:
                model._gwtr_persistent_unigram_caches.pop("task_reset", None)
            output, n_new, iterations, accepted = model.specgenerate(
                **inputs, temperature=0, max_new_tokens=200, log=True,
                return_acceptance_len=True, return_policy_trace=False,
                recycling_stats=counters, **control_kwargs(experiment, name),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens = output[0, prompt_length:].cpu().tolist()
            output_hash = _token_hash(output, prompt_length)
            if not tokens or n_new != len(tokens) or len(tokens) > 200:
                raise RuntimeError("invalid generated token count")
            if counters.get("max_candidate_nodes", 0) > 63:
                raise RuntimeError("online candidate budget exceeded")
            if name == "task_reset" and sample["task_switch"] and counters.get("persistent_rows_at_start", 0):
                raise RuntimeError("task boundary did not clear its cache namespace")
            if experiment == "granularity" and counters.get("prompt_rows", 0):
                raise RuntimeError("unexpected prompt seeding in clean-initialization control")
            if name == "suffix_only" and (counters.get("memory_rows_at_end", {}).get("U", 0) or counters.get("max_candidate_nodes", 0) > 4):
                raise RuntimeError("suffix-only control used transition candidates")
            choice = {"index": 0, "output_hashes": [output_hash], "output_token_ids": [tokens],
                      "idxs": [int(iterations)], "new_tokens": [int(n_new)], "wall_time": [elapsed],
                      "acceptance_length": [accepted], "recap_counters": counters,
                      "stream_position": sample["stream_position"], "task_episode": sample["task_episode"],
                      "request_since_switch": sample["request_since_switch"]}
            save_result(str(root / dataset_slug(sample["dataset"]) / name / "results.jsonl"),
                        {"id": sample["id"], "topic": sample["dataset"], "source_index": sample["source_index"],
                         "image_cluster_id": sample["image_cluster_id"]}, f"recap-control-{name}", [choice])
            compact["policies"][name] = {"tokens": int(n_new), "seconds": elapsed, "hash": output_hash,
                                          "accepted": accepted, "counters": counters}
        records.append(compact)
        if experiment == "scope" and (sample["task_switch"] or sample["stream_position"] + 1 == len(stream)):
            memory_snapshots.append({"stream_position": sample["stream_position"], "caches": {
                name: {"rows": len(cache["transitions"]), "python_object_bytes": memory_bytes(cache)}
                for name, cache in getattr(model, "_gwtr_persistent_unigram_caches", {}).items()
            }})
        print(f"experiment={experiment} regime={regime} seed={seed} request={len(records)}/{len(stream)} dataset={sample['dataset']}", flush=True)
    summaries, paired = {}, {}
    for dataset in DATASETS:
        ds_root = root / dataset_slug(dataset)
        summaries[dataset] = _summarize_dataset(dataset, ds_root, policies, manifests[dataset])
        expected = sum(s["dataset"] == dataset for s in samples)
        for name, summary in summaries[dataset].items():
            if summary["num_records"] != expected:
                raise RuntimeError("incomplete online summary")
            counters = [r["policies"][name]["counters"] for r in records if r["dataset"] == dataset]
            summary["observed_candidate_nodes"] = {
                "total": sum(c.get("total_candidate_nodes", 0) for c in counters),
                "max": max((c.get("max_candidate_nodes", 0) for c in counters), default=0),
                "iterations": sum(c.get("iterations", 0) for c in counters),
                "note": "CPU counters; legacy trace-derived node fields are unmeasured in compact mode",
            }
            _write_json(ds_root / name / "summary.json", summary)
        paired[dataset] = {}
        for left, right in COMPARISONS[experiment]:
            if experiment == "scope":
                result = shallow_stream_comparison([r for r in records if r["dataset"] == dataset], left, right)
            else:
                result = compare(ds_root / left / "results.jsonl", ds_root / right / "results.jsonl",
                                 bootstrap_samples=10000, exact_reference_path=ds_root / "target/results.jsonl")
            paired[dataset][f"{left}_over_{right}"] = result
        _write_json(ds_root / "paired.json", paired[dataset])
    recovery = {}
    if experiment == "scope":
        grouped = defaultdict(list)
        for row in records:
            # Episode zero is the genuinely cold block, not recovery after a switch.
            if row["task_episode"] > 0:
                grouped[recovery_bin(row["request_since_switch"])].append(row)
        recovery = {key: {f"{a}_over_{b}": shallow_stream_comparison(rows, a, b)
                          for a, b in COMPARISONS["scope"]} for key, rows in grouped.items()}
    payload = {"complete": True, "experiment": experiment, "num_inputs": len(samples), "seed": seed,
               "regime": regime, "datasets": summaries, "paired": paired, "recovery": recovery,
               "memory_snapshots": memory_snapshots,
               "scope_uncertainty": "report variation across three order seeds; no IID request bootstrap with shared memory"}
    _write_json(root / "summary.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--cached-source", required=True)
    parser.add_argument("--base-model-path", default=str(PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--scope-samples-per-dataset", type=int, default=50)
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "protocol.json").exists() or not 1 <= args.scope_samples_per_dataset <= 100:
        raise ValueError("require fresh output and scope sample count in [1, 100]")
    source_paths = [Path(__file__), SCRIPT_DIR / "eval_recap_path_coverage.py",
                    SCRIPT_DIR.parent / "method/sam_grounded/tree_recycling_model.py",
                    SCRIPT_DIR.parent / "method/sam_grounded/controller.py",
                    SCRIPT_DIR / "analyze_policy_pairs.py", SCRIPT_DIR / "summarize_training_free.py",
                    PROJECT_ROOT / "method/fovea/evaluation/eval_llava_fixed_multi.py",
                    PROJECT_ROOT / "scripts/launch_recap_paper_controls.sh",
                    SCRIPT_DIR.parent / "method/sam/samd/sam_draft.py",
                    SCRIPT_DIR.parent / "method/sam/samd/dyn_sam.py"]
    manifest_paths = {d: Path(args.manifest_dir) / f"{dataset_slug(d)}_seed42_n100.jsonl" for d in DATASETS}
    source_paths += list(manifest_paths.values())
    protocol = {"protocol": "recap_paper_controls_345_v1", "arguments": vars(args),
                "scope_order_seeds": [17, 29, 43], "scope_regimes": ["blocked", "alternating"],
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "data_role": "mechanism follow-up; previously evaluated inputs; not final untouched confirmation",
                "known_caveat": "native/packed token equality unresolved; no lossless-speedup claim",
                "default_method_unchanged": True}
    _write_json(root / "protocol.json", protocol)
    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')} output_root={root} protocol={protocol['protocol']}", flush=True)
    for experiment, modes, initialization in (
        ("granularity", "U,UC,backoff,uniform,fusion", "none"),
        ("suffix", "backoff,fusion,backoff_suffix,fusion_suffix", "unigram"),
    ):
        print(f"stage=offline_{experiment}", flush=True)
        subprocess.run([sys.executable, str(SCRIPT_DIR / "eval_recap_path_coverage.py"),
                        "--source-root", args.cached_source, "--output-root", str(root / f"offline_{experiment}"),
                        "--cached-rows", "--score-normalization", "topk", "--memory-init", initialization,
                        "--modes", modes, "--samples-per-benchmark", "100"], check=True)
        summary = json.loads((root / f"offline_{experiment}" / "summary.json").read_text())
        if not summary.get("complete") or summary["num_records"] != 400:
            raise RuntimeError("incomplete offline controls")
    samples, manifests = [], {}
    manifest_args = SimpleNamespace(manifest_dir=args.manifest_dir, seed=42, sample_num=100, regenerate_manifest=False)
    for dataset in DATASETS:
        selected, manifest, path = load_or_create_manifest(dataset, load_base_dataset(dataset), manifest_args)
        if len(selected) != 100 or len(manifest) != 100:
            raise RuntimeError("require 100 manifest inputs per dataset")
        manifests[dataset] = path
        for index, meta in enumerate(manifest):
            row = selected[index]
            samples.append({"id": f"{dataset_slug(dataset)}:{index:04d}", "dataset": dataset, "row": row,
                            "source_index": int(meta["source_index"]), "sample_position": index,
                            "image_cluster_id": _image_cluster_identity(dataset, row, meta)})
    print(f"stage=load_model CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    torch.set_num_threads(4)
    model = TreeRecyclingSpecModel.from_pretrained(base_model_path=args.base_model_path, spec_model_path="",
              total_token=112, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa")
    model.processor = load_processor_for_model(args.base_model_path)
    model.eval()
    completed = []
    for experiment in ("granularity", "suffix"):
        run_online(model, samples, root / f"online_{experiment}", experiment, manifests)
        completed.append(f"online_{experiment}")
    scope_samples = [s for s in samples if s["sample_position"] < args.scope_samples_per_dataset]
    scope_runs = []
    for seed in (17, 29, 43):
        for regime in ("blocked", "alternating"):
            name = f"scope_{regime}_seed{seed}"
            result = run_online(model, scope_samples, root / name, "scope", manifests, seed, regime)
            scope_runs.append({"seed": seed, "regime": regime, "paired": result["paired"], "recovery": result["recovery"]})
            completed.append(name)
    aggregate = {}
    for regime in ("blocked", "alternating"):
        aggregate[regime] = {}
        for dataset in DATASETS:
            aggregate[regime][dataset] = {}
            for left, right in COMPARISONS["scope"]:
                key = f"{left}_over_{right}"
                values = [r["paired"][dataset][key]["mean_speed_ratio"] for r in scope_runs if r["regime"] == regime]
                aggregate[regime][dataset][key] = {"order_seed_values": values,
                    "mean_over_order_seeds": statistics.fmean(values), "stdev_over_order_seeds": statistics.stdev(values)}
    _write_json(root / "scope_repeats.json", {
        "unique_inputs": len(scope_samples), "runs": scope_runs,
        "aggregate": aggregate,
        "note": "same inputs recur across six streams; do not count them as independent new samples",
    })
    _write_json(root / "summary.json", {"complete": True, "completed_online_runs": completed,
        "offline_runs": ["offline_granularity", "offline_suffix"], "online_granularity_inputs": 400,
        "online_suffix_inputs": 400, "scope_unique_inputs": len(scope_samples), "scope_streams": 6,
        "protocol": protocol["protocol"], "lossless_claim": False})
    print(f"stage=finished summary={root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
