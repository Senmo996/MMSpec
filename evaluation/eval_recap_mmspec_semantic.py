"""Compact paired MMSpec pilot of root-only semantic/recent-row fallback."""

import argparse
import atexit
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import threading

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "MMSpec"))
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel

POLICY = "context-score-trigram-fusion-persistent-suffix4-hotpath-cpp-depth10-node63-wide-plus4"
VARIANTS = ("target", "original", "semantic_root", "recent_root")
SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def select_samples(path, per_topic, selection_seed=42, order_seed=17):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[row["topic"]].append(row)
    chosen = []
    rng = random.Random(selection_seed)
    for topic in sorted(groups):
        rows = groups[topic][:]
        rng.shuffle(rows)
        if len(rows) < per_topic:
            raise ValueError(f"{topic}: need {per_topic}, found {len(rows)}")
        chosen.extend(rows[:per_topic])
    random.Random(order_seed).shuffle(chosen)
    return chosen


def options(variant):
    if variant == "target":
        return {"draft_policy": "target"}
    return dict(draft_policy=POLICY, matrix_top_k=8, tree_fixed_width=8,
                tree_fixed_depth=10, tree_broad_width=4, tree_shallow_depth=3,
                tree_node_budget=63, recycling_prompt_init="unigram", recycling_max_order=3,
                recycling_suffix_length=4, recycling_suffix_implementation="fixed",
                recycling_suffix_bridge=False, recycling_sam_clone_mode="flat",
                recycling_total_node_budget=63, recycling_cache_namespace=variant,
                recycling_root_backoff={"original": "off", "semantic_root": "cosine", "recent_root": "recent"}[variant],
                recycling_root_backoff_capacity=2048, recycling_root_backoff_neighbors=3)


def first_prompt(row, processor, image_dir):
    messages = [dict(role="system", content=[dict(type="text", text=SYSTEM)]),
                dict(role="user", content=[dict(type="image"), dict(type="text", text=row["turns"][0])])]
    with Image.open(image_dir / row["image"]) as image:
        inputs = processor(images=[image.convert("RGB")],
                           text=processor.apply_chat_template(messages, add_generation_prompt=True),
                           return_tensors="pt")
    return inputs.to("cuda:0")


def next_turn(inputs, generated_ids, question, tokenizer):
    """Carry target-reference history as IDs: no generated-text decoding.

    Qwen ChatML delimiters are checked when loading the tokenizer. Each variant
    sees exactly the same target-reference history, including on multi-turn
    samples; generation and image encoding are still repeated inside each timer.
    """
    special = set(tokenizer.all_special_ids)
    reply = [int(token) for token in generated_ids if int(token) not in special]
    suffix = tokenizer.encode("<|im_end|>\n<|im_start|>user\n" + question +
                              "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    result = dict(inputs)
    append = torch.tensor([reply + suffix], device=inputs["input_ids"].device, dtype=torch.long)
    result["input_ids"] = torch.cat([inputs["input_ids"], append], dim=1)
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    return result


def read_gpu_processes(gpu_uuid):
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                    "--format=csv,noheader,nounits"], text=True)
    return {int(parts[1]): int(parts[2]) for line in text.splitlines()
            if len(parts := [s.strip() for s in line.split(",")]) == 3
            and parts[0] == gpu_uuid and parts[2].isdigit()}


class GPUActivityMonitor:
    """Log per-process utilization and reject activity by other GPU users."""
    def __init__(self, gpu_id, log_path):
        self.violations = []
        self.measuring = False
        self.process = subprocess.Popen(["nvidia-smi", "pmon", "-i", str(gpu_id), "-s", "u", "-d", "1"],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        def collect():
            with log_path.open("w") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    fields = line.split()
                    if (self.measuring and len(fields) >= 5 and fields[1].isdigit()
                            and fields[2] == "C" and int(fields[1]) != os.getpid()
                            and any(v.isdigit() and int(v) > 0 for v in fields[3:5])):
                        self.violations.append(line.strip())
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.thread.join(timeout=2)


def run_turn(model, inputs, variant, max_tokens):
    counters = {}
    torch.cuda.synchronize()
    started = time.perf_counter()
    output, n_new, _, accepted = model.specgenerate(
        **inputs, temperature=0, max_new_tokens=max_tokens, log=True,
        return_acceptance_len=True, return_policy_trace=False, recycling_stats=counters,
        **options(variant))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
    if n_new != len(tokens) or not n_new:
        raise RuntimeError("invalid generated-token count")
    if counters.get("max_candidate_nodes", 0) > 63:
        raise RuntimeError("candidate budget exceeded")
    return dict(num_tokens=n_new, seconds=elapsed, tokens_per_second=n_new / elapsed,
                accepted_lengths=[int(n) for n in accepted],
                average_accept_length=float(np.mean(accepted)) if accepted else 0.,
                average_tokens_per_iteration=n_new / (len(accepted) + 1),
                verification_iterations=len(accepted), output_token_ids=tokens,
                output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                counters=counters)


def summarize(rows, path):
    accepted = [n for row in rows for n in row["accepted_lengths"]]
    per_request_accept = [row["average_accept_length"] for row in rows]
    speeds = [row["speedup"] for row in rows]
    iterations = sum(row["verification_iterations"] for row in rows)
    misses = sum(row["counters"].get("root_lexical_misses", 0) for row in rows)
    triggers = sum(row["counters"].get("root_backoff_triggers", 0) for row in rows)
    total_tokens = sum(row["num_tokens"] for row in rows)
    wall = sum(row["seconds"] for row in rows)
    mean = lambda x: float(np.mean(x)) if len(x) else 0.
    return dict(num_records=len(rows), num_requests=len({r["sample_id"] for r in rows}),
                avg_accept_length=mean(accepted), median_accept_length=float(np.median(accepted)) if accepted else 0.,
                max_accept_length=max(accepted, default=0), p90_accept_length=float(np.percentile(accepted, 90)) if accepted else 0.,
                avg_tokens_per_iteration=mean([r["average_tokens_per_iteration"] for r in rows]),
                avg_sample_speedup=mean(speeds), median_sample_speedup=float(np.median(speeds)), max_sample_speedup=max(speeds),
                sample_speedup_gt_1_ratio=mean([s > 1 for s in speeds]),
                accept_le_0_5_ratio=mean([n <= .5 for n in per_request_accept]), jsonl_path=str(path),
                tau=1. + mean(accepted), sample_mean_tau=1. + mean(per_request_accept),
                total_tokens=total_tokens, total_seconds=wall, pooled_tokens_per_second=total_tokens / wall,
                pooled_speedup=(total_tokens / wall) / (sum(r["target_num_tokens"] for r in rows) / sum(r["target_seconds"] for r in rows)),
                mean_speed_relative_original=mean([r["relative_original_speed"] for r in rows]),
                target_equal_ratio=mean([r["matches_target"] for r in rows]),
                root_lexical_misses=misses, root_lexical_miss_rate=misses / max(iterations, 1),
                root_backoff_triggers=triggers, root_backoff_trigger_rate=triggers / max(iterations, 1),
                backoff_positive_rate=sum(r["counters"].get("root_backoff_positive_iterations", 0) for r in rows) / max(triggers, 1),
                backoff_index_peak_rows=max((r["counters"].get("backoff_index_peak_rows", 0) for r in rows), default=0),
                backoff_index_evicted=sum(r["counters"].get("backoff_index_evicted", 0) for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, default=PROJECT / "MMSpec/dataset/MMSpec/test")
    parser.add_argument("--model-path", type=Path, default=PROJECT / "models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--samples-per-topic", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--order-seed", type=int, default=17)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--allowed-resident-pids", default="")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if (root / "config.json").exists():
        raise RuntimeError("refusing to overwrite an existing experiment")
    root.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), PROJECT / "MMSpec/method/sam_grounded/tree_recycling_model.py",
                    PROJECT / "MMSpec/method/sam_grounded/root_semantic_backoff.py",
                    PROJECT / "MMSpec/method/sam_grounded/fast_tree_builder.cpp"]
    source_hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    samples = select_samples(args.data_folder / "mmspec.jsonl", args.samples_per_topic, args.selection_seed, args.order_seed)
    write_json(root / "manifest.json", samples)
    config = dict(protocol="mmspec_root_backoff_online_v1", dataset="MMSpec", model=str(args.model_path),
                  samples_per_topic=args.samples_per_topic, num_samples=len(samples),
                  expected_turns=sum(len(r["turns"]) for r in samples),
                  selection_seed=args.selection_seed, order_seed=args.order_seed,
                  max_new_tokens=args.max_new_tokens, variants={v: options(v) for v in VARIANTS},
                  history="shared target-reference token IDs, no decode/re-encode", precision="model default",
                  timing="synchronized generation wall time including visual encoding, prefill, index maintenance and decode; no timing hooks",
                  semantic_pool="current-request active U/C/G row union; at most 2048 rows; FIFO eviction",
                  semantic_query="last verified predecessor state h[t-1]",
                  semantic_source="predecessor h[s-1], using actual topology parents for packed verification",
                  source_sha256=source_hashes, gpu_uuid=args.gpu_uuid)
    write_json(root / "config.json", config)
    print(json.dumps({"stage": "loading_model", "num_samples": len(samples), "expected_turns": config["expected_turns"]}), flush=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), use_fast=True,
                                              min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    tokenizer = processor.tokenizer
    for delimiter in ("<|im_end|>", "<|im_start|>"):
        if len(tokenizer.encode(delimiter, add_special_tokens=False)) != 1:
            raise RuntimeError("this compact multi-turn protocol requires Qwen ChatML")
    torch.set_num_threads(1)
    model = TreeRecyclingSpecModel.from_pretrained(base_model_path=str(args.model_path), spec_model_path=None,
        total_token=64, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa").eval()
    if model.base_model.config.architectures[0] != "Qwen2_5_VLForConditionalGeneration":
        raise RuntimeError("the initial pilot is Qwen2.5-VL only")
    config["actual_dtype"] = str(model.base_model.dtype)
    write_json(root / "config.json", config)
    allowed = {int(v) for v in args.allowed_resident_pids.split(",") if v} | {os.getpid()}
    warm_inputs = first_prompt(samples[0], processor, args.data_folder / "images")
    print("stage=warmup", flush=True)
    monitor = GPUActivityMonitor(args.gpu_id, root / (time.strftime("%Y%m%d-%H%M%S") + "-gpu-activity.log"))
    for _ in range(2):
        for variant in VARIANTS:
            run_turn(model, warm_inputs, variant, 24)
    model.reset_persistent_recycling_cache()
    all_rows = {v: [] for v in VARIANTS}
    paths = {v: root / v / "results.jsonl" for v in VARIANTS}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    print("stage=evaluation warmup_caches_cleared=true", flush=True)
    monitor.measuring = True
    turn_counter = 0
    for position, sample in enumerate(samples):
        inputs = first_prompt(sample, processor, args.data_folder / "images")
        for turn, question in enumerate(sample["turns"]):
            if turn:
                inputs = next_turn(inputs, results["target"]["output_token_ids"], question, tokenizer)
            processes = read_gpu_processes(args.gpu_uuid)
            unexpected = set(processes) - allowed
            if unexpected or monitor.violations:
                write_json(root / "gpu_interference.json", dict(position=position, turn=turn, processes=processes))
                raise RuntimeError(f"unexpected GPU processes {sorted(unexpected)}; stopped before timing")
            offset = turn_counter % len(VARIANTS)
            order = VARIANTS[offset:] + VARIANTS[:offset]
            results = {}
            for variant in order:
                results[variant] = run_turn(model, inputs, variant, args.max_new_tokens)
            after = read_gpu_processes(args.gpu_uuid)
            if set(after) - allowed or monitor.violations:
                write_json(root / "gpu_interference.json", dict(position=position, turn=turn, processes=after, activity=monitor.violations))
                raise RuntimeError("GPU interference during paired timing; preserve partial outputs")
            for variant, result in results.items():
                result.update(sample_id=sample["id"], topic=sample["topic"], turn=turn, position=position,
                              prompt_tokens=inputs["input_ids"].shape[1], policy_order=list(order),
                              speedup=result["tokens_per_second"] / results["target"]["tokens_per_second"],
                              relative_original_speed=result["tokens_per_second"] / results["original"]["tokens_per_second"],
                              matches_target=result["output_token_ids"] == results["target"]["output_token_ids"],
                              target_seconds=results["target"]["seconds"], target_num_tokens=results["target"]["num_tokens"])
                with paths[variant].open("a") as handle:
                    handle.write(json.dumps(result) + "\n")
                all_rows[variant].append(result)
            turn_counter += 1
            print(json.dumps(dict(sample=f"{position + 1}/{len(samples)}", turn=turn,
                topic=sample["topic"], metrics={v: dict(tau=round(results[v]["average_accept_length"] + 1, 3),
                speedup=round(results[v]["speedup"], 3), triggers=results[v]["counters"].get("root_backoff_triggers", 0)) for v in VARIANTS})), flush=True)
    monitor.close()
    if monitor.violations:
        write_json(root / "gpu_interference.json", dict(activity=monitor.violations))
        raise RuntimeError("GPU interference at end of timing")
    summaries = {}
    for variant, rows in all_rows.items():
        summary = summarize(rows, paths[variant])
        summary["topics"] = {topic: summarize([r for r in rows if r["topic"] == topic], paths[variant])
                             for topic in sorted({r["topic"] for r in rows})}
        summary["macro_topic_speedup"] = statistics.mean(s["avg_sample_speedup"] for s in summary["topics"].values())
        summary["macro_topic_tau"] = statistics.mean(s["tau"] for s in summary["topics"].values())
        write_json(paths[variant].with_name("summary.json"), summary)
        summaries[variant] = summary
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == source_hashes[str(p.relative_to(PROJECT))] for p in source_paths)
    complete = turn_counter == config["expected_turns"] and all(len(rows) == turn_counter for rows in all_rows.values())
    write_json(root / "summary.json", dict(complete=complete, sources_unchanged=unchanged,
               num_samples=len(samples), num_turns=turn_counter, variants=summaries))
    if not complete or not unchanged:
        raise RuntimeError("incomplete records or code changed during evaluation")
    print(json.dumps(dict(stage="finished", num_turns=turn_counter,
        variants={v: {k: s[k] for k in ("tau", "pooled_speedup", "avg_sample_speedup", "root_backoff_trigger_rate")} for v, s in summaries.items()})), flush=True)


if __name__ == "__main__":
    main()
