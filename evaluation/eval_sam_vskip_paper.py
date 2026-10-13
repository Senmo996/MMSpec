"""Paired AR / SAM+ViSkip evaluation on the current RECAP paper benchmarks."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "MMSpec"))
from method.sam_vskip.paper_model import PaperSAMViSkip
from method.fovea.evaluation.eval_llava_fixed_multi import (
    build_inputs_for_record, dataset_slug, load_base_dataset,
    load_processor_for_model, is_qwen25_vl_model_path,
)

FIXED_DATASETS = ("MMT-Bench", "SEEDBench", "ScienceQA", "OCRBench",
                  "ChartQA", "MathVista", "TextVQA")
POLICIES = ("ar", "sam_vskip")
SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return statistics.fmean(values) if values else 0.


def percentile(values, percent):
    if not values:
        return 0.
    values = sorted(values)
    position = (len(values) - 1) * percent / 100.
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows, path, expected):
    acceptance = [row["average_accept_length"] for row in rows]
    speeds = [row["speedup"] for row in rows]
    probes = sum(row["counters"]["visual_probes"] for row in rows)
    hits = sum(row["counters"]["visual_gate_hits"] for row in rows)
    total_time = sum(row["wall_time"] for row in rows)
    return dict(
        complete=len(rows) == expected, num_records=len(rows), expected_records=expected,
        num_inputs=len({row["sample_id"] for row in rows}), num_turns=len(rows),
        avg_accept_length=mean(acceptance),
        median_accept_length=statistics.median(acceptance) if acceptance else 0.,
        max_accept_length=max(acceptance, default=0.), p90_accept_length=percentile(acceptance, 90),
        avg_tokens_per_iteration=mean([row["average_tokens_per_iteration"] for row in rows]),
        avg_sample_speedup=mean(speeds),
        median_sample_speedup=statistics.median(speeds) if speeds else 0.,
        max_sample_speedup=max(speeds, default=0.),
        sample_speedup_gt_1_ratio=mean([speed > 1. for speed in speeds]),
        accept_le_0_5_ratio=mean([value <= .5 for value in acceptance]),
        tau=1. + mean(acceptance),
        acceptance_aggregation="mean of per-generation-turn mean accepted draft lengths; prefill excluded",
        total_new_tokens=sum(row["new_tokens"] for row in rows),
        total_wall_time=total_time,
        tokens_per_second=sum(row["new_tokens"] for row in rows) / total_time if total_time else 0.,
        output_hash_match_ratio=mean([row["matches_ar"] for row in rows]),
        visual_probes=probes, visual_gate_hits=hits,
        visual_gate_hit_ratio=hits / probes if probes else 0.,
        jsonl_path=str(Path(path).resolve()),
    )


def mmspec_messages(sample, turn, markers):
    messages = [dict(role="system", content=[dict(type="text", text=SYSTEM)])]
    for index in range(turn + 1):
        content = ([dict(type="image")] if index == 0 else [])
        content.append(dict(type="text", text=sample["turns"][index]))
        messages.append(dict(role="user", content=content))
        if index < turn:
            messages.append(dict(role="assistant", content=[dict(type="text", text=markers[index])]))
    return messages


def build_mmspec_inputs(model, sample, turn, history, data_folder):
    """Render the existing chat template without decoding generated token IDs.

Each policy carries its own committed history, as in the existing full MMSpec
evaluation. Template separators are tokenized separately; generated tokens are
inserted directly. The first-turn prompt is identical to evaluation.utils.
"""
    if len(history) != turn:
        raise ValueError("history/turn mismatch")
    markers = [f"SAM_VSKIP_HISTORY_PLACEHOLDER_{index}_B7A9" for index in range(turn)]
    rendered = model.processor.apply_chat_template(
        mmspec_messages(sample, turn, markers), add_generation_prompt=True)
    parts = re.split("(" + "|".join(map(re.escape, markers)) + ")", rendered) if markers else [rendered]
    if markers and [part for part in parts if part in markers] != markers:
        raise RuntimeError("chat template did not preserve unique history markers")
    with Image.open(data_folder / "images" / sample["image"]) as source:
        inputs = model.processor(images=[source.convert("RGB")], text=parts[0],
                                 return_tensors="pt").to(model.base_model.device)
    if turn:
        ids = inputs["input_ids"][0].tolist()
        tokenizer = model.processor.tokenizer
        special = set(tokenizer.all_special_ids)
        for index in range(turn):
            ids.extend(token for token in history[index] if token not in special)
            ids.extend(tokenizer.encode(parts[2 * index + 2], add_special_tokens=False))
        inputs["input_ids"] = torch.tensor([ids], dtype=torch.long, device=model.base_model.device)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def run_generation(model, inputs, policy, args, token_limit=None):
    limit = token_limit or args.max_new_tokens
    # Allocate a common reusable workspace outside generation timing, as in a
    # warmed-up server. Vision encoding, prefill, SAM maintenance and gates ARE timed.
    model.prepare_cache(inputs["input_ids"].shape[1] + limit + args.max_draft_tokens + 1)
    torch.manual_seed(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output, n_new, accepted, counters = model.paper_generate(
        policy, **inputs, max_new_tokens=limit,
        max_draft_tokens=args.max_draft_tokens,
        visual_attn_threshold=args.visual_attn_threshold)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    ids = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
    if len(ids) != n_new or not (0 < n_new <= limit):
        raise RuntimeError(f"invalid generation length for {policy}: {n_new}/{len(ids)}")
    return dict(new_tokens=n_new, wall_time=elapsed, tokens_per_second=n_new / elapsed,
                average_accept_length=mean(accepted), accepted_lengths=accepted,
                average_tokens_per_iteration=n_new / (counters["verification_iterations"] + 1),
                output_token_ids=ids,
                output_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                input_tokens=int(inputs["input_ids"].shape[1]), counters=counters)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT / "result/manifests/dream_baseline_seed42")
    parser.add_argument("--data-folder", type=Path, default=PROJECT / "MMSpec/dataset/MMSpec/test")
    parser.add_argument("--datasets", default=",".join(FIXED_DATASETS + ("MMSpec",)))
    parser.add_argument("--sample-num", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-draft-tokens", type=int, default=40)
    parser.add_argument("--visual-attn-threshold", type=float, default=.35)
    parser.add_argument("--attn-implementation", choices=("auto", "sdpa", "eager"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)
    args.datasets = tuple(args.datasets.split(","))
    if set(args.datasets) - set(FIXED_DATASETS + ("MMSpec",)):
        parser.error("only the current seven fixed benchmarks plus MMSpec are allowed (no MME)")
    if args.sample_num < 1 or args.max_new_tokens < 1 or args.max_draft_tokens < 1:
        parser.error("sample and token limits must be positive")
    if not 0. <= args.visual_attn_threshold <= 1.:
        parser.error("visual-attn-threshold must be in [0, 1]")
    return args


def prepare_manifests(args):
    info = {}
    for dataset in args.datasets:
        if dataset == "MMSpec":
            path = args.data_folder / "mmspec.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 600:
                raise RuntimeError(f"expected current MMSpec 600-input split, found {len(rows)}")
            if args.smoke:
                counts = Counter()
                selected = []
                for row in rows:
                    if not counts[row["topic"]]:
                        selected.append(row)
                    counts[row["topic"]] += 1
                rows = selected
            expected = sum(len(row["turns"]) for row in rows)
        else:
            path = args.manifest_dir / f"{dataset_slug(dataset)}_seed42_n{args.sample_num}.jsonl"
            rows = read_jsonl(path)
            if len(rows) != args.sample_num:
                raise RuntimeError(f"incorrect manifest count in {path}")
            if args.smoke:
                rows = rows[:2]
            expected = len(rows)
        info[dataset] = dict(path=str(path.resolve()), sha256=sha256(path),
                             rows=rows, expected_turns=expected)
    return info


def main():
    args = parse_args()
    os.chdir(PROJECT)
    root = args.output_root.resolve()
    if (root / "config.json").exists():
        raise RuntimeError("refusing to overwrite an existing evaluation; use a new output root")
    manifests = prepare_manifests(args)
    root.mkdir(parents=True, exist_ok=True)
    qwen = is_qwen25_vl_model_path(str(args.model_path))
    attention = ("sdpa" if qwen else "eager") if args.attn_implementation == "auto" else args.attn_implementation
    paths = [Path(__file__), PROJECT / "MMSpec/method/sam_vskip/paper_model.py",
             PROJECT / "MMSpec/method/sam_vskip/spec_model.py",
             PROJECT / "MMSpec/method/sam/spec_model_sam.py",
             PROJECT / "MMSpec/method/sam/samd/sam_draft.py",
             PROJECT / "MMSpec/method/sam/samd/dyn_sam.py",
             PROJECT / "MMSpec/method/vispec/kv_cache.py",
             PROJECT / "MMSpec/method/vispec/modeling_qwen2_5_vl_kv.py",
             PROJECT / "MMSpec/method/vispec_vskip/spec_model.py",
             PROJECT / "MMSpec/method/llava_adapter.py",
             PROJECT / "method/fovea/evaluation/eval_llava_fixed_multi.py"]
    sources = {str(path.relative_to(PROJECT)): sha256(path) for path in paths}
    config = dict(protocol="sam_vskip_paper_paired_v1", policies=list(POLICIES),
                  model=str(args.model_path.resolve()), datasets=list(args.datasets),
                  temperature=0., batch_size=1, max_new_tokens=args.max_new_tokens,
                  max_draft_tokens=args.max_draft_tokens, total_token=64,
                  visual_attn_threshold=args.visual_attn_threshold,
                  visual_score="repository ViSkip: last decoder layer, head-mean attention mass over image tokens, query 0",
                  gating="existing SAM-vskip: set accepted draft prefix to zero on visual-score >= threshold",
                  attention_implementation=attention,
                  probe_attention="eager (existing SDPA fallback on Qwen when attention weights requested)",
                  timing="synchronized full generation including vision encoding, prefill, SAM and gate; excludes preprocessing and reusable KV workspace allocation",
                  repeat_guard=False, compact=True, decoded_output=False,
                  multi_turn_history="policy-specific committed token IDs with original chat-template delimiters; no decode/re-encode",
                  dtype="model config", seed=42, smoke=args.smoke,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), source_sha256=sources,
                  manifests={name: {k: v for k, v in item.items() if k != "rows"}
                             for name, item in manifests.items()},
                  expected_turns=sum(item["expected_turns"] for item in manifests.values()))
    write_json(root / "config.json", config)
    for name, item in manifests.items():
        write_json(root / "manifests" / f"{dataset_slug(name)}.json", item["rows"])
    print(json.dumps(dict(stage="loading_model", model=config["model"],
                          expected_turns=config["expected_turns"], attention=attention)), flush=True)
    torch.set_num_threads(1)
    model = PaperSAMViSkip.from_pretrained(
        base_model_path=str(args.model_path), spec_model_path=None, total_token=64,
        torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto",
        attn_implementation=attention).eval()
    model.processor = load_processor_for_model(str(args.model_path))
    config["actual_dtype"] = str(model.base_model.dtype)
    config["gpu_name"] = torch.cuda.get_device_name()
    write_json(root / "config.json", config)
    dataset_summaries = {}
    warmed_up = False
    global_position = 0

    def checkpoint(name, rows, paths):
        expected = manifests[name]["expected_turns"]
        values = {}
        for policy in POLICIES:
            values[policy] = summarize(rows[policy], paths[policy], expected)
            if name == "MMSpec":
                topics = sorted({row["topic"] for row in rows[policy]})
                values[policy]["by_topic"] = {
                    topic: summarize([row for row in rows[policy] if row["topic"] == topic],
                                     paths[policy], sum(len(sample["turns"]) for sample in manifests[name]["rows"]
                                                        if sample["topic"] == topic))
                    for topic in topics}
            write_json(paths[policy].parent / "summary.json", values[policy])
        dataset_summaries[name] = values
        complete = (len(dataset_summaries) == len(args.datasets)
                    and all(item[p]["complete"] for item in dataset_summaries.values() for p in POLICIES))
        write_json(root / "summary.json", dict(
            complete=complete, policies=list(POLICIES), expected_datasets=len(args.datasets),
            completed_datasets=sum(all(v[p]["complete"] for p in POLICIES) for v in dataset_summaries.values()),
            expected_turns=config["expected_turns"],
            completed_turns=sum(item["sam_vskip"]["num_turns"] for item in dataset_summaries.values()),
            sources_unchanged=all(sha256(PROJECT / name) == digest for name, digest in sources.items()),
            datasets=dataset_summaries, config_path=str(root / "config.json")))

    for dataset in args.datasets:
        item = manifests[dataset]
        selected = item["rows"]
        base = None if dataset == "MMSpec" else load_base_dataset(dataset)
        rows = {policy: [] for policy in POLICIES}
        paths = {policy: root / dataset_slug(dataset) / policy / "results.jsonl" for policy in POLICIES}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint(dataset, rows, paths)
        print(json.dumps(dict(stage="dataset_start", dataset=dataset, inputs=len(selected),
                              expected_turns=item["expected_turns"])), flush=True)
        try:
            for position, entry in enumerate(selected):
                if dataset == "MMSpec":
                    sample = entry
                    turns = len(sample["turns"])
                    sample_id = str(sample["id"])
                    first_inputs = build_mmspec_inputs(model, sample, 0, [], args.data_folder)
                else:
                    sample = base[int(entry["source_index"])]
                    turns = 1
                    sample_id = f"{dataset}:{entry['source_index']}"
                    first_inputs = build_inputs_for_record(model, dataset, sample)
                if not warmed_up:
                    print("stage=warmup policies=ar,sam_vskip", flush=True)
                    for _ in range(2):
                        for policy in POLICIES:
                            run_generation(model, first_inputs, policy, args, min(32, args.max_new_tokens))
                    warmed_up = True
                    print("stage=evaluation", flush=True)
                histories = {policy: [] for policy in POLICIES}
                for turn in range(turns):
                    offset = global_position % 2
                    order = POLICIES[offset:] + POLICIES[:offset]
                    generated = {}
                    for policy in order:
                        inputs = first_inputs if turn == 0 else build_mmspec_inputs(
                            model, sample, turn, histories[policy], args.data_folder)
                        generated[policy] = run_generation(model, inputs, policy, args)
                        histories[policy].append(generated[policy]["output_token_ids"])
                    for policy in POLICIES:
                        result = generated[policy]
                        result.update(dataset=dataset, sample_id=sample_id,
                                      sample_position=position, turn_index=turn, policy=policy,
                                      topic=sample.get("topic", dataset), order=list(order),
                                      speedup=result["tokens_per_second"] / generated["ar"]["tokens_per_second"],
                                      ar_new_tokens=generated["ar"]["new_tokens"],
                                      ar_wall_time=generated["ar"]["wall_time"],
                                      matches_ar=result["output_sha256"] == generated["ar"]["output_sha256"])
                        rows[policy].append(result)
                        with paths[policy].open("a") as handle:
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    global_position += 1
                if position == 0 or (position + 1) % args.log_every == 0:
                    checkpoint(dataset, rows, paths)
                    summary = dataset_summaries[dataset]["sam_vskip"]
                    print(json.dumps(dict(stage="progress", dataset=dataset,
                                          inputs_done=position + 1, total_inputs=len(selected),
                                          turns_done=summary["num_turns"], speedup=summary["avg_sample_speedup"],
                                          tau=summary["tau"], gate_hits=summary["visual_gate_hits"])), flush=True)
        finally:
            checkpoint(dataset, rows, paths)
        if sum(row["counters"]["visual_probes"] for row in rows["sam_vskip"]) == 0:
            raise RuntimeError(f"no valid visual-attention probes recorded on {dataset}")
        del base
    final = json.loads((root / "summary.json").read_text())
    if not final["complete"] or not final["sources_unchanged"]:
        raise RuntimeError("evaluation incomplete or source code changed while running")
    print(json.dumps(dict(stage="complete", summary=str(root / "summary.json"),
                          completed_turns=final["completed_turns"])), flush=True)


if __name__ == "__main__":
    main()
