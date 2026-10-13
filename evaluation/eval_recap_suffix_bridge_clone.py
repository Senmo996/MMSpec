"""Factorial RECAP suffix-bridge and SAM-clone validation on fixed manifests.

All suffix arms use the previous committed-state and single-pass fixes. Bridge
expansion only spends spare nodes after that tree is complete, never displacing
its candidates. Copy mode changes no SAM semantics. No occurrence selection,
length policy, fusion weight, auxiliary model or verification rule is changed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for directory in (SCRIPT_DIR, SCRIPT_DIR.parent, PROJECT_ROOT):
    sys.path.insert(0, str(directory))

import torch
from evaluation.eval_recap_paper_controls import (
    COMPARISONS, CONTROL_POLICIES, DATASETS, FUSION, control_kwargs, run_online,
)
from evaluation.eval_recap_suffix_fixes import REQUIRED_SUMMARY_FIELDS, load_samples
from evaluation.eval_sam_grounded_fixed_multi import _write_json, dataset_slug, load_processor_for_model
from method.sam.samd.dyn_sam import DynSAM
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model

EXPERIMENT = "suffix_bridge_clone"
BASE = dict(
    draft_policy=FUSION, recycling_suffix_length=4,
    recycling_suffix_implementation="fixed", recycling_suffix_bridge=False,
    recycling_sam_clone_mode="deepcopy", recycling_suffix_diagnostics=True,
)
CONTROL_POLICIES[EXPERIMENT] = {
    "target": BASE | dict(draft_policy="target", recycling_suffix_length=0),
    "no_suffix": BASE | dict(recycling_suffix_length=0),
    "fixed_suffix": BASE.copy(),
    "clone_only": BASE | dict(recycling_sam_clone_mode="flat"),
    "bridge_only": BASE | dict(recycling_suffix_bridge=True),
    "bridge_and_clone": BASE | dict(recycling_suffix_bridge=True, recycling_sam_clone_mode="flat"),
}
COMPARISONS[EXPERIMENT] = (
    ("fixed_suffix", "no_suffix"), ("clone_only", "fixed_suffix"),
    ("bridge_only", "fixed_suffix"), ("bridge_and_clone", "clone_only"),
    ("bridge_and_clone", "bridge_only"), ("bridge_and_clone", "fixed_suffix"),
    ("clone_only", "no_suffix"), ("bridge_only", "no_suffix"),
    ("bridge_and_clone", "no_suffix"),
)
COUNTER_KEYS = (
    "iterations", "total_candidate_nodes", "suffix_added_nodes",
    "suffix_bridge_opportunities", "suffix_bridge_active_iterations",
    "suffix_bridge_added_nodes", "suffix_bridge_accepted_tokens",
    "total_pre_bridge_candidate_nodes",
)


def semantic_state(sam):
    return {key: value for key, value in vars(sam).items() if key != "clone_mode"}


def cpu_checks():
    rng = random.Random(20260910)
    for _ in range(30):
        history = [rng.randrange(32) for _ in range(400)]
        a, b = DynSAM(device="cpu"), DynSAM(device="cpu", clone_mode="flat")
        for start in range(0, len(history), 13):
            a.add_tokens(history[start:start + 13])
            b.add_tokens(history[start:start + 13])
            assert semantic_state(a) == semantic_state(b)
    history = [rng.randrange(64) for _ in range(6000)]
    timings = {mode: [] for mode in ("deepcopy", "flat")}
    for repeat in range(8):
        order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
        for mode in order:
            started = time.perf_counter()
            sam = DynSAM(device="cpu", clone_mode=mode)
            sam.add_tokens(history)
            timings[mode].append(time.perf_counter() - started)
    args = dict(
        root_token=1, transitions=torch.empty(0, dtype=torch.long),
        transition_valid=torch.empty(0, dtype=torch.bool), width=2, depth=6,
        node_budget=8, blocked_token_id=-1, host_transitions={3: [4, 5], 4: [6]},
        host_transition_scores={3: [.8, .2], 4: [1.]},
        score_priority_layout=True, reserved_path_tokens=[2, 3],
    )
    before = Model._build_tree_budgeted_single_pass(**args)
    metadata = {"_suffix_bridge_diagnostics": True}
    after = Model._build_tree_budgeted_single_pass(**args, suffix_bridge=True, metadata_out=metadata)
    size = len(before[0])
    assert after[0][:size] == before[0] and after[3][:size - 1] == before[3]
    assert torch.equal(after[1][..., :size, :size], before[1])
    assert torch.equal(after[2][:size], before[2])
    assert metadata["suffix_bridge_added_nodes"] == 3 and len(after[0]) - 1 <= 8
    return dict(
        complete=True, clone_equivalent_histories=30,
        fixture="CPU-only synthetic SAM indexing and sparse suffix bridge",
        timing_scope="SAM construction only; not an end-to-end speedup",
        history_tokens=6000, clone_seconds_by_repeat=timings,
        clone_median_seconds={mode: statistics.median(values) for mode, values in timings.items()},
        baseline_paths=Model._tree_path_token_ids(before[0], before[3]),
        bridge_paths=Model._tree_path_token_ids(after[0], after[3]),
        baseline_prefix_unchanged=True, bridge_diagnostics=metadata,
    )


def enrich_summaries(root, result, expected_count):
    """Verify compact outputs and save bridge contribution beside each JSONL."""
    clone_checks = {}
    for dataset in DATASETS:
        indexed = {}
        for name in CONTROL_POLICIES[EXPERIMENT]:
            directory = root / dataset_slug(dataset) / name
            rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
            indexed[name] = {row["question_id"]: row["choices"][0] for row in rows}
            if len(rows) != expected_count or len(indexed[name]) != expected_count:
                raise RuntimeError(f"incomplete or duplicate records: {dataset}/{name}")
            summary = result["datasets"][dataset][name]
            if REQUIRED_SUMMARY_FIELDS - summary.keys() or summary["num_records"] != expected_count:
                raise RuntimeError(f"incomplete summary: {dataset}/{name}")
            counters = [row["choices"][0]["recap_counters"] for row in rows]
            if any(c.get("max_candidate_nodes", 0) > 63 for c in counters):
                raise RuntimeError("candidate cap exceeded")
            if name != "target":
                expected_mode = CONTROL_POLICIES[EXPERIMENT][name]["recycling_sam_clone_mode"]
                for counter in counters:
                    if counter.get("sam_clone_mode") != expected_mode:
                        raise RuntimeError("SAM copy mode was not reset for a policy")
                    if counter["total_pre_bridge_candidate_nodes"] + counter["suffix_bridge_added_nodes"] != counter["total_candidate_nodes"]:
                        raise RuntimeError("bridge node accounting mismatch")
            totals = {key: sum(c.get(key, 0) for c in counters) for key in COUNTER_KEYS}
            if not CONTROL_POLICIES[EXPERIMENT][name]["recycling_suffix_bridge"] and totals["suffix_bridge_added_nodes"]:
                raise RuntimeError("bridge nodes appeared in a non-bridge arm")
            denom = max(totals["iterations"], 1)
            totals.update(
                bridge_opportunity_ratio=totals["suffix_bridge_opportunities"] / denom,
                bridge_active_iteration_ratio=totals["suffix_bridge_active_iterations"] / denom,
                avg_bridge_added_nodes=totals["suffix_bridge_added_nodes"] / denom,
                avg_pre_bridge_candidate_nodes=totals["total_pre_bridge_candidate_nodes"] / denom,
                avg_bridge_accepted_tokens=totals["suffix_bridge_accepted_tokens"] / denom,
            )
            summary["suffix_component_counters"] = totals
            _write_json(directory / "summary.json", summary)
        reference_keys = set(indexed["target"])
        if any(set(values) != reference_keys for values in indexed.values()):
            raise RuntimeError("policies do not use identical samples")
        clone_checks[dataset] = {}
        for left, right in (("clone_only", "fixed_suffix"), ("bridge_and_clone", "bridge_only")):
            fields = ("output_token_ids", "acceptance_length", "new_tokens", "idxs")
            clone_checks[dataset][f"{left}_vs_{right}"] = {
                "num_samples": expected_count,
                "identical_output_and_acceptance_samples": sum(
                    all(indexed[left][key][field] == indexed[right][key][field] for field in fields)
                    for key in reference_keys
                ),
                "identical_tree_counter_samples": sum(
                    all(indexed[left][key]["recap_counters"].get(field) == indexed[right][key]["recap_counters"].get(field) for field in COUNTER_KEYS)
                    for key in reference_keys
                ),
            }
    result["clone_semantic_checks"] = clone_checks
    _write_json(root / "clone_semantic_checks.json", clone_checks)
    _write_json(root / "summary.json", result)
    return clone_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "outputs/experiments/recap/20260907-171145-recap-online-confirm4x100-gpu2/manifests")
    parser.add_argument("--base-model-path", default=str(PROJECT_ROOT / "models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--before-source-root", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples_per_dataset <= 100:
        parser.error("samples-per-dataset must lie in [1, 100]")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__), SCRIPT_DIR / "eval_recap_suffix_fixes.py",
        SCRIPT_DIR / "eval_recap_paper_controls.py", SCRIPT_DIR / "eval_sam_grounded_fixed_multi.py",
        SCRIPT_DIR / "summarize_training_free.py", SCRIPT_DIR / "analyze_policy_pairs.py",
        SCRIPT_DIR.parent / "method/sam_grounded/tree_recycling_model.py",
        SCRIPT_DIR.parent / "method/sam_grounded/controller.py",
        SCRIPT_DIR.parent / "method/sam/samd/dyn_sam.py",
        SCRIPT_DIR.parent / "method/sam/samd/sam_draft.py",
        SCRIPT_DIR.parent / "tests/test_recap_suffix_bridge_clone.py",
        PROJECT_ROOT / "method/fovea/evaluation/eval_llava_fixed_multi.py",
        PROJECT_ROOT / "scripts/launch_recap_suffix_bridge_clone.sh",
    ]
    if not args.cpu_only:
        sources += [args.manifest_dir / f"{dataset_slug(d)}_seed42_n100.jsonl" for d in DATASETS]
    hashes = {}
    for source in sources:
        relative = source.resolve().relative_to(PROJECT_ROOT)
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    before_hashes = {}
    if args.before_source_root:
        before_hashes = {str(p.relative_to(args.before_source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(args.before_source_root.rglob("*.py"))}
    protocol = dict(
        protocol="recap_suffix_bridge_clone_factorial_v1",
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        controls={name: control_kwargs(EXPERIMENT, name) for name in CONTROL_POLICIES[EXPERIMENT]},
        source_sha256=hashes, before_source_sha256=before_hashes,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_role="previously evaluated inputs; mechanism development controls",
        bridge_rule="extend only a newly inserted suffix endpoint, using spare budget; preserve original packed nodes",
        bridge_scoring="existing context candidate distributions and depth hit masses; common endpoint mass cancels in local ranking",
        total_candidate_cap=63, max_suffix_tokens=4, max_tree_depth=10,
        original_candidate_eviction=False, suffix_occurrence_policy_changed=False,
        suffix_length_policy_changed=False, fusion_weights_changed=False,
        cpp_builder_enabled=False, visual_feature_reuse_enabled=False,
        clone_default_unchanged=True, per_forward_timing_hooks=False,
        cpu_component_counters_timed=True, decoded_output_saved=False,
        timing_scope="synchronized end-to-end specgenerate, including suffix indexing and bridge work",
        order="six policies rotated within every request; shared manifest, input processing and warmup",
    )
    _write_json(root / "protocol.json", protocol)
    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')} output_root={root}", flush=True)
    torch.set_num_threads(4)
    checks = cpu_checks()
    _write_json(root / "cpu_component_checks.json", checks)
    print("stage=cpu_checks_complete clone_states_equal=true original_tree_preserved=true", flush=True)
    if args.cpu_only:
        _write_json(root / "summary.json", dict(complete=True, cpu_only=True, checks=checks))
        return
    print("stage=load_manifest_inputs", flush=True)
    samples, manifests = load_samples(args.manifest_dir, args.samples_per_dataset)
    print(f"stage=load_model requests={len(samples)} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    model = Model.from_pretrained(
        base_model_path=args.base_model_path, spec_model_path="", total_token=112,
        torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa",
    )
    model.processor = load_processor_for_model(args.base_model_path)
    model.eval()
    online_root = root / "online_suffix_bridge_clone"
    result = run_online(model, samples, online_root, EXPERIMENT, manifests, seed=args.order_seed)
    clone_checks = enrich_summaries(online_root, result, args.samples_per_dataset)
    _write_json(root / "summary.json", dict(
        complete=True, cpu_only=False, protocol=protocol["protocol"],
        num_inputs=len(samples), samples_per_dataset=args.samples_per_dataset,
        policies=list(CONTROL_POLICIES[EXPERIMENT]), online_summary=str(online_root / "summary.json"),
        datasets=result["datasets"], paired=result["paired"], clone_semantic_checks=clone_checks,
    ))
    print(f"stage=finished summary={root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
