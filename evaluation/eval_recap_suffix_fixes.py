"""Paired, fixed-budget validation of the two RECAP suffix implementation fixes.

Compare no suffix, the legacy suffix, and the corrected suffix; use the same
target-only reference, manifests, fusion weights and Python tree backend.
These inputs were evaluated previously and are mechanism follow-up controls.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for directory in (SCRIPT_DIR, SCRIPT_DIR.parent, PROJECT_ROOT):
    sys.path.insert(0, str(directory))

import torch
from evaluation.eval_recap_paper_controls import (
    COMPARISONS, CONTROL_POLICIES, DATASETS, FUSION, control_kwargs, run_online,
)
from evaluation.eval_sam_grounded_fixed_multi import (
    _image_cluster_identity, _write_json, dataset_slug, load_base_dataset,
    load_or_create_manifest, load_processor_for_model,
)
from method.sam.samd.sam_draft import DraftModel
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel

EXPERIMENT = "suffix_fixes"
CONTROL_POLICIES[EXPERIMENT] = {
    "target": {"draft_policy": "target"},
    "no_suffix": {"draft_policy": FUSION, "recycling_suffix_length": 0},
    "legacy_suffix": {
        "draft_policy": FUSION, "recycling_suffix_length": 4,
        "recycling_suffix_implementation": "legacy",
    },
    "fixed_suffix": {
        "draft_policy": FUSION, "recycling_suffix_length": 4,
        "recycling_suffix_implementation": "fixed",
    },
}
COMPARISONS[EXPERIMENT] = (
    ("legacy_suffix", "no_suffix"), ("fixed_suffix", "no_suffix"),
    ("fixed_suffix", "legacy_suffix"),
)
REQUIRED_SUMMARY_FIELDS = {
    "num_records", "avg_accept_length", "median_accept_length",
    "max_accept_length", "p90_accept_length", "avg_tokens_per_iteration",
    "avg_sample_speedup", "median_sample_speedup", "max_sample_speedup",
    "sample_speedup_gt_1_ratio", "accept_le_0_5_ratio", "jsonl_path",
}


def cpu_component_checks():
    """Small synthetic checks; these timings are not end-to-end GPU results."""
    draft = DraftModel(device="cpu")
    draft.update_tokens([11, 22, 33, 44, 55, 22, 66, 44, 55, 22])
    old_tokens = draft.lookup_tokens(22)[0][:4]
    new_tokens = draft.lookup_continuation_tokens(4)
    assert old_tokens == [33, 44, 55, 22] and new_tokens == [66, 44, 55, 22]
    args = dict(
        root_token=1, transitions=torch.empty(0, dtype=torch.long),
        transition_valid=torch.empty(0, dtype=torch.bool), width=8, depth=10,
        node_budget=63, blocked_token_id=-1,
        host_transitions={i: list(range(i + 1, i + 9)) for i in range(1, 100)},
        host_transition_scores={i: [.65, .15, .08, .04, .03, .02, .02, .01] for i in range(1, 100)},
        score_priority_layout=True, reserved_path_tokens=[2, 3, 4, 5],
    )
    model = TreeRecyclingSpecModel
    methods = {"legacy": model._build_tree_budgeted, "fixed": model._build_tree_budgeted_single_pass}
    results, trees = {}, {}
    for name, method in methods.items():
        with patch.object(model, "_build_tree", wraps=model._build_tree) as counted:
            trees[name] = method(**args)
            results[name] = {"tree_build_calls": counted.call_count}
    left, right = trees["legacy"], trees["fixed"]
    assert left[0] == right[0] and left[3] == right[3]
    assert torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
    assert len(right[0]) - 1 == 63 and results["fixed"]["tree_build_calls"] == 1
    caches = {name: {} for name in methods}
    measurements = {name: [] for name in methods}
    for name, method in methods.items():
        for _ in range(10):
            method(**args, topology_cache=caches[name])
    for repeat in range(8):
        names = list(methods) if repeat % 2 == 0 else list(reversed(methods))
        for name in names:
            started = time.perf_counter()
            for _ in range(100):
                methods[name](**args, topology_cache=caches[name])
            measurements[name].append((time.perf_counter() - started) / 100)
    for name in methods:
        results[name]["cpu_seconds_per_call_by_repeat"] = measurements[name]
        results[name]["median_cpu_seconds_per_call"] = statistics.median(measurements[name])
    return {
        "complete": True, "fixture": "synthetic CPU-only, warm topology cache",
        "timing_scope": "tree allocation only; not a GPU speedup estimate",
        "committed_match_length": draft.sam_dyn.cur_length,
        "legacy_requeried_match_length": draft.sam_dyn.lookup(22)[1],
        "legacy_continuation": old_tokens, "fixed_continuation": new_tokens,
        "candidate_nodes": 63, "identical_tree_given_same_suffix": True,
        "tree_builders": results,
    }


def load_samples(manifest_dir, count):
    samples, manifests = [], {}
    manifest_args = SimpleNamespace(
        manifest_dir=str(manifest_dir), seed=42, sample_num=100,
        regenerate_manifest=False,
    )
    for dataset in DATASETS:
        selected, manifest, path = load_or_create_manifest(
            dataset, load_base_dataset(dataset), manifest_args,
        )
        if len(selected) != 100 or len(manifest) != 100:
            raise RuntimeError("require the existing 100-input manifest for every dataset")
        manifests[dataset] = path
        for index, meta in enumerate(manifest[:count]):
            row = selected[index]
            samples.append({
                "id": f"{dataset_slug(dataset)}:{index:04d}", "dataset": dataset,
                "row": row, "source_index": int(meta["source_index"]),
                "sample_position": index,
                "image_cluster_id": _image_cluster_identity(dataset, row, meta),
            })
    return samples, manifests


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
        Path(__file__), SCRIPT_DIR / "eval_recap_paper_controls.py",
        SCRIPT_DIR / "eval_sam_grounded_fixed_multi.py",
        SCRIPT_DIR / "summarize_training_free.py", SCRIPT_DIR / "analyze_policy_pairs.py",
        SCRIPT_DIR.parent / "method/sam_grounded/tree_recycling_model.py",
        SCRIPT_DIR.parent / "method/sam_grounded/controller.py",
        SCRIPT_DIR.parent / "method/sam/samd/sam_draft.py",
        SCRIPT_DIR.parent / "method/sam/samd/dyn_sam.py",
        SCRIPT_DIR.parent / "tests/test_recap_suffix_fixes.py",
        PROJECT_ROOT / "method/fovea/evaluation/eval_llava_fixed_multi.py",
        PROJECT_ROOT / "scripts/launch_recap_suffix_fixes.sh",
    ]
    if not args.cpu_only:
        sources += [args.manifest_dir / f"{dataset_slug(d)}_seed42_n100.jsonl" for d in DATASETS]
    hashes = {}
    for source in sources:
        source = source.resolve()
        relative = source.relative_to(PROJECT_ROOT)
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    before_hashes = {}
    if args.before_source_root:
        for source in sorted(args.before_source_root.rglob("*.py")):
            before_hashes[str(source.relative_to(args.before_source_root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = {
        "protocol": "recap_suffix_two_fixes_v1",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "controls": {name: control_kwargs(EXPERIMENT, name) for name in CONTROL_POLICIES[EXPERIMENT]},
        "source_sha256": hashes, "before_source_sha256": before_hashes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_role": "previously evaluated inputs; implementation follow-up",
        "scope": ["committed-state suffix lookup", "single-pass shared-budget allocation", "remove unused suffix position tensor"],
        "suffix_gating": False, "fusion_weights_changed": False,
        "cpp_builder_enabled": False, "visual_feature_reuse_enabled": False,
        "candidate_cap_includes_suffix": 63, "max_suffix_tokens": 4,
        "per_forward_timing_hooks": False, "decoded_output_saved": False,
        "timing_scope": "synchronized end-to-end specgenerate, including suffix work",
        "order": "policies rotated within each request; shared inputs and warmup",
    }
    _write_json(root / "protocol.json", protocol)
    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')} output_root={root}", flush=True)
    torch.set_num_threads(4)
    checks = cpu_component_checks()
    _write_json(root / "cpu_component_checks.json", checks)
    print("stage=cpu_checks_complete same_tree=true single_pass=true", flush=True)
    if args.cpu_only:
        _write_json(root / "summary.json", {"complete": True, "cpu_only": True, "checks": checks})
        return
    print("stage=load_manifest_inputs", flush=True)
    samples, manifests = load_samples(args.manifest_dir, args.samples_per_dataset)
    print(f"stage=load_model requests={len(samples)} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    model = TreeRecyclingSpecModel.from_pretrained(
        base_model_path=args.base_model_path, spec_model_path="", total_token=112,
        torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto",
        attn_implementation="sdpa",
    )
    model.processor = load_processor_for_model(args.base_model_path)
    model.eval()
    result = run_online(model, samples, root / "online_suffix_fixes", EXPERIMENT,
                        manifests, seed=args.order_seed)
    for dataset in DATASETS:
        for name, summary in result["datasets"][dataset].items():
            missing = REQUIRED_SUMMARY_FIELDS - summary.keys()
            if missing or summary["num_records"] != args.samples_per_dataset:
                raise RuntimeError(f"invalid completed summary: {dataset}/{name}, missing={missing}")
    _write_json(root / "summary.json", {
        "complete": True, "cpu_only": False, "protocol": protocol["protocol"],
        "num_inputs": len(samples), "samples_per_dataset": args.samples_per_dataset,
        "policies": list(CONTROL_POLICIES[EXPERIMENT]),
        "cpu_checks": str(root / "cpu_component_checks.json"),
        "online_summary": str(root / "online_suffix_fixes/summary.json"),
        "datasets": result["datasets"], "paired": result["paired"],
    })
    print(f"stage=finished summary={root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
