"""Wrong-image positive control for multimodal selective-reuse diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MMSPEC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = MMSPEC_ROOT.parent
for path in (str(MMSPEC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.eval_sam_grounded_mmspec import (  # noqa: E402
    _sample_order_indices,
    _token_hash,
)
from evaluation.utils import build_prompt, load_mmspec_data  # noqa: E402
from method.sam_grounded.tree_recycling_model import (  # noqa: E402
    TreeRecyclingSpecModel,
)
from new_dream.evaluation.eval_llava_fixed_multi import (  # noqa: E402
    build_inputs_for_record,
    dataset_slug,
    load_base_dataset,
    load_image_for_record,
    load_or_create_manifest,
    load_processor_for_model,
    parse_datasets,
)


DEFAULT_DATASETS = (
    "MMT-Bench,SEEDBench,ScienceQA,OCRBench,ChartQA,MathVista,"
    "TextVQA,MME_Benchmark"
)


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def _category(row, manifest_row=None):
    manifest_row = manifest_row or {}
    for key in ("category", "type", "topic", "dataset_name"):
        value = manifest_row.get(key, row.get(key))
        if value not in (None, "", "default", "unknown"):
            return str(value)
    return "default"


def _image_identity(row, manifest_row=None):
    manifest_row = manifest_row or {}
    for key in ("image_id", "image_path", "imgname"):
        value = manifest_row.get(key, row.get(key))
        if value not in (None, ""):
            return f"{key}:{value}"
    value = row.get("image")
    if isinstance(value, str):
        return "image:" + hashlib.sha1(value.encode("utf-8")).hexdigest()
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return "bytes:" + hashlib.sha1(value["bytes"]).hexdigest()
        if value.get("path"):
            return f"path:{value['path']}"
    return f"row:{row.get('_fixed_index', id(row))}"


def build_pairing(rows, manifest, count: int, seed: int, name: str):
    """Choose deterministic targets and distinct, same-category source images."""

    if len(rows) < 2:
        raise ValueError(f"{name} needs at least two rows for wrong-image pairing")
    rng = random.Random(_stable_seed(seed, name))
    # The source pool is already a seed-42 permutation.  Taking its prefix
    # keeps the control deterministic and makes partial smoke runs joinable.
    targets = list(range(min(count, len(rows))))
    categories = [
        _category(row, manifest[index] if manifest else None)
        for index, row in enumerate(rows)
    ]
    identities = [
        _image_identity(row, manifest[index] if manifest else None)
        for index, row in enumerate(rows)
    ]
    pairs = []
    for target in targets:
        candidates = [
            index
            for index in range(len(rows))
            if index != target
            and identities[index] != identities[target]
            and categories[index] == categories[target]
        ]
        used_category_fallback = False
        if not candidates:
            used_category_fallback = True
            candidates = [
                index
                for index in range(len(rows))
                if index != target and identities[index] != identities[target]
            ]
        if not candidates:
            raise ValueError(f"{name}:{target} has no distinct source image")
        source = candidates[rng.randrange(len(candidates))]
        pairs.append(
            {
                "target_position": int(target),
                "source_position": int(source),
                "category": categories[target],
                "target_image_identity": identities[target],
                "source_image_identity": identities[source],
                "used_category_fallback": used_category_fallback,
            }
        )
    return pairs


def _resize_wrong_image(wrong_image, target_image):
    if not isinstance(wrong_image, Image.Image):
        return wrong_image
    wrong_image = wrong_image.convert("RGB")
    if isinstance(target_image, Image.Image) and wrong_image.size != target_image.size:
        wrong_image = wrong_image.resize(
            target_image.size, resample=Image.Resampling.BICUBIC
        )
    return wrong_image


def _generation_kwargs(args):
    """Use target-only decoding so image replacement is the sole treatment."""

    return {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_token,
        "log": True,
        "return_acceptance_len": True,
        "return_policy_trace": False,
        "draft_policy": "target",
    }


def _append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _existing_ids(path: Path):
    if not path.exists():
        return set()
    return {
        str(row["question_id"])
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _save_result(
    path,
    *,
    question_id,
    benchmark,
    category,
    pair,
    input_len,
    result,
    elapsed,
    policy,
):
    output_ids, n_new, idx, accepted = result
    choice = {
        "index": 0,
        "output_hashes": [_token_hash(output_ids, input_len)],
        "idxs": [int(idx)],
        "new_tokens": [int(n_new)],
        "wall_time": [float(elapsed)],
        "acceptance_length": [accepted],
        "output_token_ids": [
            output_ids[0, input_len:].detach().cpu().tolist()
        ],
    }
    _append_jsonl(
        path,
        {
            "question_id": str(question_id),
            "topic": benchmark,
            "category": category,
            "model_id": "selective-reuse-wrong-image-target-only",
            "wrong_image_control": True,
            "wrong_image_pair": pair,
            "choices": [choice],
            "tstamp": time.time(),
        },
    )


def _summarize_result_file(path: Path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accept = []
    for row in rows:
        for choice in row.get("choices", []):
            for turn in choice.get("acceptance_length", []):
                accept.extend(float(value) for value in turn)
    payload = {
        "num_records": len(rows),
        "avg_accept_length": float(np.mean(accept)) if accept else 0.0,
        "jsonl_path": str(path.resolve()),
    }
    (path.parent / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


@torch.inference_mode()
def evaluate(args):
    model = TreeRecyclingSpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path="",
        total_token=112,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.processor = load_processor_for_model(args.base_model_path)
    model.eval()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_manifest_path = output_root / "pairing_manifest.jsonl"
    summaries = {}

    fixed_manifest_args = SimpleNamespace(
        manifest_dir=args.manifest_dir,
        seed=args.seed,
        sample_num=args.source_pool_size,
        regenerate_manifest=False,
    )
    dataset_payloads = []
    for dataset in args.datasets:
        base = load_base_dataset(dataset)
        selected, manifest, _ = load_or_create_manifest(
            dataset, base, fixed_manifest_args
        )
        rows = [selected[index] for index in range(len(selected))]
        dataset_payloads.append((dataset, rows, manifest))

    mmspec = load_mmspec_data(args.data_folder)
    mmspec = mmspec.select(
        _sample_order_indices(len(mmspec), args.seed)
    )
    mmspec = mmspec.select(
        range(min(args.source_pool_size, len(mmspec)))
    )
    dataset_payloads.insert(
        0,
        (
            "MMSpec",
            [mmspec[index] for index in range(len(mmspec))],
            None,
        ),
    )

    # Warm target-only decoding once. No recycling cache participates in this
    # positive control.
    warm_dataset, warm_rows, _ = dataset_payloads[1]
    warm_inputs = build_inputs_for_record(model, warm_dataset, warm_rows[0])
    model.specgenerate(
        **warm_inputs,
        temperature=0.0,
        max_new_tokens=min(args.max_new_token, 16),
        log=True,
        draft_policy="target",
    )

    for dataset, rows, manifest in dataset_payloads:
        pairs = build_pairing(
            rows,
            manifest,
            args.samples_per_benchmark,
            args.seed,
            dataset,
        )
        slug = dataset_slug(dataset)
        result_path = output_root / slug / "target" / "results.jsonl"
        existing = _existing_ids(result_path)
        for pair in tqdm(pairs, desc=f"Wrong-image {dataset}"):
            target_position = pair["target_position"]
            source_position = pair["source_position"]
            target_row = rows[target_position]
            source_row = rows[source_position]
            question_id = (
                str(target_row["id"])
                if dataset == "MMSpec"
                else f"{slug}:{target_position:04d}"
            )
            pair_payload = {"benchmark": dataset, **pair}
            if question_id in existing:
                continue

            if dataset == "MMSpec":
                target_image = target_row["image"].convert("RGB")
                wrong_image = _resize_wrong_image(
                    source_row["image"], target_image
                )
                prompt_row = dict(target_row)
                prompt_row["image"] = wrong_image
                prompt_args = SimpleNamespace(model=args.base_model_path)
                inputs = build_prompt(prompt_row, prompt_args, turn_idx=0)
                benchmark = "MMSpec"
                category = _category(target_row)
            else:
                target_image = load_image_for_record(dataset, target_row)
                wrong_image = _resize_wrong_image(
                    load_image_for_record(dataset, source_row), target_image
                )
                inputs = build_inputs_for_record(
                    model,
                    dataset,
                    target_row,
                    image_override=wrong_image,
                )
                benchmark = dataset
                category = _category(
                    target_row,
                    manifest[target_position] if manifest else None,
                )

            input_len = int(inputs["input_ids"].shape[1])
            torch.manual_seed(0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = model.specgenerate(
                **inputs,
                **_generation_kwargs(args),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            _save_result(
                result_path,
                question_id=question_id,
                benchmark=benchmark,
                category=category,
                pair=pair_payload,
                input_len=input_len,
                result=result,
                elapsed=elapsed,
                policy="target",
            )
            _append_jsonl(
                pair_manifest_path,
                {"question_id": question_id, **pair_payload},
            )
            existing.add(question_id)

        summaries[dataset] = _summarize_result_file(result_path)
        torch.cuda.empty_cache()

    summary = {
        "control": "distinct wrong image, same benchmark and same category when available",
        "samples_per_benchmark": args.samples_per_benchmark,
        "source_pool_size": args.source_pool_size,
        "seed": args.seed,
        "generation_policy": "target-only",
        "comparison_method_policy": args.policy,
        "datasets": summaries,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--source-pool-size", type=int, default=100)
    parser.add_argument("--samples-per-benchmark", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-token", type=int, default=200)
    parser.add_argument("--policy", required=True)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    args = parser.parse_args()
    args.datasets = parse_datasets(args.datasets)
    if args.source_pool_size < 2:
        parser.error("--source-pool-size must be at least two")
    if args.samples_per_benchmark <= 0:
        parser.error("--samples-per-benchmark must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
