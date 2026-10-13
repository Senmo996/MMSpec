"""Build fixed-benchmark manifests disjoint from all declared prior sets."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import sys
from pathlib import Path


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MMSPEC_ROOT.parent
for path in (str(MMSPEC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.eval_sam_grounded_fixed_multi import _image_cluster_identity
from method.fovea.evaluation.eval_llava_fixed_multi import (
    dataset_slug,
    load_base_dataset,
    load_image_for_record,
    manifest_record,
    parse_datasets,
)


DEFAULT_DATASETS = (
    "MMT-Bench,SEEDBench,ScienceQA,OCRBench,ChartQA,MathVista,TextVQA"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pixel_identity(image) -> str:
    image = image.convert("RGB")
    digest = hashlib.sha256(f"{image.width}x{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def build_one(dataset: str, args) -> dict:
    slug = dataset_slug(dataset)
    reference_path = (
        args.source_manifest_dir
        / f"{slug}_seed{args.seed}_n{args.reference_count}.jsonl"
    )
    pool_path = (
        args.source_manifest_dir
        / f"{slug}_seed{args.seed}_n{args.pool_count}.jsonl"
    )
    reference = read_jsonl(reference_path)
    base_dataset = load_base_dataset(dataset)
    if getattr(args, "full_pool", False):
        shuffled = base_dataset.shuffle(seed=args.seed)
        pool = [manifest_record(dataset, index, shuffled[index]) for index in range(len(shuffled))]
    else:
        pool = read_jsonl(pool_path)

    check_pixels = getattr(args, "check_pixels", False)

    @lru_cache(maxsize=None)
    def pixel_identity(source_index):
        return _pixel_identity(load_image_for_record(dataset, base_dataset[source_index]))

    def identity(manifest_row: dict) -> str:
        source_index = int(manifest_row["source_index"])
        return _image_cluster_identity(
            dataset, base_dataset[source_index], manifest_row
        )

    reference_source_indices = {
        int(row["source_index"]) for row in reference
    }
    reference_clusters = {identity(row) for row in reference}
    additional_exclusion_paths = []
    additional_exclusions = []
    for directory in args.exclude_manifest_dir:
        matches = sorted(
            directory.glob(f"{slug}_seed{args.seed}_n*.jsonl")
        )
        if not matches:
            raise ValueError(
                f"no exclusion manifest for {dataset} under {directory}"
            )
        for path in matches:
            additional_exclusion_paths.append(path.resolve())
            additional_exclusions.extend(read_jsonl(path))
    additional_source_indices = {
        int(row["source_index"]) for row in additional_exclusions
    }
    additional_clusters = {identity(row) for row in additional_exclusions}
    excluded_source_indices = reference_source_indices | additional_source_indices
    excluded_clusters = reference_clusters | additional_clusters
    excluded_pixels = {pixel_identity(index) for index in excluded_source_indices} if check_pixels else set()
    eligible = []
    eligible_clusters = set()
    eligible_pixels = set()
    rejected_source_overlap = 0
    rejected_cluster_overlap = 0
    rejected_duplicate_cluster = 0
    rejected_pixel_overlap = 0
    rejected_duplicate_pixel = 0
    for row in pool:
        source_index = int(row["source_index"])
        if source_index in excluded_source_indices:
            rejected_source_overlap += 1
            continue
        cluster = identity(row)
        if cluster in excluded_clusters:
            rejected_cluster_overlap += 1
            continue
        if cluster in eligible_clusters:
            rejected_duplicate_cluster += 1
            continue
        if check_pixels:
            pixel = pixel_identity(source_index)
            if pixel in excluded_pixels:
                rejected_pixel_overlap += 1
                continue
            if pixel in eligible_pixels:
                rejected_duplicate_pixel += 1
                continue
            eligible_pixels.add(pixel)
        eligible.append((row, cluster))
        eligible_clusters.add(cluster)

    start = int(args.selection_offset)
    stop = start + int(args.sample_num)
    chosen_pairs = eligible[start:stop]
    if len(chosen_pairs) != args.sample_num:
        raise ValueError(
            f"{dataset} has only {len(eligible)} eligible rows; "
            f"cannot select [{start}:{stop}]"
        )
    chosen = []
    for sample_position, (row, _cluster) in enumerate(chosen_pairs):
        copied = dict(row)
        copied["sample_position"] = sample_position
        copied["disjoint_reference_count"] = int(args.reference_count)
        copied["disjoint_selection_offset"] = start
        if check_pixels:
            copied["rgb_pixel_sha256"] = pixel_identity(int(row["source_index"]))
        chosen.append(copied)

    output_path = (
        args.output_dir
        / f"{slug}_seed{args.seed}_n{args.sample_num}.jsonl"
    )
    write_jsonl(output_path, chosen)
    chosen_source_indices = {int(row["source_index"]) for row in chosen}
    chosen_clusters = {cluster for _row, cluster in chosen_pairs}
    chosen_pixels = {row["rgb_pixel_sha256"] for row in chosen} if check_pixels else set()
    if check_pixels and (chosen_pixels & excluded_pixels or len(chosen_pixels) != len(chosen)):
        raise ValueError("pixel-level disjointness or uniqueness check failed")
    return {
        "dataset": dataset,
        "slug": slug,
        "reference_manifest": str(reference_path.resolve()),
        "candidate_pool_manifest": None if getattr(args, "full_pool", False) else str(pool_path.resolve()),
        "candidate_pool_scope": "full dataset, shuffled with declared seed" if getattr(args, "full_pool", False) else "existing manifest",
        "additional_exclusion_manifests": [
            str(path) for path in additional_exclusion_paths
        ],
        "output_manifest": str(output_path.resolve()),
        "reference_count": len(reference),
        "pool_count": len(pool),
        "eligible_unique_clusters": len(eligible),
        "selection_offset": start,
        "selected_count": len(chosen),
        "selected_unique_clusters": len(chosen_clusters),
        "source_index_overlap_with_reference": len(
            chosen_source_indices & reference_source_indices
        ),
        "image_cluster_overlap_with_reference": len(
            chosen_clusters & reference_clusters
        ),
        "source_index_overlap_with_additional_exclusions": len(
            chosen_source_indices & additional_source_indices
        ),
        "image_cluster_overlap_with_additional_exclusions": len(
            chosen_clusters & additional_clusters
        ),
        "source_index_overlap_with_all_exclusions": len(
            chosen_source_indices & excluded_source_indices
        ),
        "image_cluster_overlap_with_all_exclusions": len(
            chosen_clusters & excluded_clusters
        ),
        "rejected_source_overlap": rejected_source_overlap,
        "rejected_cluster_overlap_after_source_filter": rejected_cluster_overlap,
        "rejected_duplicate_cluster": rejected_duplicate_cluster,
        "pixel_check_enabled": check_pixels,
        "pixel_overlap_with_all_exclusions": len(chosen_pixels & excluded_pixels) if check_pixels else None,
        "selected_unique_pixel_hashes": len(chosen_pixels) if check_pixels else None,
        "rejected_pixel_overlap": rejected_pixel_overlap,
        "rejected_duplicate_pixel": rejected_duplicate_pixel,
        "output_manifest_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "result/manifests/dream_baseline_seed42",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=parse_datasets, default=parse_datasets(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-count", type=int, default=100)
    parser.add_argument("--pool-count", type=int, default=1000)
    parser.add_argument("--full-pool", action="store_true", help="Select from the full loaded dataset instead of a pool manifest.")
    parser.add_argument("--check-pixels", action="store_true", help="Additionally exclude RGB pixel duplicates, including metadata aliases.")
    parser.add_argument(
        "--exclude-manifest-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional directory containing matching per-dataset JSONL "
            "manifests to exclude. May be supplied more than once."
        ),
    )
    parser.add_argument("--sample-num", type=int, required=True)
    parser.add_argument("--selection-offset", type=int, default=0)
    args = parser.parse_args()
    if args.sample_num <= 0:
        parser.error("--sample-num must be positive")
    if args.selection_offset < 0:
        parser.error("--selection-offset must be non-negative")
    if any(dataset == "MME_Benchmark" for dataset in args.datasets):
        parser.error("MME is excluded from this diagnostic")

    reports = []
    for dataset in args.datasets:
        print(f"preparing_dataset={dataset} full_pool={args.full_pool} pixel_check={args.check_pixels}", flush=True)
        reports.append(build_one(dataset, args))
        print(f"prepared_dataset={dataset} selected={reports[-1]['selected_count']}", flush=True)
    payload = {
        "schema_version": 1,
        "purpose": "image-cluster-disjoint selective-reuse evaluation",
        "seed": args.seed,
        "reference_count": args.reference_count,
        "pool_count": None if args.full_pool else args.pool_count,
        "full_pool": args.full_pool,
        "pixel_check_enabled": args.check_pixels,
        "sample_num": args.sample_num,
        "selection_offset": args.selection_offset,
        "additional_exclusion_manifest_dirs": [
            str(path.resolve()) for path in args.exclude_manifest_dir
        ],
        "excluded_benchmarks": ["MME"],
        "datasets": reports,
        "all_source_index_overlaps_zero": all(
            row["source_index_overlap_with_all_exclusions"] == 0
            for row in reports
        ),
        "all_image_cluster_overlaps_zero": all(
            row["image_cluster_overlap_with_all_exclusions"] == 0
            for row in reports
        ),
        "all_pixel_overlaps_zero": all(row["pixel_overlap_with_all_exclusions"] == 0 for row in reports) if args.check_pixels else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
