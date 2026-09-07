"""Build fixed-benchmark manifests disjoint from all declared prior sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MMSPEC_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MMSPEC_ROOT.parent
for path in (str(MMSPEC_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.eval_sam_grounded_fixed_multi import _image_cluster_identity
from new_dream.evaluation.eval_llava_fixed_multi import (
    dataset_slug,
    load_base_dataset,
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
    pool = read_jsonl(pool_path)
    base_dataset = load_base_dataset(dataset)

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
    eligible = []
    eligible_clusters = set()
    rejected_source_overlap = 0
    rejected_cluster_overlap = 0
    rejected_duplicate_cluster = 0
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
        chosen.append(copied)

    output_path = (
        args.output_dir
        / f"{slug}_seed{args.seed}_n{args.sample_num}.jsonl"
    )
    write_jsonl(output_path, chosen)
    chosen_source_indices = {int(row["source_index"]) for row in chosen}
    chosen_clusters = {cluster for _row, cluster in chosen_pairs}
    return {
        "dataset": dataset,
        "slug": slug,
        "reference_manifest": str(reference_path.resolve()),
        "candidate_pool_manifest": str(pool_path.resolve()),
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

    reports = [build_one(dataset, args) for dataset in args.datasets]
    payload = {
        "schema_version": 1,
        "purpose": "image-cluster-disjoint selective-reuse evaluation",
        "seed": args.seed,
        "reference_count": args.reference_count,
        "pool_count": args.pool_count,
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
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
