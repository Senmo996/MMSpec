"""Audit deterministic MMSpec shuffle ranges for sample/image disjointness."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_range(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("range must be LABEL:START:COUNT")
    label, start_text, count_text = parts
    if not label:
        raise argparse.ArgumentTypeError("range label must not be empty")
    try:
        start, count = int(start_text), int(count_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("range bounds must be integers") from error
    if start < 0 or count <= 0:
        raise argparse.ArgumentTypeError("range requires START>=0 and COUNT>0")
    return label, start, count


def read_metadata(path: Path) -> list[dict]:
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


def image_identity(row: dict) -> str:
    for key in ("image_id", "image", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("MMSpec row lacks image_id, image, and id")


def audit(rows: list[dict], *, seed: int, ranges: list[tuple[str, int, int]]) -> dict:
    order = list(range(len(rows)))
    random.Random(int(seed)).shuffle(order)
    selections = {}
    for label, start, count in ranges:
        stop = start + count
        if stop > len(order):
            raise ValueError(
                f"range {label}=[{start}:{stop}] exceeds {len(order)} rows"
            )
        indices = order[start:stop]
        clusters = [image_identity(rows[index]) for index in indices]
        selections[label] = {
            "start": start,
            "count": count,
            "stop": stop,
            "source_indices": indices,
            "unique_source_indices": len(set(indices)),
            "unique_image_clusters": len(set(clusters)),
            "image_clusters": clusters,
        }

    pairwise = []
    labels = list(selections)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_row, right_row = selections[left], selections[right]
            source_overlap = set(left_row["source_indices"]) & set(
                right_row["source_indices"]
            )
            cluster_overlap = set(left_row["image_clusters"]) & set(
                right_row["image_clusters"]
            )
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "source_index_overlap": len(source_overlap),
                    "image_cluster_overlap": len(cluster_overlap),
                }
            )
    all_unique_within_range = all(
        selected["unique_source_indices"] == selected["count"]
        and selected["unique_image_clusters"] == selected["count"]
        for selected in selections.values()
    )
    all_pairwise_overlaps_zero = all(
        row["source_index_overlap"] == 0 and row["image_cluster_overlap"] == 0
        for row in pairwise
    )
    return {
        "schema_version": 1,
        "dataset": "MMSpec",
        "seed": int(seed),
        "num_rows": len(rows),
        "ranges": selections,
        "pairwise_overlaps": pairwise,
        "all_unique_within_range": all_unique_within_range,
        "all_pairwise_overlaps_zero": all_pairwise_overlaps_zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--range", dest="ranges", action="append", type=parse_range, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(read_metadata(args.data_jsonl), seed=args.seed, ranges=args.ranges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "num_rows": payload["num_rows"],
                "all_unique_within_range": payload["all_unique_within_range"],
                "all_pairwise_overlaps_zero": payload[
                    "all_pairwise_overlaps_zero"
                ],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not payload["all_unique_within_range"]:
        raise SystemExit(2)
    if not payload["all_pairwise_overlaps_zero"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
