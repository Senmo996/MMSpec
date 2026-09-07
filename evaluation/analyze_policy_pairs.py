"""Paired, sample-level speed comparison for training-free policy runs."""

import argparse
import json
import random
import statistics
from pathlib import Path

try:
    from .summarize_training_free import load_turns
except ImportError:  # Direct script execution from MMSpec/evaluation.
    from summarize_training_free import load_turns


def _sample_rows(jsonl_path):
    grouped = {}
    for turn in load_turns(jsonl_path):
        question_id, choice_index, _ = turn["key"]
        key = (question_id, choice_index)
        row = grouped.setdefault(
            key,
            {
                "topic": turn["topic"],
                "tokens": 0,
                "wall_time": 0.0,
                "hashes": [],
            },
        )
        row["tokens"] += turn["new_tokens"]
        row["wall_time"] += turn["wall_time"]
        row["hashes"].append(turn["output_hash"])
    return grouped


def _bootstrap_mean_ci(values, samples=10000, seed=0):
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(samples):
        means.append(
            statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        )
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def _summarize_rows(rows, bootstrap_samples):
    ratios = [row["speed_ratio"] for row in rows]
    return {
        "num_paired_samples": len(rows),
        "mean_speed_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "median_speed_ratio": statistics.median(ratios) if ratios else 0.0,
        "bootstrap_95_ci": _bootstrap_mean_ci(
            ratios, samples=bootstrap_samples
        ),
        "speedup_gt_1_ratio": (
            sum(value > 1.0 for value in ratios) / len(ratios) if ratios else 0.0
        ),
    }


def compare(
    policy_path,
    baseline_path,
    bootstrap_samples=10000,
    exact_reference_path=None,
):
    policy = _sample_rows(policy_path)
    baseline = _sample_rows(baseline_path)
    exact_reference = (
        _sample_rows(exact_reference_path) if exact_reference_path else None
    )
    paired_keys = sorted(set(policy) & set(baseline))

    rows = []
    for key in paired_keys:
        current = policy[key]
        reference = baseline[key]
        if current["wall_time"] <= 0 or reference["wall_time"] <= 0:
            continue
        current_speed = current["tokens"] / current["wall_time"]
        reference_speed = reference["tokens"] / reference["wall_time"]
        if reference_speed <= 0:
            continue
        row = {
            "key": key,
            "topic": current["topic"],
            "speed_ratio": current_speed / reference_speed,
            "hash_match": current["hashes"] == reference["hashes"],
        }
        if exact_reference is not None and key in exact_reference:
            exact_hashes = exact_reference[key]["hashes"]
            row["both_match_exact_reference"] = (
                current["hashes"] == exact_hashes
                and reference["hashes"] == exact_hashes
            )
        rows.append(row)

    result = {
        **_summarize_rows(rows, bootstrap_samples),
        "sample_hash_match_ratio": (
            sum(row["hash_match"] for row in rows) / len(rows) if rows else None
        ),
    }

    by_topic = {}
    for topic in sorted({row["topic"] for row in rows}):
        topic_ratios = [
            row["speed_ratio"] for row in rows if row["topic"] == topic
        ]
        by_topic[topic] = {
            "num_samples": len(topic_ratios),
            "mean_speed_ratio": statistics.fmean(topic_ratios),
            "median_speed_ratio": statistics.median(topic_ratios),
        }
    result["by_topic"] = by_topic
    if exact_reference is not None:
        exact_rows = [
            row for row in rows if row.get("both_match_exact_reference", False)
        ]
        result["both_match_exact_reference_ratio"] = (
            len(exact_rows) / len(rows) if rows else 0.0
        )
        result["exact_reference_subset"] = _summarize_rows(
            exact_rows, bootstrap_samples
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--baseline", default="broad")
    parser.add_argument("--policies", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--exact-reference",
        help="Policy used to report a subset where both compared outputs match it.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.output_root)
    baseline_path = root / args.baseline / "results.jsonl"
    exact_reference_path = (
        root / args.exact_reference / "results.jsonl"
        if args.exact_reference
        else None
    )
    payload = {
        "baseline_policy": args.baseline,
        "exact_reference_policy": args.exact_reference,
        "comparisons": {},
    }
    for policy in [value.strip() for value in args.policies.split(",") if value.strip()]:
        payload["comparisons"][policy] = compare(
            root / policy / "results.jsonl",
            baseline_path,
            bootstrap_samples=args.bootstrap_samples,
            exact_reference_path=exact_reference_path,
        )

    output_path = (
        Path(args.output) if args.output else root / f"paired_vs_{args.baseline}.json"
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
