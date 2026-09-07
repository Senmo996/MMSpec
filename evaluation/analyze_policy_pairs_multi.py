"""Combine paired policy comparisons across disjoint evaluation runs."""

import argparse
import json
from pathlib import Path

try:
    from .analyze_policy_pairs import _sample_rows, _summarize_rows
except ImportError:  # Direct script execution from MMSpec/evaluation.
    from analyze_policy_pairs import _sample_rows, _summarize_rows


def _load_runs(roots, policy):
    merged = {}
    for run_index, root in enumerate(roots):
        rows = _sample_rows(root / policy / "results.jsonl")
        for key, row in rows.items():
            merged[(run_index, *key)] = row
    return merged


def compare(roots, policy, baseline, exact_reference, bootstrap_samples):
    current_rows = _load_runs(roots, policy)
    baseline_rows = _load_runs(roots, baseline)
    exact_rows = _load_runs(roots, exact_reference)
    paired_keys = sorted(set(current_rows) & set(baseline_rows))

    rows = []
    for key in paired_keys:
        current = current_rows[key]
        reference = baseline_rows[key]
        exact = exact_rows.get(key)
        if current["wall_time"] <= 0 or reference["wall_time"] <= 0:
            continue
        current_speed = current["tokens"] / current["wall_time"]
        reference_speed = reference["tokens"] / reference["wall_time"]
        if reference_speed <= 0:
            continue
        rows.append(
            {
                "run_index": key[0],
                "topic": current["topic"],
                "speed_ratio": current_speed / reference_speed,
                "hash_match": current["hashes"] == reference["hashes"],
                "both_match_exact_reference": bool(
                    exact is not None
                    and current["hashes"] == exact["hashes"]
                    and reference["hashes"] == exact["hashes"]
                ),
            }
        )

    exact_subset = [row for row in rows if row["both_match_exact_reference"]]
    result = {
        **_summarize_rows(rows, bootstrap_samples),
        "sample_hash_match_ratio": (
            sum(row["hash_match"] for row in rows) / len(rows) if rows else None
        ),
        "both_match_exact_reference_ratio": (
            len(exact_subset) / len(rows) if rows else 0.0
        ),
        "exact_reference_subset": _summarize_rows(
            exact_subset, bootstrap_samples
        ),
        "by_run": {},
        "by_topic": {},
    }
    for run_index, root in enumerate(roots):
        run_rows = [row for row in rows if row["run_index"] == run_index]
        result["by_run"][root.name] = _summarize_rows(
            run_rows, bootstrap_samples
        )
    for topic in sorted({row["topic"] for row in rows}):
        topic_rows = [row for row in rows if row["topic"] == topic]
        result["by_topic"][topic] = _summarize_rows(
            topic_rows, bootstrap_samples
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_roots", nargs="+")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--exact-reference", default="target")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = [Path(value) for value in args.output_roots]
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    payload = {
        "output_roots": [str(root.resolve()) for root in roots],
        "baseline_policy": args.baseline,
        "exact_reference_policy": args.exact_reference,
        "comparisons": {
            policy: compare(
                roots,
                policy,
                args.baseline,
                args.exact_reference,
                args.bootstrap_samples,
            )
            for policy in policies
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
