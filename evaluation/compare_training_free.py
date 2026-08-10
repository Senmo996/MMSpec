"""Collect per-policy summary files into a ranked comparison."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "policy": summary_path.parent.name,
                "tokens_per_second": payload.get("tokens_per_second", 0.0),
                "avg_sample_speedup": payload.get("avg_sample_speedup", 0.0),
                "median_sample_speedup": payload.get("median_sample_speedup", 0.0),
                "output_hash_match_ratio": payload.get("output_hash_match_ratio"),
                "avg_accept_length": payload.get("avg_accept_length", 0.0),
                "avg_tokens_per_iteration": payload.get("avg_tokens_per_iteration", 0.0),
                "avg_used_draft_len": payload.get("avg_used_draft_len", 0.0),
                "avg_verified_tree_nodes": payload.get("avg_verified_tree_nodes", 0.0),
                "avg_verified_tree_nodes_when_drafted": payload.get(
                    "avg_verified_tree_nodes_when_drafted", 0.0
                ),
                "verified_tree_nodes_per_output_token": payload.get(
                    "verified_tree_nodes_per_output_token", 0.0
                ),
                "drafted_iteration_ratio": payload.get(
                    "drafted_iteration_ratio", 0.0
                ),
                "root_residual_active_ratio": payload.get(
                    "root_residual_active_ratio", 0.0
                ),
                "visual_lexical_backoff_active_ratio": payload.get(
                    "visual_lexical_backoff_active_ratio", 0.0
                ),
                "visual_lexical_backoff_accept_ratio": payload.get(
                    "visual_lexical_backoff_accept_ratio", 0.0
                ),
                "visual_hst_backoff_active_ratio": payload.get(
                    "visual_hst_backoff_active_ratio", 0.0
                ),
                "visual_hst_backoff_accept_ratio": payload.get(
                    "visual_hst_backoff_accept_ratio", 0.0
                ),
                "avg_draft_accept_ratio": payload.get("avg_draft_accept_ratio", 0.0),
                "avg_grounding_score": payload.get("avg_grounding_score", 0.0),
                "summary_path": str(summary_path.resolve()),
            }
        )
    rows.sort(key=lambda row: row["avg_sample_speedup"], reverse=True)
    result = {"reference_policy": "target", "ranked_policies": rows}
    output = Path(args.output) if args.output else root / "comparison.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
