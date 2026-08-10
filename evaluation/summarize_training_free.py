"""Summarize compact MMSpec speculative-decoding JSONL results."""

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_turns(jsonl_path):
    turns = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for choice in row.get("choices", []):
                n_turns = len(choice.get("new_tokens", []))
                for turn_index in range(n_turns):
                    acceptance = choice.get("acceptance_length", [])
                    traces = choice.get("policy_trace", [])
                    hashes = choice.get("output_hashes", [])
                    turns.append(
                        {
                            "key": (
                                str(row["question_id"]),
                                int(choice.get("index", 0)),
                                turn_index,
                            ),
                            "topic": row.get("topic", "unknown"),
                            "new_tokens": int(choice["new_tokens"][turn_index]),
                            "wall_time": float(choice["wall_time"][turn_index]),
                            "acceptance": (
                                acceptance[turn_index]
                                if turn_index < len(acceptance)
                                else []
                            ),
                            "trace": (
                                traces[turn_index] if turn_index < len(traces) else []
                            ),
                            "output_hash": (
                                hashes[turn_index] if turn_index < len(hashes) else None
                            ),
                        }
                    )
    return turns


def safe_mean(values):
    return float(statistics.fmean(values)) if values else 0.0


def safe_median(values):
    return float(statistics.median(values)) if values else 0.0


def summarize(jsonl_path, reference_path=None):
    turns = load_turns(jsonl_path)
    reference = load_turns(reference_path) if reference_path else turns
    reference_by_key = {turn["key"]: turn for turn in reference}

    acceptances = [
        float(value)
        for turn in turns
        for value in turn["acceptance"]
    ]
    trace_rows = [row for turn in turns for row in turn["trace"]]
    drafted_trace_rows = [
        row for row in trace_rows if int(row.get("used_draft_len", 0)) > 0
    ]
    total_tokens = sum(turn["new_tokens"] for turn in turns)
    total_time = sum(turn["wall_time"] for turn in turns)
    total_iterations = sum(max(len(turn["trace"]) + 1, 1) for turn in turns)
    turn_speeds = [
        turn["new_tokens"] / turn["wall_time"]
        for turn in turns
        if turn["wall_time"] > 0
    ]

    sample_speedups = []
    hash_matches = []
    for turn in turns:
        ref = reference_by_key.get(turn["key"])
        if ref is None:
            continue
        if turn["wall_time"] > 0 and ref["wall_time"] > 0:
            # Outputs are expected to be token-identical; normalize by produced
            # token count as a safeguard for early stop differences.
            speed = turn["new_tokens"] / turn["wall_time"]
            ref_speed = ref["new_tokens"] / ref["wall_time"]
            if ref_speed > 0:
                sample_speedups.append(speed / ref_speed)
        if turn["output_hash"] is not None and ref["output_hash"] is not None:
            hash_matches.append(turn["output_hash"] == ref["output_hash"])

    used_drafts = [float(row.get("used_draft_len", 0)) for row in trace_rows]
    raw_drafts = [float(row.get("raw_draft_len", 0)) for row in trace_rows]
    verified_tree_nodes = [
        float(row["verified_tree_nodes"])
        for row in trace_rows
        if "verified_tree_nodes" in row
    ]
    drafted_tree_nodes = [
        float(row.get("verified_tree_nodes", 0)) for row in drafted_trace_rows
    ]
    grounding_scores = [
        float(row["grounding_score"])
        for row in trace_rows
        if "grounding_score" in row
    ]
    budgets = [float(row.get("budget", 0)) for row in trace_rows]
    accept_ratios = [
        float(row.get("accept_ratio", 0.0))
        for row in trace_rows
        if row.get("used_draft_len", 0) > 0
    ]
    conditional_rows = [
        row
        for row in trace_rows
        if int(row.get("conditional_transition_count", 0)) > 0
    ]
    conditional_overlaps = [
        float(row["conditional_global_overlap"])
        for row in conditional_rows
        if row.get("conditional_global_overlap") is not None
    ]
    residual_candidate_counts = [
        int(row.get("root_residual_candidate_count", 0)) for row in trace_rows
    ]
    visual_lexical_rows = [
        row
        for row in trace_rows
        if bool(row.get("visual_lexical_backoff_active", False))
    ]
    visual_hst_rows = [
        row
        for row in trace_rows
        if bool(row.get("visual_hst_backoff_active", False))
    ]
    verification_recheck_rows = [
        row
        for row in trace_rows
        if bool(row.get("verification_boundary_recheck", False))
    ]
    verification_changed_rows = [
        row
        for row in verification_recheck_rows
        if bool(row.get("verification_recheck_changed_token", False))
    ]
    verification_rebased_rows = [
        row
        for row in verification_recheck_rows
        if bool(row.get("verification_prefix_rebased", False))
    ]
    compact_path_repair_rows = [
        row
        for row in trace_rows
        if bool(row.get("verification_compact_path_repair", False))
    ]
    compact_root_recheck_rows = [
        row
        for row in trace_rows
        if bool(row.get("verification_compact_root_recheck", False))
    ]

    summary = {
        "num_records": len({turn["key"][0] for turn in turns}),
        "num_turns": len(turns),
        "total_new_tokens": total_tokens,
        "total_wall_time": total_time,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
        "median_turn_tokens_per_second": safe_median(turn_speeds),
        "avg_accept_length": safe_mean(acceptances),
        "median_accept_length": safe_median(acceptances),
        "max_accept_length": max(acceptances, default=0.0),
        "p90_accept_length": percentile(acceptances, 0.9),
        "avg_tokens_per_iteration": (
            total_tokens / total_iterations if total_iterations > 0 else 0.0
        ),
        "avg_sample_speedup": safe_mean(sample_speedups),
        "median_sample_speedup": safe_median(sample_speedups),
        "max_sample_speedup": max(sample_speedups, default=0.0),
        "sample_speedup_gt_1_ratio": (
            sum(value > 1.0 for value in sample_speedups) / len(sample_speedups)
            if sample_speedups
            else 0.0
        ),
        "accept_le_0_5_ratio": (
            sum(value <= 0.5 for value in acceptances) / len(acceptances)
            if acceptances
            else 0.0
        ),
        "output_hash_match_ratio": (
            sum(hash_matches) / len(hash_matches) if hash_matches else None
        ),
        "avg_raw_draft_len": safe_mean(raw_drafts),
        "avg_used_draft_len": safe_mean(used_drafts),
        "avg_verified_tree_nodes": safe_mean(verified_tree_nodes),
        "avg_verified_tree_nodes_when_drafted": safe_mean(drafted_tree_nodes),
        "total_verified_tree_nodes": sum(verified_tree_nodes),
        "verified_tree_nodes_per_output_token": (
            sum(verified_tree_nodes) / total_tokens if total_tokens > 0 else 0.0
        ),
        "drafted_iteration_ratio": (
            len(drafted_trace_rows) / len(trace_rows) if trace_rows else 0.0
        ),
        "conditional_transition_active_ratio": (
            len(conditional_rows) / len(trace_rows) if trace_rows else 0.0
        ),
        "avg_conditional_transition_count": safe_mean(
            [float(row["conditional_transition_count"]) for row in conditional_rows]
        ),
        "avg_conditional_mix_weight": safe_mean(
            [float(row.get("conditional_mix_weight", 0.0)) for row in conditional_rows]
        ),
        "avg_conditional_global_overlap": safe_mean(conditional_overlaps),
        "root_residual_active_ratio": (
            sum(value > 0 for value in residual_candidate_counts)
            / len(residual_candidate_counts)
            if residual_candidate_counts
            else 0.0
        ),
        "avg_root_residual_candidate_count": safe_mean(
            [float(value) for value in residual_candidate_counts]
        ),
        "visual_lexical_backoff_active_ratio": (
            len(visual_lexical_rows) / len(trace_rows) if trace_rows else 0.0
        ),
        "visual_lexical_backoff_num_iterations": len(visual_lexical_rows),
        "visual_lexical_backoff_accept_ratio": (
            sum(int(row.get("accept_len", 0)) > 0 for row in visual_lexical_rows)
            / len(visual_lexical_rows)
            if visual_lexical_rows
            else 0.0
        ),
        "visual_lexical_backoff_avg_accept_length": safe_mean(
            [float(row.get("accept_len", 0)) for row in visual_lexical_rows]
        ),
        "visual_lexical_backoff_avg_verified_tree_nodes": safe_mean(
            [float(row.get("verified_tree_nodes", 0)) for row in visual_lexical_rows]
        ),
        "visual_hst_backoff_active_ratio": (
            len(visual_hst_rows) / len(trace_rows) if trace_rows else 0.0
        ),
        "visual_hst_backoff_num_iterations": len(visual_hst_rows),
        "visual_hst_backoff_accept_ratio": (
            sum(int(row.get("accept_len", 0)) > 0 for row in visual_hst_rows)
            / len(visual_hst_rows)
            if visual_hst_rows
            else 0.0
        ),
        "visual_hst_backoff_avg_accept_length": safe_mean(
            [float(row.get("accept_len", 0)) for row in visual_hst_rows]
        ),
        "visual_hst_backoff_avg_verified_tree_nodes": safe_mean(
            [float(row.get("verified_tree_nodes", 0)) for row in visual_hst_rows]
        ),
        "verification_recheck_num_iterations": len(verification_recheck_rows),
        "verification_recheck_ratio": (
            len(verification_recheck_rows) / len(trace_rows)
            if trace_rows
            else 0.0
        ),
        "verification_recheck_changed_token_num_iterations": len(
            verification_changed_rows
        ),
        "verification_recheck_changed_token_ratio": (
            len(verification_changed_rows) / len(verification_recheck_rows)
            if verification_recheck_rows
            else 0.0
        ),
        "verification_prefix_rebase_num_iterations": len(
            verification_rebased_rows
        ),
        "verification_total_replayed_tokens": sum(
            int(row.get("verification_replayed_tokens", 0))
            for row in verification_recheck_rows
        ),
        "verification_avg_discarded_accept_length": safe_mean(
            [
                float(row.get("verification_packed_accept_len", 0))
                for row in verification_recheck_rows
            ]
        ),
        "verification_total_discarded_accepted_tokens": sum(
            int(row.get("verification_packed_accept_len", 0))
            for row in verification_recheck_rows
        ),
        "verification_compact_path_repair_num_iterations": len(
            compact_path_repair_rows
        ),
        "verification_compact_path_repair_ratio": (
            len(compact_path_repair_rows) / len(trace_rows)
            if trace_rows
            else 0.0
        ),
        "verification_compact_path_truncated_num_iterations": sum(
            bool(row.get("verification_compact_path_truncated", False))
            for row in compact_path_repair_rows
        ),
        "verification_compact_path_changed_correction_num_iterations": sum(
            bool(
                row.get(
                    "verification_compact_path_changed_correction", False
                )
            )
            for row in compact_path_repair_rows
        ),
        "verification_compact_path_avg_packed_accept_length": safe_mean(
            [
                float(row.get("verification_packed_accept_len", 0))
                for row in compact_path_repair_rows
            ]
        ),
        "verification_compact_path_avg_repaired_accept_length": safe_mean(
            [
                float(row.get("verification_repaired_accept_len", 0))
                for row in compact_path_repair_rows
            ]
        ),
        "verification_compact_root_recheck_num_iterations": len(
            compact_root_recheck_rows
        ),
        "verification_compact_root_recheck_ratio": (
            len(compact_root_recheck_rows) / len(trace_rows)
            if trace_rows
            else 0.0
        ),
        "verification_compact_root_changed_token_num_iterations": sum(
            bool(row.get("verification_compact_root_changed_token", False))
            for row in compact_root_recheck_rows
        ),
        "avg_draft_accept_ratio": safe_mean(accept_ratios),
        "avg_grounding_score": safe_mean(grounding_scores),
        "p90_grounding_score": percentile(grounding_scores, 0.9),
        "avg_draft_budget": safe_mean(budgets),
        "jsonl_path": str(Path(jsonl_path).resolve()),
        "reference_jsonl_path": (
            str(Path(reference_path).resolve()) if reference_path else None
        ),
    }

    by_topic = {}
    for topic in sorted({turn["topic"] for turn in turns}):
        topic_turns = [turn for turn in turns if turn["topic"] == topic]
        topic_tokens = sum(turn["new_tokens"] for turn in topic_turns)
        topic_time = sum(turn["wall_time"] for turn in topic_turns)
        by_topic[topic] = {
            "num_turns": len(topic_turns),
            "tokens_per_second": (
                topic_tokens / topic_time if topic_time > 0 else 0.0
            ),
        }
    summary["by_topic"] = by_topic
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    args = parser.parse_args()
    summary = summarize(args.jsonl_path, args.reference)
    output_path = Path(args.output) if args.output else Path(args.jsonl_path).with_name("summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
