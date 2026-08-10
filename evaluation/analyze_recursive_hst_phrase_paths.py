"""Offline phrase-continuation audit for recursive Hidden-State Transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from transformers import AutoTokenizer


PHRASE_POLICIES = (
    "bridge2_hybrid1",
    "phrase_only",
    "bridge2_phrase1",
    "phrase2_bridge1",
    "bridge1_phrase1_hybrid1",
    "hybrid2_phrase1",
    "phrase2_hybrid1",
)


def prompt_lookup_candidates(
    source_token_ids: Sequence[int],
    generated_prefix: Sequence[int],
    *,
    max_ngram: int,
    limit: int,
    excluded_token_ids: Iterable[int] = (),
) -> List[int]:
    """Return successors of the longest generated suffix found in the prompt."""

    source = [int(token) for token in source_token_ids]
    prefix = [int(token) for token in generated_prefix]
    excluded = {int(token) for token in excluded_token_ids}
    if not source or not prefix or int(limit) <= 0:
        return []
    largest = min(int(max_ngram), len(prefix), max(len(source) - 1, 0))
    for ngram in range(largest, 0, -1):
        suffix = prefix[-ngram:]
        successors = []
        seen = set()
        for start in range(len(source) - ngram - 1, -1, -1):
            if source[start : start + ngram] != suffix:
                continue
            successor = int(source[start + ngram])
            if successor in excluded or successor in seen:
                continue
            successors.append(successor)
            seen.add(successor)
            if len(successors) >= int(limit):
                return successors
        if successors:
            return successors
    return []


def fuse_sources(
    sources: Sequence[Sequence[int]],
    slots: Sequence[int],
    *,
    width: int,
) -> List[int]:
    """Allocate source slots, then fill remaining width in source order."""

    if len(sources) != len(slots):
        raise ValueError("sources and slots must have equal length")
    selected = []
    seen = set()

    def append(token: int) -> None:
        token = int(token)
        if token not in seen and len(selected) < int(width):
            selected.append(token)
            seen.add(token)

    for source, slot_count in zip(sources, slots):
        for token in list(source)[: int(slot_count)]:
            append(token)
    for source, slot_count in zip(sources, slots):
        for token in list(source)[int(slot_count) :]:
            append(token)
            if len(selected) >= int(width):
                return selected
    return selected


def build_phrase_policy_candidates(
    *,
    bridge: Sequence[int],
    phrase: Sequence[int],
    hybrid: Sequence[int],
    width: int,
) -> Dict[str, List[int]]:
    return {
        "bridge2_hybrid1": fuse_sources(
            [bridge, hybrid], [2, 1], width=width
        ),
        "phrase_only": list(phrase)[:width],
        "bridge2_phrase1": fuse_sources(
            [bridge, phrase, hybrid], [2, 1, 0], width=width
        ),
        "phrase2_bridge1": fuse_sources(
            [phrase, bridge, hybrid], [2, 1, 0], width=width
        ),
        "bridge1_phrase1_hybrid1": fuse_sources(
            [bridge, phrase, hybrid], [1, 1, 1], width=width
        ),
        "hybrid2_phrase1": fuse_sources(
            [hybrid, phrase, bridge], [2, 1, 0], width=width
        ),
        "phrase2_hybrid1": fuse_sources(
            [phrase, hybrid, bridge], [2, 1, 0], width=width
        ),
    }


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_map(data_folder: Path) -> Dict[str, str]:
    result = {}
    with (data_folder / "mmspec.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            turns = row.get("turns", [row.get("prompt", "")])
            result[str(row["id"])] = str(turns[0] if turns else "")
    return result


def analyze_run(
    run_dir: Path,
    *,
    tokenizer,
    questions: Dict[str, str],
    max_ngram: int,
    width: int,
) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    records = _read_jsonl(run_dir / "ranker_records.jsonl")
    by_sample: Dict[int, List[dict]] = {}
    for record in records:
        by_sample.setdefault(int(record["sample_index"]), []).append(record)
    for sample_records in by_sample.values():
        sample_records.sort(key=lambda row: int(row["step_index"]))

    configs = list(summary["configuration"]["recursive_configs"])
    counters = {
        config: {
            policy: {
                "num_states_with_next": 0,
                "first_hits": 0,
                "phrase_available_first_hits": 0,
                "path_hits": 0,
            }
            for policy in PHRASE_POLICIES
        }
        for config in configs
    }
    excluded = set(int(token) for token in tokenizer.all_special_ids)

    for sample_records in by_sample.values():
        if not sample_records:
            continue
        question_id = str(sample_records[0]["question_id"])
        source_ids = tokenizer(
            questions[question_id], add_special_tokens=False
        ).input_ids
        for index, record in enumerate(sample_records[:-1]):
            if not record.get("high_visual_state", False):
                continue
            if record.get("baseline_row_available", False):
                continue
            next_record = sample_records[index + 1]
            if int(next_record["step_index"]) != int(record["step_index"]) + 1:
                continue
            generated_prefix = [
                int(previous["root_token"])
                for previous in sample_records[: index + 1]
            ]
            actual_first = int(record["target_token"])
            actual_second = int(next_record["target_token"])
            for config in configs:
                recursive = record.get("recursive_paths", {}).get(config)
                if recursive is None:
                    continue
                first_hit = actual_first in {
                    int(token) for token in recursive["first_candidates"]
                }
                branch = recursive["branches"].get(str(actual_first), {})
                bridge = branch.get("bridge_only", [])
                hybrid = branch.get("hybrid_hst", [])
                phrase = (
                    prompt_lookup_candidates(
                        source_ids,
                        [*generated_prefix, actual_first],
                        max_ngram=max_ngram,
                        limit=width,
                        excluded_token_ids=excluded,
                    )
                    if first_hit
                    else []
                )
                policy_candidates = build_phrase_policy_candidates(
                    bridge=bridge,
                    phrase=phrase,
                    hybrid=hybrid,
                    width=width,
                )
                for policy, candidates in policy_candidates.items():
                    counter = counters[config][policy]
                    counter["num_states_with_next"] += 1
                    counter["first_hits"] += int(first_hit)
                    counter["phrase_available_first_hits"] += int(
                        first_hit and bool(phrase)
                    )
                    counter["path_hits"] += int(
                        first_hit and actual_second in set(candidates)
                    )

    metrics = {}
    for config, policy_counters in counters.items():
        metrics[config] = {}
        for policy, counter in policy_counters.items():
            count = counter["num_states_with_next"]
            first_hits = counter["first_hits"]
            path_hits = counter["path_hits"]
            metrics[config][policy] = {
                **counter,
                "first_hit_rate": first_hits / count if count else None,
                "phrase_available_given_first_rate": (
                    counter["phrase_available_first_hits"] / first_hits
                    if first_hits
                    else None
                ),
                "path_hit_rate": path_hits / count if count else None,
                "second_given_first_rate": (
                    path_hits / first_hits if first_hits else None
                ),
            }

    first_metrics = summary["metrics"]["high_visual_row_absent"][
        "observed_visual"
    ]
    specificity = summary["paired_controls"]["mismatched_visual"]
    return {
        "run_dir": str(run_dir.resolve()),
        "topic_offset": summary["configuration"]["topic_offset"],
        "num_samples": summary["num_samples"],
        "max_ngram": int(max_ngram),
        "width": int(width),
        "configs": configs,
        "first_hop_metrics": {config: first_metrics[config] for config in configs},
        "paired_specificity": {config: specificity[config] for config in configs},
        "phrase_path_metrics": metrics,
    }


def select_dev_action(
    dev: dict,
    *,
    min_top3_hit_rate: float,
    min_path_hit_rate: float,
) -> tuple[Optional[str], Optional[str], List[dict]]:
    eligible = []
    for config in dev["configs"]:
        top3 = dev["first_hop_metrics"][config]["top3_hit_rate"]
        specificity = dev["paired_specificity"][config][
            "observed_minus_control_pp"
        ]
        for policy, path in dev["phrase_path_metrics"][config].items():
            if (
                top3 >= min_top3_hit_rate
                and specificity is not None
                and specificity > 0.0
                and path["path_hit_rate"] >= min_path_hit_rate
            ):
                eligible.append(
                    {
                        "config": config,
                        "policy": policy,
                        "top3_hit_rate": top3,
                        "specificity_delta_pp": specificity,
                        "path_hit_rate": path["path_hit_rate"],
                        "phrase_available_given_first_rate": path[
                            "phrase_available_given_first_rate"
                        ],
                    }
                )
    if not eligible:
        return None, None, []
    selected = max(
        eligible,
        key=lambda row: (
            row["path_hit_rate"],
            row["top3_hit_rate"],
            row["specificity_delta_pp"],
            row["config"],
            row["policy"],
        ),
    )
    return selected["config"], selected["policy"], eligible


def summarize_splits(
    split_results: Dict[str, dict],
    *,
    min_top3_hit_rate: float,
    min_path_hit_rate: float,
) -> dict:
    selected_config, selected_policy, eligible = select_dev_action(
        split_results["dev_offset0"],
        min_top3_hit_rate=min_top3_hit_rate,
        min_path_hit_rate=min_path_hit_rate,
    )
    reports = {}
    gates = {}
    for name, result in split_results.items():
        if selected_config is None or selected_policy is None:
            reports[name] = None
            gates[name] = {"pass": False}
            continue
        first = result["first_hop_metrics"][selected_config]
        specificity = result["paired_specificity"][selected_config]
        path = result["phrase_path_metrics"][selected_config][selected_policy]
        reports[name] = {
            "topic_offset": result["topic_offset"],
            "first_hop": first,
            "paired_specificity": specificity,
            "phrase_path": path,
        }
        gates[name] = {
            "top3_pass": first["top3_hit_rate"] >= min_top3_hit_rate,
            "specificity_pass": specificity["observed_minus_control_pp"] is not None
            and specificity["observed_minus_control_pp"] > 0.0,
            "path_pass": path["path_hit_rate"] >= min_path_hit_rate,
        }
        gates[name]["pass"] = all(gates[name].values())
    passed = selected_config is not None and all(
        row["pass"] for row in gates.values()
    )
    return {
        "method": "Question phrase lookup for recursive HST paths",
        "selected_config": selected_config,
        "selected_policy": selected_policy,
        "dev_eligible_action_count": len(eligible),
        "dev_eligible_actions": eligible,
        "split_reports": reports,
        "all_policy_metrics": split_results,
        "gate": {
            "decision": "go" if passed else "no-go",
            "pass": passed,
            "end_to_end_authorized": passed,
            "thresholds": {
                "min_top3_hit_rate": min_top3_hit_rate,
                "min_path_hit_rate": min_path_hit_rate,
                "positive_paired_specificity_required": True,
            },
            "split_results": gates,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--fresh-run", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--min-top3-hit-rate", type=float, default=0.15)
    parser.add_argument("--min-path-hit-rate", type=float, default=0.05)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    questions = _question_map(args.data_folder)
    split_results = {
        "dev_offset0": analyze_run(
            args.dev_run,
            tokenizer=tokenizer,
            questions=questions,
            max_ngram=args.max_ngram,
            width=args.width,
        ),
        "validation_offset1": analyze_run(
            args.validation_run,
            tokenizer=tokenizer,
            questions=questions,
            max_ngram=args.max_ngram,
            width=args.width,
        ),
        "confirmation_offset2": analyze_run(
            args.confirmation_run,
            tokenizer=tokenizer,
            questions=questions,
            max_ngram=args.max_ngram,
            width=args.width,
        ),
        "fresh_offset3": analyze_run(
            args.fresh_run,
            tokenizer=tokenizer,
            questions=questions,
            max_ngram=args.max_ngram,
            width=args.width,
        ),
    }
    summary = summarize_splits(
        split_results,
        min_top3_hit_rate=args.min_top3_hit_rate,
        min_path_hit_rate=args.min_path_hit_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary_path={args.output}")


if __name__ == "__main__":
    main()
