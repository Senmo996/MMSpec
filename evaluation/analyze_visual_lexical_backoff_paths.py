"""Analyze two-token paths for history-aware visual lexical backoff.

The proposed router is deliberately conservative:

* if Token Recycling already has a row for the current root, keep it unchanged;
* otherwise, fill the idle first-hop width from a prompt-native visual inventory;
* use an already available recycled language row for the second hop.

For a non-self adjacent trajectory ``x -> y1 -> y2``, the next state record
contains exactly the language row that was available for ``y1`` while drafting
from ``x``.  Self transitions are excluded because processing ``x`` updates the
same row and makes that reconstruction temporally ambiguous.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


RunKey = Tuple[str, int]


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def load_runs(run_roots: Sequence[str]):
    state_groups: Dict[RunKey, List[dict]] = defaultdict(list)
    inventories: Dict[RunKey, dict] = {}
    resolved_roots = []
    for raw_root in run_roots:
        root = Path(raw_root).resolve()
        resolved_roots.append(str(root))
        state_path = root / "state_records.jsonl"
        inventory_path = root / "inventory_records.jsonl"
        if not state_path.is_file() or not inventory_path.is_file():
            raise FileNotFoundError(
                f"run root must contain state_records.jsonl and "
                f"inventory_records.jsonl: {root}"
            )
        for record in _load_jsonl(inventory_path):
            key = (str(root), int(record["sample_index"]))
            inventories[key] = record
        for state in _load_jsonl(state_path):
            key = (str(root), int(state["sample_index"]))
            state_groups[key].append(state)
    for rows in state_groups.values():
        rows.sort(key=lambda row: int(row["step_index"]))
    return resolved_roots, state_groups, inventories


def build_path_cases(
    state_groups: Dict[RunKey, List[dict]],
    inventories: Dict[RunKey, dict],
    source: str,
    inventory_pool_size: int,
    anchor_budget: int,
    bridge_budget: int,
    *,
    exclude_self_transitions: bool = True,
) -> List[dict]:
    cases = []
    for key, rows in state_groups.items():
        inventory = inventories[key]["inventories"][source][
            :inventory_pool_size
        ]
        for current, following in zip(rows, rows[1:]):
            if int(following["step_index"]) != int(current["step_index"]) + 1:
                continue
            if int(following["root_token"]) != int(current["target_token"]):
                continue
            if exclude_self_transitions and int(current["root_token"]) == int(
                following["root_token"]
            ):
                continue

            baseline_root = [
                int(token)
                for token in current["baseline_candidates"][:anchor_budget]
            ]
            backoff_used = not bool(current["baseline_row_available"])
            backoff_root = (
                [int(token) for token in inventory[:anchor_budget]]
                if backoff_used
                else list(baseline_root)
            )
            bridge = [
                int(token)
                for token in following["baseline_candidates"][:bridge_budget]
            ]
            first_target = int(current["target_token"])
            second_target = int(following["target_token"])
            baseline_root_hit = first_target in baseline_root
            backoff_root_hit = first_target in backoff_root
            bridge_hit = second_target in bridge
            baseline_path_hit = baseline_root_hit and bridge_hit
            backoff_path_hit = backoff_root_hit and bridge_hit
            cases.append(
                {
                    "run_root": key[0],
                    "sample_index": key[1],
                    "question_id": current["question_id"],
                    "step_index": int(current["step_index"]),
                    "high_visual_state": bool(current["high_visual_state"]),
                    "grounding_score": float(current["grounding_score"]),
                    "backoff_used": backoff_used,
                    "bridge_row_available": bool(
                        following["baseline_row_available"]
                    ),
                    "baseline_root_hit": baseline_root_hit,
                    "backoff_root_hit": backoff_root_hit,
                    "bridge_hit": bridge_hit,
                    "baseline_path_hit": baseline_path_hit,
                    "backoff_path_hit": backoff_path_hit,
                    "first_hop_added": not baseline_root_hit and backoff_root_hit,
                    "first_hop_lost": baseline_root_hit and not backoff_root_hit,
                    "path_added": not baseline_path_hit and backoff_path_hit,
                    "path_lost": baseline_path_hit and not backoff_path_hit,
                }
            )
    return cases


def summarize_cases(cases: Iterable[dict]) -> dict:
    cases = list(cases)
    count = len(cases)
    if count == 0:
        return {
            "num_states": 0,
            "baseline_path_coverage": None,
            "backoff_path_coverage": None,
            "path_coverage_gain_pp": None,
            "first_hop_oracle_gain_pp": None,
            "added_paths": 0,
            "lost_paths": 0,
        }
    baseline_path_hits = sum(int(row["baseline_path_hit"]) for row in cases)
    backoff_path_hits = sum(int(row["backoff_path_hit"]) for row in cases)
    first_hop_added = sum(int(row["first_hop_added"]) for row in cases)
    added_paths = sum(int(row["path_added"]) for row in cases)
    return {
        "num_states": count,
        "backoff_used_states": sum(int(row["backoff_used"]) for row in cases),
        "backoff_used_ratio": sum(int(row["backoff_used"]) for row in cases)
        / count,
        "bridge_available_states": sum(
            int(row["bridge_row_available"]) for row in cases
        ),
        "bridge_row_available_ratio": sum(
            int(row["bridge_row_available"]) for row in cases
        )
        / count,
        "baseline_path_hits": baseline_path_hits,
        "backoff_path_hits": backoff_path_hits,
        "baseline_path_coverage": baseline_path_hits / count,
        "backoff_path_coverage": backoff_path_hits / count,
        "path_coverage_gain_pp": 100.0
        * (backoff_path_hits - baseline_path_hits)
        / count,
        "first_hop_added_hits": first_hop_added,
        "first_hop_lost_hits": sum(int(row["first_hop_lost"]) for row in cases),
        "first_hop_oracle_gain_pp": 100.0 * first_hop_added / count,
        "added_paths": added_paths,
        "lost_paths": sum(int(row["path_lost"]) for row in cases),
        "bridge_conversion_of_first_hop_additions": (
            added_paths / first_hop_added if first_hop_added else None
        ),
    }


def analyze(
    run_roots: Sequence[str],
    source: str,
    inventory_pool_size: int,
    anchor_budget: int,
    bridge_budget: int,
    min_gate_states: int,
    min_path_coverage_gain: float,
) -> dict:
    resolved_roots, state_groups, inventories = load_runs(run_roots)
    cases = build_path_cases(
        state_groups,
        inventories,
        source,
        inventory_pool_size,
        anchor_budget,
        bridge_budget,
    )
    subsets = {
        "all": summarize_cases(cases),
        "high_visual": summarize_cases(
            row for row in cases if row["high_visual_state"]
        ),
        "low_visual": summarize_cases(
            row for row in cases if not row["high_visual_state"]
        ),
        "high_visual_backoff_used": summarize_cases(
            row
            for row in cases
            if row["high_visual_state"] and row["backoff_used"]
        ),
    }
    primary = subsets["high_visual_backoff_used"]
    enough_states = primary["num_states"] >= min_gate_states
    path_gain = primary["path_coverage_gain_pp"]
    no_lost_paths = primary["lost_paths"] == 0
    passed = bool(
        enough_states
        and path_gain is not None
        and path_gain >= 100.0 * min_path_coverage_gain
        and no_lost_paths
    )
    decision = "go" if passed else ("no-go" if enough_states else "inconclusive")

    per_run = {}
    for run_root in resolved_roots:
        run_groups = {
            key: rows for key, rows in state_groups.items() if key[0] == run_root
        }
        run_inventories = {
            key: record for key, record in inventories.items() if key[0] == run_root
        }
        run_cases = build_path_cases(
            run_groups,
            run_inventories,
            source,
            inventory_pool_size,
            anchor_budget,
            bridge_budget,
        )
        per_run[run_root] = {
            "all": summarize_cases(run_cases),
            "high_visual": summarize_cases(
                row for row in run_cases if row["high_visual_state"]
            ),
            "high_visual_backoff_used": summarize_cases(
                row
                for row in run_cases
                if row["high_visual_state"] and row["backoff_used"]
            ),
        }

    return {
        "method": "History-aware Visual Lexical Backoff path diagnostic",
        "diagnostic_only": True,
        "run_roots": resolved_roots,
        "num_samples": len(state_groups),
        "configuration": {
            "source": source,
            "inventory_pool_size": inventory_pool_size,
            "anchor_budget": anchor_budget,
            "bridge_budget": bridge_budget,
            "tree_nodes_without_root": anchor_budget
            + anchor_budget * bridge_budget,
            "router": "use visual inventory only when the root has no recycled row",
        },
        "subsets": subsets,
        "per_run": per_run,
        "gate": {
            "decision": decision,
            "pass": passed,
            "evaluated_subset": "high_visual_backoff_used",
            "enough_states": enough_states,
            "no_lost_paths": no_lost_paths,
            "thresholds": {
                "min_gate_states": min_gate_states,
                "min_path_coverage_gain": min_path_coverage_gain,
            },
            "observed": primary,
        },
        "methodology_note": (
            "The first hop uses the frozen dev-selected visual inventory only "
            "when no recycled row exists. For non-self transitions, the next "
            "record exposes the language row available for the second hop."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Visual Lexical Backoff two-token paths"
    )
    parser.add_argument("run_roots", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="visual_max")
    parser.add_argument("--inventory-pool-size", type=int, default=64)
    parser.add_argument("--anchor-budget", type=int, default=4)
    parser.add_argument("--bridge-budget", type=int, default=4)
    parser.add_argument("--min-gate-states", type=int, default=20)
    parser.add_argument("--min-path-coverage-gain", type=float, default=0.05)
    args = parser.parse_args()
    for name in ("inventory_pool_size", "anchor_budget", "bridge_budget"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    summary = analyze(
        args.run_roots,
        args.source,
        args.inventory_pool_size,
        args.anchor_budget,
        args.bridge_budget,
        args.min_gate_states,
        args.min_path_coverage_gain,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps({"gate": summary["gate"]}, indent=2, sort_keys=True))
    print(f"summary_path={output_path}")


if __name__ == "__main__":
    main()
