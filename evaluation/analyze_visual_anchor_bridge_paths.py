"""Evaluate Visual Anchor + Language Bridge two-token path coverage.

The input is one or more ``probe_records.jsonl`` files produced by
``eval_counterfactual_visual_probes.py``.  For adjacent target states
``x -> y1 -> y2``:

* the baseline path uses the recycled full-view row for ``x -> y1``;
* VA+LB uses the counterfactual visual row only for ``x -> y1``;
* both use the same full-view language row for ``y1 -> y2``.

The next record exposes the language row that was available for ``y1``.  This
is temporally exact whenever ``x != y1``, because processing state ``x`` only
updates row ``x``.  Self transitions are therefore excluded by default.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_positive_ints(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        parsed = int(item.strip())
        if parsed <= 0:
            raise argparse.ArgumentTypeError("budgets must be positive integers")
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return result


def load_record_groups(paths: Sequence[str]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for raw_path in paths:
        path = str(Path(raw_path).resolve())
        with open(path, "r", encoding="utf-8") as record_file:
            for line in record_file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                groups[(path, str(row["question_id"]))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["step_index"]))
    return groups


def build_path_cases(
    groups: Dict[Tuple[str, str], List[dict]],
    anchor_budget: int,
    bridge_budget: int,
    require_counterfactual_history: bool = True,
    exclude_self_transitions: bool = True,
) -> List[dict]:
    """Construct temporally valid two-token path-coverage cases."""

    anchor_key = str(anchor_budget)
    bridge_key = str(bridge_budget)
    cases = []
    for (source_path, question_id), rows in groups.items():
        for current, following in zip(rows, rows[1:]):
            if int(following["step_index"]) != int(current["step_index"]) + 1:
                continue
            if int(following["root_token"]) != int(current["target_token"]):
                continue
            if exclude_self_transitions and int(current["root_token"]) == int(
                following["root_token"]
            ):
                continue
            if require_counterfactual_history and not bool(
                current.get("has_counterfactual_history", False)
            ):
                continue

            root_row = current.get("coverage", {}).get(anchor_key)
            if root_row is None:
                continue
            bridge_row = following.get("coverage", {}).get(bridge_key)
            baseline_root = [int(token) for token in root_row["baseline_candidates"]]
            visual_root = [int(token) for token in root_row["cover_candidates"]]
            baseline_bridge = (
                [int(token) for token in bridge_row["baseline_candidates"]]
                if bridge_row is not None
                else []
            )
            recursive_bridge = (
                [int(token) for token in bridge_row["cover_candidates"]]
                if bridge_row is not None
                else []
            )
            first_target = int(current["target_token"])
            second_target = int(following["target_token"])
            baseline_root_hit = first_target in baseline_root
            visual_root_hit = first_target in visual_root
            baseline_bridge_hit = second_target in baseline_bridge
            recursive_bridge_hit = second_target in recursive_bridge

            cases.append(
                {
                    "source_path": source_path,
                    "question_id": question_id,
                    "step_index": int(current["step_index"]),
                    "anchor_budget": int(anchor_budget),
                    "bridge_budget": int(bridge_budget),
                    "tree_nodes_without_root": int(
                        anchor_budget + anchor_budget * bridge_budget
                    ),
                    "high_visual_state": bool(current.get("high_visual_state", False)),
                    "grounding_score": float(current.get("grounding_score", 0.0)),
                    "bridge_row_available": bridge_row is not None,
                    "baseline_root_hit": baseline_root_hit,
                    "visual_root_hit": visual_root_hit,
                    "baseline_bridge_hit": baseline_bridge_hit,
                    "recursive_bridge_hit": recursive_bridge_hit,
                    "baseline_path_hit": baseline_root_hit and baseline_bridge_hit,
                    "visual_anchor_language_bridge_path_hit": visual_root_hit
                    and baseline_bridge_hit,
                    "recursive_cover_path_hit": visual_root_hit
                    and recursive_bridge_hit,
                    "root_added_hit": not baseline_root_hit and visual_root_hit,
                    "root_lost_hit": baseline_root_hit and not visual_root_hit,
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
            "visual_anchor_language_bridge_path_coverage": None,
            "path_coverage_gain": None,
            "path_coverage_gain_pp": None,
            "added_paths": 0,
            "lost_paths": 0,
        }
    baseline_hits = sum(int(row["baseline_path_hit"]) for row in cases)
    valb_hits = sum(
        int(row["visual_anchor_language_bridge_path_hit"]) for row in cases
    )
    recursive_hits = sum(int(row["recursive_cover_path_hit"]) for row in cases)
    bridge_available_cases = [row for row in cases if row["bridge_row_available"]]
    bridge_available_count = len(bridge_available_cases)
    available_baseline_hits = sum(
        int(row["baseline_path_hit"]) for row in bridge_available_cases
    )
    available_valb_hits = sum(
        int(row["visual_anchor_language_bridge_path_hit"])
        for row in bridge_available_cases
    )
    root_added_hits = sum(int(row["root_added_hit"]) for row in cases)
    gain = (valb_hits - baseline_hits) / count
    return {
        "num_states": count,
        "bridge_row_available_ratio": sum(
            int(row["bridge_row_available"]) for row in cases
        )
        / count,
        "bridge_available_states": bridge_available_count,
        "baseline_path_coverage_given_bridge_available": (
            available_baseline_hits / bridge_available_count
            if bridge_available_count
            else None
        ),
        "visual_anchor_language_bridge_path_coverage_given_bridge_available": (
            available_valb_hits / bridge_available_count
            if bridge_available_count
            else None
        ),
        "path_coverage_gain_pp_given_bridge_available": (
            100.0
            * (available_valb_hits - available_baseline_hits)
            / bridge_available_count
            if bridge_available_count
            else None
        ),
        "baseline_path_hits": baseline_hits,
        "visual_anchor_language_bridge_path_hits": valb_hits,
        "recursive_cover_path_hits": recursive_hits,
        "baseline_path_coverage": baseline_hits / count,
        "visual_anchor_language_bridge_path_coverage": valb_hits / count,
        "recursive_cover_path_coverage": recursive_hits / count,
        "path_coverage_gain": gain,
        "path_coverage_gain_pp": 100.0 * gain,
        "added_paths": sum(
            int(
                not row["baseline_path_hit"]
                and row["visual_anchor_language_bridge_path_hit"]
            )
            for row in cases
        ),
        "lost_paths": sum(
            int(
                row["baseline_path_hit"]
                and not row["visual_anchor_language_bridge_path_hit"]
            )
            for row in cases
        ),
        "root_added_hits": root_added_hits,
        "root_lost_hits": sum(int(row["root_lost_hit"]) for row in cases),
        "oracle_bridge_path_gain_upper_bound": root_added_hits / count,
        "oracle_bridge_path_gain_upper_bound_pp": 100.0 * root_added_hits / count,
    }


def analyze(
    paths: Sequence[str],
    anchor_budgets: Sequence[int],
    bridge_budgets: Sequence[int],
    primary_anchor_budget: int,
    primary_bridge_budget: int,
    min_gate_states: int,
    min_path_coverage_gain: float,
) -> dict:
    groups = load_record_groups(paths)
    matrix = {}
    all_cases_by_shape = {}
    for anchor_budget in anchor_budgets:
        for bridge_budget in bridge_budgets:
            shape_key = f"anchor{anchor_budget}_bridge{bridge_budget}"
            cases = build_path_cases(groups, anchor_budget, bridge_budget)
            all_cases_by_shape[shape_key] = cases
            matrix[shape_key] = {
                "anchor_budget": anchor_budget,
                "bridge_budget": bridge_budget,
                "tree_nodes_without_root": anchor_budget
                + anchor_budget * bridge_budget,
                "all": summarize_cases(cases),
                "high_visual": summarize_cases(
                    row for row in cases if row["high_visual_state"]
                ),
                "low_visual": summarize_cases(
                    row for row in cases if not row["high_visual_state"]
                ),
            }

    primary_key = f"anchor{primary_anchor_budget}_bridge{primary_bridge_budget}"
    if primary_key not in matrix:
        raise ValueError("primary anchor/bridge budgets must be included in the matrix")
    primary = matrix[primary_key]["high_visual"]
    enough_states = primary["num_states"] >= min_gate_states
    gain = primary["path_coverage_gain"]
    no_lost_paths = primary["lost_paths"] == 0
    gate_pass = bool(
        enough_states
        and gain is not None
        and gain >= min_path_coverage_gain
        and no_lost_paths
    )
    decision = "go" if gate_pass else ("no-go" if enough_states else "inconclusive")

    per_source = {}
    for source_path in sorted({key[0] for key in groups}):
        source_groups = {
            key: rows for key, rows in groups.items() if key[0] == source_path
        }
        source_matrix = {}
        for anchor_budget in anchor_budgets:
            for bridge_budget in bridge_budgets:
                shape_key = f"anchor{anchor_budget}_bridge{bridge_budget}"
                source_cases = build_path_cases(
                    source_groups, anchor_budget, bridge_budget
                )
                source_matrix[shape_key] = {
                    "all": summarize_cases(source_cases),
                    "high_visual": summarize_cases(
                        row for row in source_cases if row["high_visual_state"]
                    ),
                }
        per_source[source_path] = source_matrix

    return {
        "method": "Visual Anchor + Language Bridge path-coverage diagnostic",
        "diagnostic_only": True,
        "input_paths": [str(Path(path).resolve()) for path in paths],
        "num_questions": len(groups),
        "matrix": matrix,
        "per_source": per_source,
        "gate": {
            "decision": decision,
            "pass": gate_pass,
            "primary_shape": primary_key,
            "evaluated_subset": "high_visual",
            "enough_states": enough_states,
            "no_lost_paths": no_lost_paths,
            "thresholds": {
                "min_gate_states": min_gate_states,
                "min_path_coverage_gain": min_path_coverage_gain,
            },
            "observed": primary,
        },
        "methodology_note": (
            "For non-self adjacent transitions, the next record's full-view row "
            "is the same language-bridge row available at the prior state. Self "
            "transitions are excluded to avoid a temporal row-update ambiguity."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Visual Anchor + Language Bridge two-token paths"
    )
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--anchor-budgets", type=parse_positive_ints, default=parse_positive_ints("4,6,8")
    )
    parser.add_argument(
        "--bridge-budgets", type=parse_positive_ints, default=parse_positive_ints("4,6,8")
    )
    parser.add_argument("--primary-anchor-budget", type=int, default=4)
    parser.add_argument("--primary-bridge-budget", type=int, default=4)
    parser.add_argument("--min-gate-states", type=int, default=20)
    parser.add_argument("--min-path-coverage-gain", type=float, default=0.05)
    args = parser.parse_args()

    summary = analyze(
        args.records,
        args.anchor_budgets,
        args.bridge_budgets,
        args.primary_anchor_budget,
        args.primary_bridge_budget,
        args.min_gate_states,
        args.min_path_coverage_gain,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary_path={output_path}")


if __name__ == "__main__":
    main()
