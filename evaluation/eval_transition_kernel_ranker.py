"""Evaluate a training-free prompt-local transition-kernel ranker.

This is a falsification-first diagnostic.  For an unseen root token, the
ranker retrieves prompt/history transitions by source-token and source-context
similarity, then reranks only a prompt-native visual candidate pool.  All
ranker inputs exist before the target-model reference forward for the root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.eval_sam_grounded_mmspec import _select_topic_indices
from evaluation.eval_visual_lexical_inventory import (
    _question_lexical_token_ids,
    _run_reference_forward,
    _seed_transition_rows,
)
from evaluation.utils import build_prompt, load_mmspec_data
from method.sam_grounded.controller import VisualGroundingCalibrator
from method.sam_grounded.counterfactual_probes import build_visual_probe_layout
from method.sam_grounded.spec_model import SpecModel
from method.sam_grounded.transition_kernel import (
    TransitionKernelBank,
    TransitionKernelConfig,
    build_config_grid,
)
from method.sam_grounded.visual_lexical_inventory import (
    build_visual_lexical_inventories,
)
from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids


STATIC_CONFIG_KEY = "static_visual"


def _parse_positive_ints(value: str) -> List[int]:
    result = []
    for item in value.split(","):
        parsed = int(item.strip())
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _parse_floats(value: str) -> List[float]:
    result = []
    for item in value.split(","):
        parsed = float(item.strip())
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def _percentile(values: Iterable[float], quantile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> Dict[str, Optional[float]]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _target_ranks_from_grid(
    score_grid: Dict[str, torch.Tensor],
    candidate_token_ids: torch.Tensor,
    target_token: int,
) -> Dict[str, Optional[int]]:
    keys = list(score_grid)
    matches = torch.nonzero(
        candidate_token_ids.reshape(-1) == int(target_token), as_tuple=False
    )
    if matches.numel() == 0:
        return {key: None for key in keys}
    target_index = int(matches[0, 0].item())
    scores = torch.stack([score_grid[key].reshape(-1) for key in keys], dim=0)
    target_scores = scores[:, target_index : target_index + 1]
    indices = torch.arange(scores.shape[1], device=scores.device)[None, :]
    ranks = 1 + (
        (scores > target_scores)
        | ((scores == target_scores) & (indices < target_index))
    ).sum(dim=1)
    return {
        key: int(rank)
        for key, rank in zip(keys, ranks.detach().to("cpu").tolist())
    }


def _coverage_metrics(
    records: Sequence[dict], pool_name: str, config_key: str
) -> dict:
    rows = [record for record in records if pool_name in record.get("pools", {})]
    ranks = [row["pools"][pool_name]["target_ranks"].get(config_key) for row in rows]
    count = len(rows)
    pool_hits = sum(rank is not None for rank in ranks)
    result = {
        "num_states": count,
        "pool_hits": pool_hits,
        "pool_recall": pool_hits / count if count else None,
        "mrr": (
            statistics.fmean(0.0 if rank is None else 1.0 / rank for rank in ranks)
            if count
            else None
        ),
    }
    for width in (1, 2, 3, 4, 8):
        hits = sum(rank is not None and rank <= width for rank in ranks)
        result[f"top{width}_hits"] = hits
        result[f"top{width}_hit_rate"] = hits / count if count else None
        result[f"top{width}_given_pool_rate"] = (
            hits / pool_hits if pool_hits else None
        )
    return result


def _path_metrics(
    records: Sequence[dict], pool_name: str, config_key: str, width: int
) -> dict:
    by_state = {
        (int(record["sample_index"]), int(record["step_index"])): record
        for record in records
    }
    eligible = []
    for record in records:
        if not record.get("high_visual_state", False):
            continue
        if record.get("baseline_row_available", False):
            continue
        if pool_name not in record.get("pools", {}):
            continue
        next_record = by_state.get(
            (int(record["sample_index"]), int(record["step_index"]) + 1)
        )
        if next_record is None:
            continue
        eligible.append((record, next_record))

    first_hits = 0
    bridge_available = 0
    path_hits = 0
    for record, next_record in eligible:
        rank = record["pools"][pool_name]["target_ranks"].get(config_key)
        first_hit = rank is not None and int(rank) <= int(width)
        bridge = record.get("bridge_candidates_before") or []
        second_hit = int(next_record["target_token"]) in {
            int(token) for token in bridge[: int(width)]
        }
        first_hits += int(first_hit)
        bridge_available += int(bool(bridge))
        path_hits += int(first_hit and second_hit)

    count = len(eligible)
    return {
        "num_states_with_next": count,
        "width": int(width),
        "first_hits": first_hits,
        "first_hit_rate": first_hits / count if count else None,
        "bridge_row_available": bridge_available,
        "bridge_row_available_rate": bridge_available / count if count else None,
        "path_hits": path_hits,
        "path_hit_rate": path_hits / count if count else None,
        "bridge_conversion_rate": path_hits / first_hits if first_hits else None,
    }


def _best_config(metrics: Dict[str, dict], path_metrics: Dict[str, dict]) -> str:
    candidates = [key for key in metrics if key != STATIC_CONFIG_KEY]
    if not candidates:
        return STATIC_CONFIG_KEY

    def value(key: str):
        row = metrics[key]
        path = path_metrics.get(key, {})
        return (
            row.get("top3_hit_rate") or 0.0,
            path.get("path_hit_rate") or 0.0,
            row.get("top1_hit_rate") or 0.0,
            row.get("mrr") or 0.0,
            key,
        )

    return max(candidates, key=value)


def summarize_records(
    records: Sequence[dict],
    inventory_records: Sequence[dict],
    configs: Sequence[TransitionKernelConfig],
    args,
) -> dict:
    config_keys = [STATIC_CONFIG_KEY] + [config.key for config in configs]
    subsets = {
        "all_row_absent": [
            row for row in records if not row.get("baseline_row_available", False)
        ],
        "high_visual_row_absent": [
            row
            for row in records
            if row.get("high_visual_state", False)
            and not row.get("baseline_row_available", False)
        ],
        "low_visual_row_absent": [
            row
            for row in records
            if not row.get("high_visual_state", False)
            and not row.get("baseline_row_available", False)
        ],
    }
    pool_names = ("observed_visual", "mismatched_visual", "text_control")
    metrics = {
        subset_name: {
            pool_name: {
                config_key: _coverage_metrics(subset, pool_name, config_key)
                for config_key in config_keys
            }
            for pool_name in pool_names
        }
        for subset_name, subset in subsets.items()
    }
    path_metrics = {
        pool_name: {
            f"width{width}": {
                config_key: _path_metrics(
                    records, pool_name, config_key, width
                )
                for config_key in config_keys
            }
            for width in (1, 2, 3, 4)
        }
        for pool_name in pool_names
    }
    primary_metrics = metrics["high_visual_row_absent"]["observed_visual"]
    primary_paths = path_metrics["observed_visual"]["width3"]
    selected = _best_config(primary_metrics, primary_paths)
    static = primary_metrics[STATIC_CONFIG_KEY]
    selected_metrics = primary_metrics[selected]
    selected_paths = primary_paths[selected]
    top3_gain_pp = (
        100.0
        * (
            (selected_metrics.get("top3_hit_rate") or 0.0)
            - (static.get("top3_hit_rate") or 0.0)
        )
        if selected_metrics.get("num_states")
        else None
    )
    enough_states = selected_metrics["num_states"] >= args.min_gate_states
    top3_pass = bool(
        enough_states
        and selected_metrics.get("top3_hit_rate") is not None
        and selected_metrics["top3_hit_rate"] >= args.min_top3_hit_rate
    )
    path_pass = bool(
        selected_paths.get("path_hit_rate") is not None
        and selected_paths["path_hit_rate"] >= args.min_path_hit_rate
    )

    return {
        "method": "Prompt-local Cross-Token Transition Kernel kill test",
        "diagnostic_only": True,
        "num_samples": len(inventory_records),
        "num_states": len(records),
        "configuration": {
            "topic_offset": args.topic_offset,
            "max_new_token": args.max_new_token,
            "candidate_pool_size": args.candidate_pool_size,
            "visual_threshold": args.visual_threshold,
            "candidate_row_budget": args.candidate_row_budget,
            "token_weights": args.token_weights,
            "neighbors": args.neighbors,
            "temperatures": args.temperatures,
            "visual_weights": args.visual_weights,
            "num_kernel_configs": len(configs),
            "default_latency_config": args.default_latency_config,
            "attention_implementation": args.attn_implementation,
        },
        "kernel_configs": [
            {
                "key": config.key,
                "token_weight": config.token_weight,
                "neighbors": config.neighbors,
                "temperature": config.temperature,
                "visual_weight": config.visual_weight,
            }
            for config in configs
        ],
        "subsets": {name: len(rows) for name, rows in subsets.items()},
        "metrics": metrics,
        "path_metrics": path_metrics,
        "latency_ms": {
            "default_single_config_observed_pool": _distribution(
                row["ranker_latency_ms"]
                for row in records
                if row.get("ranker_latency_ms") is not None
            ),
            "inventory_construction": _distribution(
                row["inventory_construction_ms"] for row in inventory_records
            ),
        },
        "within_split_selection": {
            "selected_config": selected,
            "selection_metric": "top3 hit; tie-break path@3, top1, MRR",
            "selected_metrics": selected_metrics,
            "static_metrics": static,
            "top3_gain_pp": top3_gain_pp,
            "selected_path_width3": selected_paths,
            "warning": (
                "This is valid for offset-0 exploration only. Freeze this config "
                "before interpreting offset-1 and offset-2."
            ),
        },
        "local_gate": {
            "decision": "go" if top3_pass and path_pass else "no-go",
            "top3_pass": top3_pass,
            "path_pass": path_pass,
            "enough_states": enough_states,
            "thresholds": {
                "min_gate_states": args.min_gate_states,
                "min_top3_hit_rate": args.min_top3_hit_rate,
                "min_path_hit_rate": args.min_path_hit_rate,
            },
        },
    }


def _make_bank(
    *,
    candidates: Sequence[int],
    prompt_token_ids: torch.Tensor,
    prompt_token_embeddings: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    prompt_logits: torch.Tensor,
    visual_mask: torch.Tensor,
    excluded_token_ids: Sequence[int],
) -> TransitionKernelBank:
    candidate_ids = torch.tensor(
        [int(token) for token in candidates],
        dtype=torch.long,
        device=prompt_logits.device,
    )
    return TransitionKernelBank.from_prompt(
        candidate_token_ids=candidate_ids,
        prompt_token_ids=prompt_token_ids,
        prompt_token_embeddings=prompt_token_embeddings,
        prompt_hidden_states=prompt_hidden_states,
        prompt_logits=prompt_logits,
        visual_mask=visual_mask,
        excluded_token_ids=excluded_token_ids,
    )


@torch.inference_mode()
def evaluate(args) -> None:
    configs = build_config_grid(
        args.token_weights,
        args.neighbors,
        args.temperatures,
        args.visual_weights,
    )
    config_by_key = {config.key: config for config in configs}
    if args.default_latency_config not in config_by_key:
        raise ValueError(
            "--default-latency-config must be present in the configured grid"
        )

    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path="",
        total_token=args.candidate_row_budget,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    base_model = model.base_model
    embedding_layer = base_model.get_input_embeddings()
    stop_token_ids = _collect_stop_token_ids(model.tokenizer, base_model)
    past_key_values, _past_data, current_length_data = initialize_past_key_values(
        base_model
    )

    data = load_mmspec_data(args.data_folder)
    if args.samples_per_topic is not None:
        data = data.select(
            _select_topic_indices(data, args.samples_per_topic, args.topic_offset)
        )
    if args.max_samples is not None:
        data = data.select(range(min(args.max_samples, len(data))))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "ranker_records.jsonl"
    inventory_path = output_root / "inventory_records.jsonl"
    records: List[dict] = []
    inventory_records: List[dict] = []
    excluded_token_ids = set(int(token) for token in model.tokenizer.all_special_ids)
    for attribute in ("image_token_id", "video_token_id"):
        token_id = getattr(base_model.config, attribute, None)
        if token_id is not None:
            excluded_token_ids.add(int(token_id))

    print(f"Loaded {len(data)} samples")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"topic_offset={args.topic_offset}")
    print(f"num_kernel_configs={len(configs)}")
    print(f"records_path={records_path}")
    print(f"inventory_path={inventory_path}")

    previous_visual_pool: Optional[List[int]] = None
    previous_question_id: Optional[str] = None
    with records_path.open("w", encoding="utf-8") as records_file, inventory_path.open(
        "w", encoding="utf-8"
    ) as inventory_file:
        for sample_index, sample in enumerate(
            tqdm(data, desc=f"Transition kernel offset {args.topic_offset}")
        ):
            current_length_data.zero_()
            model_inputs = build_prompt(sample, args)
            input_ids = model_inputs["input_ids"]
            prompt_length = int(input_ids.shape[1])
            visual_mask = model._build_visual_token_mask(input_ids).detach()
            layout = build_visual_probe_layout(
                visual_mask,
                model_inputs.get("image_grid_thw"),
                int(base_model.config.vision_config.spatial_merge_size),
                args.inventory_regions,
            )
            init_output = base_model(
                **dict(model_inputs),
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
            )
            observed_lengths = current_length_data.unique().tolist()
            if observed_lengths != [prompt_length]:
                raise RuntimeError(
                    f"prefill KV lengths {observed_lengths} do not match prompt {prompt_length}"
                )
            current_length_data.fill_(prompt_length)
            prompt_hidden = init_output.hidden_states[-1][0]
            prompt_logits = init_output.logits[0]
            calibrator = VisualGroundingCalibrator.from_prompt(
                prompt_hidden, visual_mask
            )
            question = sample.get("turns", [sample.get("prompt", "")])[0]
            question_token_ids = _question_lexical_token_ids(
                model.tokenizer, question
            )

            device = prompt_logits.device
            _synchronize_if_cuda(device)
            inventory_started = time.perf_counter()
            inventories = build_visual_lexical_inventories(
                prompt_logits,
                visual_mask,
                layout,
                question_token_ids,
                args.candidate_pool_size,
                excluded_token_ids=sorted(excluded_token_ids),
                valid_vocab_size=min(
                    len(model.tokenizer), int(prompt_logits.shape[-1])
                ),
            )
            _synchronize_if_cuda(device)
            inventory_latency = 1000.0 * (time.perf_counter() - inventory_started)

            observed_pool = [
                int(token)
                for token in inventories["visual_max"][: args.candidate_pool_size]
            ]
            text_pool = [
                int(token)
                for token in inventories["text_mean_control"][
                    : args.candidate_pool_size
                ]
            ]
            pool_candidates: Dict[str, List[int]] = {
                "observed_visual": observed_pool,
                "text_control": text_pool,
            }
            if previous_visual_pool is not None:
                pool_candidates["mismatched_visual"] = previous_visual_pool

            with torch.no_grad():
                prompt_token_embeddings = embedding_layer(input_ids[0])
            banks = {
                pool_name: _make_bank(
                    candidates=candidates,
                    prompt_token_ids=input_ids[0],
                    prompt_token_embeddings=prompt_token_embeddings,
                    prompt_hidden_states=prompt_hidden,
                    prompt_logits=prompt_logits,
                    visual_mask=visual_mask,
                    excluded_token_ids=sorted(excluded_token_ids),
                )
                for pool_name, candidates in pool_candidates.items()
            }

            inventory_record = {
                "question_id": sample["id"],
                "topic": sample.get("topic", "unknown"),
                "category": sample.get("category", "unknown"),
                "sample_index": sample_index,
                "topic_offset": args.topic_offset,
                "prompt_length": prompt_length,
                "num_visual_tokens": int(visual_mask.sum().item()),
                "inventory_construction_ms": inventory_latency,
                "observed_visual_pool": observed_pool,
                "text_control_pool": text_pool,
                "mismatched_visual_pool": previous_visual_pool,
                "mismatched_question_id": previous_question_id,
                "prompt_transition_count": banks["observed_visual"].source_count,
            }
            inventory_records.append(inventory_record)
            inventory_file.write(json.dumps(inventory_record) + "\n")
            inventory_file.flush()

            transition_rows = _seed_transition_rows(
                input_ids[0], prompt_logits, args.candidate_row_budget
            )
            root_token = int(prompt_logits[-1].argmax().item())
            parent_hidden = prompt_hidden[-1]
            if root_token in stop_token_ids:
                previous_visual_pool = observed_pool
                previous_question_id = str(sample["id"])
                continue

            for step_index in range(max(args.max_new_token - 1, 0)):
                prefix_length = int(current_length_data[0].item())
                baseline_previous = transition_rows.get(root_token)
                query_token_id = torch.tensor(
                    [root_token], dtype=torch.long, device=input_ids.device
                )
                query_token_vector = embedding_layer(query_token_id)[0]

                score_grids: Dict[str, Dict[str, torch.Tensor]] = {}
                ranker_latency_ms = None
                if baseline_previous is None:
                    observed_bank = banks["observed_visual"]
                    _synchronize_if_cuda(device)
                    latency_started = time.perf_counter()
                    observed_bank.score(
                        query_token_vector=query_token_vector,
                        query_context_vector=parent_hidden,
                        config=config_by_key[args.default_latency_config],
                    )
                    _synchronize_if_cuda(device)
                    ranker_latency_ms = 1000.0 * (
                        time.perf_counter() - latency_started
                    )
                    for pool_name, bank in banks.items():
                        score_grids[pool_name] = bank.score_grid(
                            query_token_vector=query_token_vector,
                            query_context_vector=parent_hidden,
                            configs=configs,
                        )

                reference_output = _run_reference_forward(
                    base_model,
                    root_token,
                    prefix_length,
                    past_key_values,
                    current_length_data,
                )
                reference_logits = reference_output.logits[0, 0]
                target_token = int(reference_logits.argmax().item())
                grounding_score = calibrator.score(
                    reference_output.hidden_states[-1][0, 0]
                )
                bridge_previous = transition_rows.get(target_token)

                pools_payload = {}
                for pool_name, grid in score_grids.items():
                    bank = banks[pool_name]
                    ranks = _target_ranks_from_grid(
                        grid, bank.candidate_token_ids, target_token
                    )
                    candidates = pool_candidates[pool_name]
                    try:
                        static_rank = candidates.index(target_token) + 1
                    except ValueError:
                        static_rank = None
                    ranks = {STATIC_CONFIG_KEY: static_rank, **ranks}
                    pools_payload[pool_name] = {
                        "candidate_count": len(candidates),
                        "target_in_pool": static_rank is not None,
                        "static_rank": static_rank,
                        "target_ranks": ranks,
                    }

                record = {
                    "question_id": sample["id"],
                    "topic": sample.get("topic", "unknown"),
                    "category": sample.get("category", "unknown"),
                    "sample_index": sample_index,
                    "topic_offset": args.topic_offset,
                    "step_index": step_index,
                    "prompt_length": prompt_length,
                    "prefix_kv_length": prefix_length,
                    "root_token": root_token,
                    "target_token": target_token,
                    "grounding_score": float(grounding_score),
                    "high_visual_state": grounding_score >= args.visual_threshold,
                    "baseline_row_available": baseline_previous is not None,
                    "baseline_candidates": (
                        list(baseline_previous) if baseline_previous is not None else []
                    ),
                    "bridge_candidates_before": (
                        list(bridge_previous) if bridge_previous is not None else []
                    ),
                    "ranker_latency_ms": ranker_latency_ms,
                    "pools": pools_payload,
                }
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
                records_file.flush()

                for bank in banks.values():
                    candidate_logits = reference_logits.index_select(
                        0, bank.candidate_token_ids
                    )
                    bank.append(
                        source_token_vector=query_token_vector,
                        source_context_vector=parent_hidden,
                        candidate_logits=candidate_logits,
                    )
                transition_rows[root_token] = [
                    int(token)
                    for token in reference_logits.topk(
                        args.candidate_row_budget
                    ).indices.tolist()
                ]
                parent_hidden = reference_output.hidden_states[-1][0, 0]
                root_token = target_token
                if target_token in stop_token_ids:
                    break

            previous_visual_pool = observed_pool
            previous_question_id = str(sample["id"])

    summary = summarize_records(records, inventory_records, configs, args)
    summary["records_path"] = str(records_path.resolve())
    summary["inventory_records_path"] = str(inventory_path.resolve())
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")
    print(
        json.dumps(
            {
                "method": summary["method"],
                "num_samples": summary["num_samples"],
                "num_states": summary["num_states"],
                "subsets": summary["subsets"],
                "latency_ms": summary["latency_ms"],
                "within_split_selection": summary["within_split_selection"],
                "local_gate": summary["local_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"summary_path={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt-local transition-kernel ranker kill test"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--samples-per-topic", type=int, default=1)
    parser.add_argument("--topic-offset", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=64)
    parser.add_argument("--inventory-regions", type=int, default=4)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--candidate-row-budget", type=int, default=4)
    parser.add_argument("--visual-threshold", type=float, default=0.55)
    parser.add_argument(
        "--token-weights", type=_parse_floats, default=_parse_floats("0,0.5,1")
    )
    parser.add_argument(
        "--neighbors", type=_parse_positive_ints, default=_parse_positive_ints("4,16,64")
    )
    parser.add_argument(
        "--temperatures", type=_parse_floats, default=_parse_floats("0.05,0.15")
    )
    parser.add_argument(
        "--visual-weights", type=_parse_floats, default=_parse_floats("0,0.25,0.5,0.75")
    )
    parser.add_argument(
        "--default-latency-config",
        default="tw0.50-k16-t0.15-vw0.50",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-gate-states", type=int, default=20)
    parser.add_argument("--min-top3-hit-rate", type=float, default=0.15)
    parser.add_argument("--min-path-hit-rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.samples_per_topic is not None and args.samples_per_topic <= 0:
        parser.error("--samples-per-topic must be positive")
    if args.topic_offset < 0:
        parser.error("--topic-offset must be non-negative")
    if args.max_new_token <= 1:
        parser.error("--max-new-token must be greater than one")
    if args.candidate_pool_size <= 0:
        parser.error("--candidate-pool-size must be positive")
    if args.candidate_row_budget <= 0:
        parser.error("--candidate-row-budget must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.token_weights):
        parser.error("--token-weights values must be in [0, 1]")
    if any(value <= 0.0 for value in args.temperatures):
        parser.error("--temperatures values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.visual_weights):
        parser.error("--visual-weights values must be in [0, 1]")
    args.model = args.base_model_path
    evaluate(args)


if __name__ == "__main__":
    main()
