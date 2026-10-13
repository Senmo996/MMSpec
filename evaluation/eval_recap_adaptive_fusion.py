"""Prespecified offline and online controls for adaptive RECAP row fusion."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for directory in (ROOT, SCRIPT_DIR.parent, SCRIPT_DIR):
    sys.path.insert(0, str(directory))

import numpy as np
import torch
from evaluation.eval_recap_suffix_fixes import load_samples
from evaluation.eval_sam_grounded_fixed_multi import build_inputs_for_record, load_processor_for_model
from evaluation.recap_overhead_utils import distribution
from method.sam_grounded.fusion_weighting import FusionRule, ONLINE_RULES, OFFLINE_RULES, concentration, source_weights
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model

POLICY = 'context-score-trigram-fusion-persistent-depth10-node63-wide-plus4'
MODELS = {'qwen': 'Qwen2.5-VL-7B-Instruct', 'llava': 'llava-v1.6-vicuna-7b-hf'}
SOURCE = ROOT / 'outputs/experiments/recap/20260907-163640-recap-path-coverage4x100-direct-gpu2/full'
MANIFEST = ROOT / 'outputs/experiments/recap/20260907-171145-recap-online-confirm4x100-gpu2/manifests'


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def append_json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root, args):
    files = [Path(__file__), ROOT / 'MMSpec/method/sam_grounded/tree_recycling_model.py',
             ROOT / 'MMSpec/method/sam_grounded/fusion_weighting.py',
             ROOT / 'MMSpec/method/sam_grounded/controller.py',
             ROOT / 'MMSpec/method/sam/samd/sam_draft.py', ROOT / 'MMSpec/method/sam/samd/dyn_sam.py',
             ROOT / 'MMSpec/evaluation/eval_recap_suffix_fixes.py',
             ROOT / 'MMSpec/evaluation/eval_sam_grounded_fixed_multi.py',
             ROOT / 'method/fovea/evaluation/eval_llava_fixed_multi.py',
             ROOT / 'docs/experiments/20260911-recap-adaptive-fusion.md']
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = root / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256(source)
    protocol = dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), process_id=os.getpid(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256=hashes, online_rules=ONLINE_RULES, offline_rules=OFFLINE_RULES,
        confidence='exp(-entropy of normalized stored Top-K row)', outcome_tuning=False)
    write_json(root / 'protocol.json', protocol)
    return protocol


def tree(tokens, position, banks, rows, budget, rule=None, depth=10):
    maps = {order: {key: rows[p][0] for key, p in bank.items()} for order, bank in banks.items()}
    scores = {order: {key: rows[p][1] for key, p in bank.items()} for order, bank in banks.items()}
    flat, _, _, paths = Model._build_tree(root_token=tokens[position],
        transitions=torch.empty(0, dtype=torch.long), transition_valid=torch.empty(0, dtype=torch.bool),
        width=8, depth=depth, node_budget=budget, blocked_token_id=tree.blocked,
        host_transitions=maps[1], host_transition_scores=scores[1],
        host_context_transitions=maps[2], host_context_transition_scores=scores[2],
        host_trigram_transitions=maps[3], host_trigram_transition_scores=scores[3],
        root_previous_token=tokens[position-1] if position else None,
        root_previous_previous_token=tokens[position-2] if position >= 2 else None,
        context_candidate_mode='fusion', score_priority_layout=True,
        row_fusion=FusionRule(rule) if rule else None)
    assert len(paths) == len(flat) - 1 <= budget
    return Model._tree_path_token_ids(flat, paths)


def replay(record, modes=OFFLINE_RULES, micro_rows=None):
    tokens = record['input_token_ids']
    prompt = record['prompt_length']
    tree.blocked = record['blocked_token_id']
    rows = {}
    for position, candidates, scores in record['prediction_rows']:
        total = sum(scores)
        assert position not in rows and len(candidates) == len(set(candidates)) == len(scores)
        assert total > 0 and all(math.isfinite(x) and x >= 0 for x in scores)
        rows[position] = (candidates, [s / total for s in scores])
    banks = {order: {} for order in (1, 2, 3)}
    prompt_positions, _ = Model._select_prompt_transition_rows(tokens[:prompt])
    for p in prompt_positions:
        if tokens[p] != tree.blocked:
            banks[1][tokens[p]] = p
    result = []
    for p in range(prompt, len(tokens) - 1):
        donors = []
        for order in (3, 2, 1):
            key = tokens[p] if order == 1 else tuple(tokens[p+1-order:p+1])
            if key in banks[order]:
                donors.append(banks[order][key])
        assert all(s < p for s in donors)
        retrieved = [rows[s] for s in donors]
        if micro_rows is not None and len(retrieved) > 1 and len(micro_rows) < 512:
            micro_rows.append(retrieved)
        capped = {mode: tree(tokens, p, banks, rows, 63, mode) for mode in modes}
        assert capped['fixed'] == tree(tokens, p, banks, rows, 63), 'fixed hook differs from production'
        common = min(map(len, capped.values()))
        future = tokens[p+1:p+11]
        label = rows[p][0][0]
        phase = '1-32' if p - prompt < 32 else ('33-64' if p - prompt < 64 else '65+')
        entropy = -math.log(concentration(retrieved[0][1])) if retrieved else None
        state = dict(position=p, phase=phase, num_sources=len(retrieved), source_positions=donors,
            source_disagreement=len({r[0][0] for r in retrieved}) > 1,
            primary_high_entropy=entropy is not None and entropy > .5 * math.log(8),
            matched_node_budget=common, methods={})
        for mode in modes:
            candidates, _ = FusionRule(mode)(retrieved, 8)
            candidates = [token for token in candidates if token != tree.blocked]
            if mode == 'fixed':
                assert candidates == [path[0] for path in tree(tokens, p, banks, rows, 8, depth=1)]
            equal = capped[mode] if len(capped[mode]) == common else tree(tokens, p, banks, rows, common, mode)
            assert len(equal) == common
            state['methods'][mode] = dict(candidates=candidates, available=bool(candidates),
                hit=label in candidates, native_hit=future[0] in candidates,
                nodes_at_cap=len(capped[mode]),
                matched_continuation=Model._longest_matching_path(equal, future),
                weights=source_weights(retrieved, mode))
        result.append(state)
        for order in (1, 2, 3):
            if p + 1 >= order and tokens[p] != tree.blocked:
                key = tokens[p] if order == 1 else tuple(tokens[p+1-order:p+1])
                banks[order][key] = p
    return result


def summarize_states(states, mode):
    values = [s['methods'][mode] for s in states]
    covered = sum(v['available'] for v in values)
    return dict(num_states=len(states), coverage=covered / len(states),
        recall=sum(v['hit'] for v in values) / len(states),
        conditional_recall=sum(v['hit'] for v in values) / covered if covered else None,
        native_recall=statistics.fmean(v['native_hit'] for v in values),
        mean_actual_candidates=statistics.fmean(len(v['candidates']) for v in values),
        matched_continuation=statistics.fmean(v['matched_continuation'] for v in values),
        mean_nodes_at_cap=statistics.fmean(v['nodes_at_cap'] for v in values))


def offline_summary(records, modes):
    summary = {}
    for dataset in sorted({r['dataset'] for r in records}):
        samples = [r for r in records if r['dataset'] == dataset]
        states = [s for r in samples for s in r['states']]
        groups = {'all': states, 'multiple_sources': [s for s in states if s['num_sources'] > 1],
                  'source_disagreement': [s for s in states if s['source_disagreement']],
                  'primary_high_entropy': [s for s in states if s['primary_high_entropy']],
                  **{f'phase_{p}': [s for s in states if s['phase'] == p] for p in ('1-32', '33-64', '65+')}}
        comparison = {}
        for mode in modes:
            cluster = defaultdict(lambda: np.zeros(3))
            for sample in samples:
                for state in sample['states']:
                    a, b = state['methods'][mode], state['methods']['fixed']
                    cluster[sample['image_cluster']] += [int(a['hit']) - int(b['hit']),
                        a['matched_continuation'] - b['matched_continuation'], 1]
            array = np.array(list(cluster.values()))
            rng = np.random.default_rng(20260911)
            estimates = []
            for _ in range(2000):
                total = array[rng.integers(0, len(array), len(array))].sum(axis=0)
                estimates.append(total[:2] / total[2])
            estimates = np.array(estimates)
            total = array.sum(axis=0)
            comparison[mode] = dict(recall_delta=float(total[0] / total[2]),
                recall_delta_ci95=np.quantile(estimates[:, 0], [.025, .975]).tolist(),
                matched_continuation_delta=float(total[1] / total[2]),
                matched_continuation_delta_ci95=np.quantile(estimates[:, 1], [.025, .975]).tolist(),
                image_clusters=len(array), ci_unit='paired image-cluster bootstrap')
        summary[dataset] = dict(unique_inputs=len(samples), num_states=len(states),
            strata={g: {m: summarize_states(s, m) for m in modes} for g, s in groups.items() if s},
            paired_vs_fixed=comparison)
    return summary


def run_offline(args, root):
    source = args.source_root / 'results.jsonl'
    original_summary = json.loads((args.source_root / 'summary.json').read_text())
    assert original_summary['complete']
    counts = defaultdict(int)
    records, micro_rows = [], []
    write_json(root / 'source.json', dict(path=str(source), sha256=sha256(source)))
    with source.open() as stream:
        for line in stream:
            saved = json.loads(line)
            if counts[saved['dataset']] >= args.samples_per_dataset:
                continue
            counts[saved['dataset']] += 1
            cluster = json.dumps(saved['target_image_identity'], sort_keys=True)
            states = replay(saved, micro_rows=micro_rows)
            row = dict(dataset=saved['dataset'], question_id=saved['question_id'],
                       image_cluster=cluster, states=states)
            append_json(root / 'results.jsonl', row)
            records.append(row)
            print(f"stage=offline dataset={saved['dataset']} sample={counts[saved['dataset']]} states={len(states)}", flush=True)
    assert len(counts) == 4 and all(n == args.samples_per_dataset for n in counts.values())
    timing = defaultdict(list)
    for repeat in range(7):
        modes = list(OFFLINE_RULES)
        random.Random(repeat).shuffle(modes)
        for mode in modes:
            rule = FusionRule(mode)
            start = time.perf_counter()
            for _ in range(20):
                for rows in micro_rows:
                    rule(rows, 8)
            timing[mode].append((time.perf_counter() - start) * 1e6 / (20 * len(micro_rows)))
    write_json(root / 'summary.json', dict(complete=True, num_records=len(records),
        num_states=sum(len(r['states']) for r in records), jsonl_path=str(root / 'results.jsonl'),
        datasets=offline_summary(records, OFFLINE_RULES),
        fusion_us_per_call={k: distribution(v) for k, v in timing.items()},
        fusion_timing_scope=f'isolated host fusion on {len(micro_rows)} multiple-source queries, 7 repeats; not decoding latency',
        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def online_options(namespace, rule):
    return dict(draft_policy=POLICY, matrix_top_k=8, tree_fixed_width=8, tree_fixed_depth=10,
        tree_broad_width=4, tree_shallow_depth=3, tree_node_budget=63,
        recycling_prompt_init='unigram', recycling_max_order=3, recycling_suffix_length=4,
        recycling_suffix_implementation='fixed', recycling_suffix_bridge=False,
        recycling_sam_clone_mode='flat', recycling_cache_namespace=namespace,
        recycling_fusion_rule=rule)


def summarize_online(records, root, expected):
    assert len(records) == expected * len(ONLINE_RULES)
    all_summary = {}
    for mode in ONLINE_RULES:
        rows = [r for r in records if r['rule'] == mode]
        accept = distribution(r['average_accept_length'] for r in rows)
        speed = distribution(r['speedup'] for r in rows)
        summary = dict(num_records=len(rows), unique_inputs=len({r['sample_id'] for r in rows}),
            avg_accept_length=accept['mean'], median_accept_length=accept['median'],
            max_accept_length=accept['maximum'], p90_accept_length=accept['p90'],
            avg_tokens_per_iteration=statistics.fmean(r['average_tokens_per_iteration'] for r in rows),
            avg_sample_speedup=speed['mean'], median_sample_speedup=speed['median'],
            max_sample_speedup=speed['maximum'], sample_speedup_gt_1_ratio=statistics.fmean(r['speedup'] > 1 for r in rows),
            accept_le_0_5_ratio=statistics.fmean(r['average_accept_length'] <= .5 for r in rows),
            speedup_reference='paired fixed-prior RECAP, not autoregressive target',
            jsonl_path=str(root / mode / 'results.jsonl'),
            by_dataset={})
        for dataset in sorted({r['dataset'] for r in rows}):
            group = [r for r in rows if r['dataset'] == dataset]
            counters = [r['fusion_counters'] for r in group]
            summary['by_dataset'][dataset] = dict(
                num_records=len(group), unique_inputs=len({r['sample_id'] for r in group}),
                mean_speed_ratio=statistics.fmean(r['speedup'] for r in group),
                mean_accept_length=statistics.fmean(r['average_accept_length'] for r in group),
                mean_tokens_per_iteration=statistics.fmean(r['average_tokens_per_iteration'] for r in group),
                mean_nodes_per_iteration=sum(r['counters']['total_candidate_nodes'] for r in group) /
                    max(sum(r['counters']['iterations'] for r in group), 1),
                multiple_source_calls=sum(c.get('multiple_source_calls', 0) for c in counters),
                single_source_calls=sum(c.get('single_source_calls', 0) for c in counters),
                gate_selected_rank_counts=[sum(c.get('gate_selected_rank_counts', [0]*4)[i] for c in counters) for i in range(4)],
                by_seed={str(seed): dict(mean_speed_ratio=statistics.fmean(r['speedup'] for r in group if r['repeat_seed'] == seed),
                    mean_accept_length=statistics.fmean(r['average_accept_length'] for r in group if r['repeat_seed'] == seed))
                    for seed in sorted({r['repeat_seed'] for r in group})})
        write_json(root / mode / 'summary.json', summary)
        all_summary[mode] = summary
    return dict(complete=True, num_records=len(records), rules=all_summary,
                finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def run_online(args, root):
    samples, manifests = load_samples(args.manifest_dir, args.samples_per_dataset)
    write_json(root / 'manifests.json', {k: dict(path=str(p), sha256=sha256(Path(p))) for k, p in manifests.items()})
    path = str(ROOT / 'models' / MODELS[args.model])
    print(f'stage=load_model model={args.model} unique_inputs={len(samples)}', flush=True)
    model = Model.from_pretrained(base_model_path=path, spec_model_path='', total_token=160,
        torch_dtype=torch.bfloat16 if args.model == 'qwen' else torch.float16,
        low_cpu_mem_usage=True, device_map='cuda:0', attn_implementation='sdpa')
    model.processor = load_processor_for_model(path)
    model.eval()
    for sample in {s['dataset']: s for s in reversed(samples)}.values():
        inputs = build_inputs_for_record(model, sample['dataset'], sample['row'])
        for mode in ONLINE_RULES:
            model.specgenerate(**inputs, temperature=0, max_new_tokens=24,
                               **online_options('warmup', FusionRule(mode)))
    model.reset_persistent_recycling_cache()
    print('stage=warmup_complete', flush=True)
    seeds = [int(s) for s in args.repeat_seeds.split(',')]
    records = []
    for seed in seeds:
        model.reset_persistent_recycling_cache()
        ordered = list(samples)
        random.Random(seed).shuffle(ordered)
        write_json(root / f'stream_seed{seed}.json', [{k: s[k] for k in ('id', 'dataset', 'source_index', 'image_cluster_id')} for s in ordered])
        for position, sample in enumerate(ordered):
            inputs = build_inputs_for_record(model, sample['dataset'], sample['row'])
            offset = (position + seeds.index(seed)) % len(ONLINE_RULES)
            modes = ONLINE_RULES[offset:] + ONLINE_RULES[:offset]
            paired = {}
            for mode in modes:
                counters, fusion_counters = {}, {}
                torch.manual_seed(0)
                torch.cuda.synchronize()
                start = time.perf_counter()
                output, n_new, iterations, accepted = model.specgenerate(
                    **inputs, temperature=0, max_new_tokens=args.max_new_tokens,
                    log=True, return_acceptance_len=True, return_policy_trace=False,
                    recycling_stats=counters,
                    **online_options(f'{seed}:{mode}', FusionRule(mode, fusion_counters)))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                tokens = output[0, inputs.input_ids.shape[1]:].cpu().tolist()
                assert len(tokens) == n_new and 0 < n_new <= args.max_new_tokens
                assert counters.get('max_candidate_nodes', 0) <= 67
                row = dict(sample_id=sample['id'], source_index=sample['source_index'],
                    dataset=sample['dataset'], image_cluster_id=sample['image_cluster_id'],
                    model=args.model, repeat_seed=seed, stream_position=position, rule=mode,
                    new_tokens=n_new, iterations=iterations, accepted=accepted, wall_seconds=elapsed,
                    average_accept_length=statistics.fmean(accepted) if accepted else 0.,
                    average_tokens_per_iteration=(n_new - 1) / len(accepted) if accepted else 0.,
                    output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    counters=counters, fusion_counters=fusion_counters)
                paired[mode] = row
                append_json(root / 'journal.jsonl', row)
            fixed = paired['fixed']
            for mode, row in paired.items():
                row['speedup'] = (row['new_tokens'] / row['wall_seconds']) / (fixed['new_tokens'] / fixed['wall_seconds'])
                row['output_matches_fixed'] = row['output_sha256'] == fixed['output_sha256']
                append_json(root / mode / 'results.jsonl', row)
                records.append(row)
            print(f"model={args.model} seed={seed} request={position+1}/{len(ordered)} dataset={sample['dataset']} fixed_seconds={fixed['wall_seconds']:.3f} complete_arms={len(paired)}", flush=True)
    summary = summarize_online(records, root, len(samples) * len(seeds))
    summary.update(model=args.model, persistent_streams=len(seeds),
        uncertainty_note='descriptive repeated streams, not independent repeated inputs',
        pair_output_match_ratio=statistics.fmean(r['output_matches_fixed'] for r in records))
    write_json(root / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--experiment', choices=('offline', 'online'), required=True)
    parser.add_argument('--model', choices=MODELS, default='qwen')
    parser.add_argument('--samples-per-dataset', type=int, default=50)
    parser.add_argument('--repeat-seeds', default='17,29')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--manifest-dir', type=Path, default=MANIFEST)
    args = parser.parse_args()
    if not 1 <= args.samples_per_dataset <= 100:
        parser.error('samples-per-dataset must be 1..100')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    snapshot(root, args)
    if args.experiment == 'offline':
        run_offline(args, root)
    else:
        run_online(args, root)
    print(f'stage=finished output_root={root}', flush=True)


if __name__ == '__main__':
    main()
