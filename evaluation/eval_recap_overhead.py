"""Frozen, interleaved RECAP latency decomposition and candidate-budget study."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for directory in (SCRIPT_DIR, SCRIPT_DIR.parent, PROJECT_ROOT):
    sys.path.insert(0, str(directory))

import torch
from evaluation.eval_recap_suffix_fixes import load_samples
from evaluation.eval_sam_grounded_fixed_multi import build_inputs_for_record, load_processor_for_model
from evaluation.recap_overhead_utils import (
    StageProfiler, distribution, instrument_model, kernel_stage_summary,
    object_footprint, rss_bytes,
)
from method.sam_grounded.tree_recycling_model import (
    TreeRecyclingSpecModel as Model, PERSISTENT_OPTIMIZED_DEPTH_CONFIGS,
)

POLICY = 'context-score-trigram-fusion-persistent-suffix4-hotpath-cpp-depth10-node63-wide-plus4'
MODELS = {'qwen': 'Qwen2.5-VL-7B-Instruct', 'llava': 'llava-v1.6-vicuna-7b-hf'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def options(cap, namespace):
    return dict(draft_policy=POLICY, matrix_top_k=8, tree_fixed_width=8,
                tree_fixed_depth=10, tree_broad_width=4, tree_shallow_depth=3,
                tree_node_budget=cap, recycling_prompt_init='unigram',
                recycling_max_order=3, recycling_suffix_length=4,
                recycling_suffix_implementation='fixed', recycling_suffix_bridge=False,
                recycling_sam_clone_mode='flat', recycling_cache_namespace=namespace)


def run_one(model, inputs, cap, namespace, max_tokens, measured=True, trace_path=None):
    counters = {}
    profiler = StageProfiler(annotate=trace_path is not None) if measured else None
    trace = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                profile_memory=False, with_stack=False) if trace_path else None)
    context = instrument_model(model, profiler) if profiler else nullcontext()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    rss_before = rss_bytes()
    with context, (trace if trace is not None else nullcontext()):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if profiler:
            profiler.switch('initialization_other')
        output, n_new, iterations, accepted = model.specgenerate(
            **inputs, temperature=0, max_new_tokens=max_tokens, log=True,
            return_acceptance_len=True, return_policy_trace=False,
            recycling_stats=counters, recycling_profiler=profiler,
            **options(cap, namespace),
        )
        if profiler:
            profiler.finish()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    prompt_length = inputs.input_ids.shape[1]
    tokens = output[0, prompt_length:].cpu().tolist()
    if len(tokens) != n_new or not 0 < len(tokens) <= max_tokens:
        raise RuntimeError('invalid generation length')
    if counters.get('max_candidate_nodes', 0) > cap + 4:
        raise RuntimeError('candidate cap exceeded')
    memory = dict(profiler.memory_objects) if profiler else {}
    if memory:
        memory['suffix_index'] = model.draft.sam_dyn
    footprints = {name: object_footprint(value) for name, value in memory.items()}
    footprint_all = object_footprint(memory)
    payload = dict(
        measured=measured, diagnostic_kernel_trace=trace is not None,
        transition_cap=cap, max_packed_nodes=cap + 5,
        new_tokens=n_new, iterations=iterations, accepted=accepted,
        average_accept_length=statistics.fmean(accepted) if accepted else 0.0,
        average_tokens_per_iteration=(n_new - 1) / len(accepted) if accepted else 0.0,
        output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        wall_seconds=elapsed, counters=counters,
        stages=dict(profiler.rows) if profiler else {}, memory=footprints,
        memory_union=footprint_all, rss_before_bytes=rss_before, rss_after_bytes=rss_bytes(),
        cuda_allocated_before_bytes=allocated_before,
        cuda_peak_allocated_bytes=peak_allocated, cuda_peak_reserved_bytes=peak_reserved,
    )
    if trace is not None:
        trace.export_chrome_trace(str(trace_path))
        payload['kernel_stages'] = kernel_stage_summary(trace_path)
        if not any(r['device_total_us'] > 0 for r in payload['kernel_stages'].values()):
            raise RuntimeError('CUDA kernel timing unavailable; do not substitute CPU time')
    if profiler:
        profiled = sum(row['wall_seconds'] for row in profiler.rows.values())
        payload['unassigned_instrumentation_seconds'] = elapsed - profiled
        if profiled > elapsed + 1e-6:
            raise RuntimeError('stage times overlap')
    return payload


def summarize(root, records, expected):
    measured = [r for r in records if r['measured'] and not r['diagnostic_kernel_trace']]
    summaries = {}
    for cap in sorted({r['transition_cap'] for r in measured}):
        rows = [r for r in measured if r['transition_cap'] == cap]
        if len(rows) != expected:
            raise RuntimeError(f'incomplete cap {cap}: {len(rows)} != {expected}')
        stage_names = sorted({key for r in rows for key in r['stages']})
        stages = {}
        for name in stage_names:
            times = [r['stages'].get(name, {}).get('wall_seconds', 0.0) for r in rows]
            per_iter = [t / max(len(r['accepted']), 1) * 1000 for r, t in zip(rows, times)]
            stages[name] = dict(wall_ms_per_request=distribution(t * 1000 for t in times),
                wall_ms_per_iteration=distribution(per_iter),
                pooled_generation_time_fraction=sum(times) / sum(r['wall_seconds'] for r in rows),
                process_cpu_ms_per_request=distribution(
                    r['stages'].get(name, {}).get('process_cpu_seconds', 0.0) * 1000 for r in rows))
        accept = [r['average_accept_length'] for r in rows]
        summary = dict(num_records=len(rows), unique_inputs=len({r['sample_id'] for r in rows}),
            transition_cap=cap, stages=stages, latency_seconds=distribution(r['wall_seconds'] for r in rows),
            avg_accept_length=statistics.fmean(accept), median_accept_length=statistics.median(accept),
            max_accept_length=max(accept), p90_accept_length=distribution(accept)['p90'],
            avg_tokens_per_iteration=statistics.fmean(r['average_tokens_per_iteration'] for r in rows),
            avg_sample_speedup=None, median_sample_speedup=None, max_sample_speedup=None,
            sample_speedup_gt_1_ratio=None, accept_le_0_5_ratio=statistics.fmean(a <= .5 for a in accept),
            speedup_note='latency diagnostic without an autoregressive comparator; not a headline speedup run',
            jsonl_path=str((root / f'cap{cap}/results.jsonl').resolve()),
            memory={name: distribution(r['memory'][name]['host_object_bytes'] for r in rows)
                    for name in ('U', 'C', 'G', 'topology', 'suffix_index')},
            memory_union_host_bytes=distribution(r['memory_union']['host_object_bytes'] for r in rows),
            cuda_peak_allocated_bytes=distribution(r['cuda_peak_allocated_bytes'] for r in rows),
            cuda_peak_reserved_bytes=distribution(r['cuda_peak_reserved_bytes'] for r in rows),
            observed_candidate_nodes=distribution(r['counters']['total_candidate_nodes'] /
                max(r['counters']['iterations'], 1) for r in rows),
            by_dataset={dataset: dict(num_records=sum(r['dataset'] == dataset for r in rows),
                latency_seconds=distribution(r['wall_seconds'] for r in rows if r['dataset'] == dataset))
                for dataset in sorted({r['dataset'] for r in rows})})
        write_json(root / f'cap{cap}/summary.json', summary)
        summaries[str(cap)] = summary
    paired = defaultdict(dict)
    for row in records:
        if row['transition_cap'] == 63 and not row['diagnostic_kernel_trace']:
            paired[(row['repeat_seed'], row['sample_id'])][row['measured']] = row
    complete_pairs = [p for p in paired.values() if len(p) == 2]
    calibration = dict(num_pairs=len(complete_pairs),
        profile_over_unprofiled_latency=distribution(p[True]['wall_seconds'] / p[False]['wall_seconds'] for p in complete_pairs),
        identical_tokens_and_acceptance=sum(p[True]['output_sha256'] == p[False]['output_sha256'] and
            p[True]['accepted'] == p[False]['accepted'] for p in complete_pairs))
    if complete_pairs and calibration['identical_tokens_and_acceptance'] != len(complete_pairs):
        raise RuntimeError('instrumentation changed generated tokens or acceptance')
    unprofiled = [r for r in records if not r['measured']]
    if unprofiled:
        write_json(root / 'unprofiled_cap63/summary.json', dict(
            complete=True, num_records=len(unprofiled),
            latency_seconds=distribution(r['wall_seconds'] for r in unprofiled),
            avg_accept_length=statistics.fmean(r['average_accept_length'] for r in unprofiled),
            avg_tokens_per_iteration=statistics.fmean(r['average_tokens_per_iteration'] for r in unprofiled),
            jsonl_path=str((root / 'unprofiled_cap63/results.jsonl').resolve()),
            comparison_scope='same-method normal-latency control for diagnostic overhead'))
    write_json(root / 'summary.json', dict(complete=True, budgets=summaries, calibration=calibration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--samples-per-dataset', type=int, default=10)
    parser.add_argument('--repeat-seeds', default='17,29')
    parser.add_argument('--caps', default='31,63,127')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--kernel-traces', action='store_true')
    parser.add_argument('--manifest-dir', type=Path, default=PROJECT_ROOT /
        'outputs/experiments/recap/20260907-171145-recap-online-confirm4x100-gpu2/manifests')
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    caps = [int(x) for x in args.caps.split(',')]
    seeds = [int(x) for x in args.repeat_seeds.split(',')]
    if not caps or min(caps) < 1 or max(caps) > 127:
        parser.error('caps must be within [1, 127]')
    torch.set_num_threads(4)
    PERSISTENT_OPTIMIZED_DEPTH_CONFIGS[POLICY] = (max(caps), 10)
    sources = [Path(__file__), SCRIPT_DIR / 'recap_overhead_utils.py',
        SCRIPT_DIR / 'recap_overhead_trace.py',
        SCRIPT_DIR.parent / 'method/sam_grounded/tree_recycling_model.py',
        SCRIPT_DIR.parent / 'method/sam_grounded/fast_tree_builder.cpp',
        SCRIPT_DIR.parent / 'method/sam_grounded/fast_tree_builder.py',
        SCRIPT_DIR.parent / 'method/sam/samd/dyn_sam.py',
        PROJECT_ROOT / 'method/fovea/evaluation/eval_llava_fixed_multi.py']
    hashes = {}
    for source in sources:
        relative = source.relative_to(PROJECT_ROOT)
        destination = root / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(root / 'protocol.json', dict(arguments={k:str(v) if isinstance(v, Path) else v
        for k,v in vars(args).items()}, policy=POLICY, resolved_settings={str(c):options(c, 'separate namespace') for c in caps},
        torch=torch.__version__, python=sys.version, platform=platform.platform(),
        process_id=os.getpid(), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        cuda_runtime=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=hashes,
        model_path=str(PROJECT_ROOT / 'models' / MODELS[args.model]),
        note='synchronized stage diagnostics; CUDA kernel traces and normal latency controls are separate'))
    print(f'stage=load_inputs model={args.model}', flush=True)
    samples, manifests = load_samples(args.manifest_dir, args.samples_per_dataset)
    write_json(root / 'manifest_paths.json', {k:str(v) for k,v in manifests.items()})
    model_path = str(PROJECT_ROOT / 'models' / MODELS[args.model])
    print(f'stage=load_model model={args.model} inputs={len(samples)}', flush=True)
    model = Model.from_pretrained(base_model_path=model_path, spec_model_path='', total_token=160,
        torch_dtype=torch.bfloat16 if args.model == 'qwen' else torch.float16,
        low_cpu_mem_usage=True, device_map='cuda:0', attn_implementation='sdpa')
    model.processor = load_processor_for_model(model_path)
    model.eval()
    for sample in {s['dataset']:s for s in reversed(samples)}.values():
        warm_inputs = build_inputs_for_record(model, sample['dataset'], sample['row'])
        for cap in caps:
            model.specgenerate(**warm_inputs, temperature=0, max_new_tokens=32, **options(cap, 'warmup'))
    model.reset_persistent_recycling_cache()
    print('stage=warmup_complete', flush=True)
    records = []
    for seed in seeds:
        model.reset_persistent_recycling_cache()
        stream = list(samples)
        random.Random(seed).shuffle(stream)
        write_json(root / f'stream_seed{seed}.json', [{k:s[k] for k in ('id','dataset','source_index')} for s in stream])
        arms = [(c, True) for c in caps] + ([(63, False)] if 63 in caps else [])
        for position, sample in enumerate(stream):
            inputs = build_inputs_for_record(model, sample['dataset'], sample['row'])
            for cap, measured in arms[position % len(arms):] + arms[:position % len(arms)]:
                namespace = f'seed{seed}:cap{cap}:profile{int(measured)}'
                row = run_one(model, inputs, cap, namespace, args.max_new_tokens, measured=measured)
                row.update(sample_id=sample['id'], dataset=sample['dataset'], source_index=sample['source_index'],
                           repeat_seed=seed, stream_position=position, model=args.model)
                directory = root / (f'cap{cap}' if measured else 'unprofiled_cap63')
                directory.mkdir(exist_ok=True)
                with (directory / 'results.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                records.append(row)
                print(f"model={args.model} seed={seed} request={position+1}/{len(stream)} cap={cap} profile={measured} tokens={row['new_tokens']} seconds={row['wall_seconds']:.3f}", flush=True)
        write_json(root / f'progress_seed{seed}.json', {'complete':True,'num_requests':len(stream)})
    summarize(root, records, len(samples) * len(seeds))
    if args.kernel_traces:
        trace_root = root / 'kernel_traces'
        trace_root.mkdir()
        selected = {s['dataset']:s for s in reversed(samples)}
        kernel_rows = []
        for dataset, sample in selected.items():
            model.reset_persistent_recycling_cache()
            inputs = build_inputs_for_record(model, dataset, sample['row'])
            trace_path = trace_root / (dataset.lower().replace('-','_') + '.json')
            row = run_one(model, inputs, 63, 'kernel_trace', min(args.max_new_tokens, 64), trace_path=trace_path)
            row.update(dataset=dataset, sample_id=sample['id'], model=args.model)
            kernel_rows.append(row)
            print(f'stage=kernel_trace_complete model={args.model} dataset={dataset}', flush=True)
        write_json(trace_root / 'summary.json', dict(complete=True, num_records=len(kernel_rows), records=kernel_rows,
            scope='separate diagnostic captures, up to 64 tokens; no latency-speedup claim'))
    summary = json.loads((root / 'summary.json').read_text())
    summary.update(model=args.model, kernel_traces_complete=args.kernel_traces,
                   finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_json(root / 'summary.json', summary)
    print(f'stage=finished model={args.model} output_root={root}', flush=True)


if __name__ == '__main__':
    main()
