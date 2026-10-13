"""Natural-trajectory update replay and separate full-vocabulary capacity stress."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
from recap_overhead_utils import distribution, object_footprint, rss_bytes


def clone_row(ids, scores):
    # The real .tolist() creates scalar Python objects for each prediction row.
    return [int(str(v)) for v in ids], [float(str(v)) for v in scores]


def capacity_worker(vocab, count, k=8):
    gc.collect()
    before = rss_bytes()
    rows, scores = {}, {}
    for token in range(count):
        ids = [(token + j + 1) % vocab for j in range(k)]
        probability = [1 / (j + 1) for j in range(k)]
        total = sum(probability)
        rows[token], scores[token] = clone_row(ids, [v / total for v in probability])
    cache = {'transitions': rows, 'scores': scores}
    after = rss_bytes()  # Excludes the auxiliary set used by the size traversal.
    footprint = object_footprint(cache)
    rng = random.Random(42)
    queries = [rng.randrange(count) for _ in range(100000)]
    latencies = []
    checksum = 0.0
    for _ in range(7):
        start = time.perf_counter_ns()
        for token in queries:
            row = rows.get(token)
            weight = scores.get(token)
            checksum += row[0] * weight[0]
        latencies.append((time.perf_counter_ns() - start) / len(queries))
    from method.sam_grounded.fast_tree_builder import build_score_priority_nodes
    build_args = dict(root_previous_token=-1, root_previous_previous_token=-1,
        width=8, branch_width=8, depth=10, node_budget=63, blocked_token_id=-1,
        unigram_rows=rows, unigram_scores=scores, context_rows={}, context_scores={},
        trigram_rows={}, trigram_scores={}, hit_masses=[.68,.63,.50,.43,.40,.38])
    build_score_priority_nodes(root_token=0, **build_args)  # load/warm outside timing
    roots = [rng.randrange(max(count-100, 1)) for _ in range(128)]
    tree_times = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for token in roots:
            nodes = build_score_priority_nodes(root_token=token, **build_args)
            if len(nodes) != 63:
                raise RuntimeError('capacity timing must compare full 63-node trees')
        tree_times.append((time.perf_counter_ns()-start)/len(roots)/1000)
    # Repeated overwrites must not append histories to U or add vocabulary keys.
    before_count = len(rows)
    for token in queries[:10000]:
        rows[token], scores[token] = clone_row(rows[token], scores[token])
    after_count = len(rows)
    if before_count != after_count or not checksum > 0:
        raise RuntimeError('capacity/update integrity check failed')
    return dict(vocab_size=vocab, rows=count, k=k, footprint=footprint,
        process_rss_before_bytes=before, process_rss_after_fill_bytes=after,
        process_rss_growth_bytes=after-before,
        lookup_ns_per_query_by_repeat=latencies, lookup_ns_per_query=distribution(latencies),
        lookup_scope='two host dictionary gets plus access to first ID/score; 100000 seeded hits per repeat',
        overwrite_count=10000, rows_after_overwrites=after_count,
        tree_search_fusion_us_per_call=distribution(tree_times),
        tree_search_fusion_us_by_repeat=tree_times,
        tree_scope='actual host builder, 63 nodes, width 8, depth 10, U-only synthetic rows; packing excluded',
        source='synthetic capacity stress, not observed target predictions',
        logical_int64_float32_payload_bytes=count*k*12)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--source',type=Path,default=PROJECT_ROOT /
        'outputs/experiments/recap/20260907-163640-recap-path-coverage4x100-direct-gpu2/full/results.jsonl')
    parser.add_argument('--requests',type=int,default=10000)
    parser.add_argument('--seeds',default='17,29,43')
    parser.add_argument('--capacity-worker',nargs=2,type=int,metavar=('VOCAB','ROWS'))
    args=parser.parse_args()
    if args.capacity_worker:
        print(json.dumps(capacity_worker(*args.capacity_worker)))
        return
    if args.output_root is None:
        parser.error('--output-root is required')
    root=args.output_root.resolve()
    root.mkdir(parents=True,exist_ok=False)
    saved=[json.loads(line) for line in args.source.open() if line.strip()]
    requests=[]
    for record in saved:
        tokens=record['input_token_ids']
        updates=[]
        for position, ids, scores in record['prediction_rows']:
            key=tokens[position]
            if key == record['blocked_token_id']:
                continue
            if len(ids)!=8 or len(scores)!=8 or len(set(ids))!=8:
                raise RuntimeError('invalid source row')
            updates.append((key,ids,scores))
        requests.append({'id':record['question_id'],'dataset':record['dataset'],'updates':updates})
    if len({r['id'] for r in requests})!=len(requests) or len(requests)!=400:
        raise RuntimeError('require the original 400 distinct trajectories')
    del saved
    gc.collect()
    checkpoints=sorted({1,10,100,400,1000,5000,args.requests})
    checkpoints=[c for c in checkpoints if c<=args.requests]
    seeds=[int(x) for x in args.seeds.split(',')]
    all_rows=[]
    for seed in seeds:
        rows,scores={},{}
        cache={'transitions':rows,'scores':scores}
        rng=random.Random(seed)
        count=0
        stream_hash=hashlib.sha256()
        while count<args.requests:
            order=list(range(len(requests)))
            rng.shuffle(order)
            for index in order:
                if count>=args.requests:
                    break
                record=requests[index]
                stream_hash.update((record['id']+'\n').encode())
                start=time.perf_counter_ns()
                for key,ids,weights in record['updates']:
                    rows[key],scores[key]=clone_row(ids,weights)
                update_ns=time.perf_counter_ns()-start
                count+=1
                if count in checkpoints:
                    rss=rss_bytes()
                    footprint=object_footprint(cache)
                    item=dict(seed=seed,requests=count,unique_source_requests=min(count,400),
                        rows=len(rows),footprint=footprint,process_rss_before_size_walk_bytes=rss,
                        last_request_update_ns=update_ns,stream_sha256=stream_hash.hexdigest())
                    all_rows.append(item)
                    with (root/'results.jsonl').open('a') as handle:
                        handle.write(json.dumps(item)+'\n')
                    print(f'stage=natural_replay seed={seed} requests={count} rows={len(rows)} host_MiB={footprint["host_object_bytes"]/2**20:.3f}',flush=True)
        if len(rows)>152064:
            raise RuntimeError('U exceeded vocabulary bound')
    capacities=[]
    for model,vocab in [('qwen',152064),('llava',32064)]:
        for count in [1000,10000,vocab]:
            # Each point has its own process, so RSS excludes previous size walks.
            value=subprocess.check_output([sys.executable,str(Path(__file__).resolve()),
                '--capacity-worker',str(vocab),str(count)],text=True)
            result=json.loads(value)
            result['model']=model
            capacities.append(result)
            print(f'stage=capacity model={model} rows={count} host_MiB={result["footprint"]["host_object_bytes"]/2**20:.3f}',flush=True)
    summary=dict(complete=True,num_records=len(all_rows),natural_replay=dict(
        source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        model='Qwen2.5-VL-7B-Instruct',unique_requests=400,requests_per_seed=args.requests,seeds=seeds,
        source_scope='saved native-trajectory Top-8 rows; cyclic replay, not fresh online generation',
        checkpoints={str(c):{'rows':distribution(r['rows'] for r in all_rows if r['requests']==c),
            'host_object_bytes':distribution(r['footprint']['host_object_bytes'] for r in all_rows if r['requests']==c)}
            for c in checkpoints}),capacity_stress=capacities,
        bound='U has at most |V| keys and stores K candidate IDs and K scores per key; O(|V|K) space',
        cuda_tensor_storage_bytes=0,
        caveats=['RSS includes the interpreter and source trajectory pool; only isolated capacity RSS deltas characterize allocation growth.',
                 'Natural replay covers 400 distinct frozen trajectories. Repetitions test replacement and leakage, not new-task coverage.',
                 'The synthetic full-occupancy test is an upper-capacity measurement, not a natural-stream observation.'])
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'stage=finished output_root={root}',flush=True)


if __name__=='__main__':
    main()
