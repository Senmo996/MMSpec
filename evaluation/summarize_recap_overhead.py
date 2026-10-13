"""Validate completed overhead runs and produce paper-facing, traceable numbers."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recap_overhead_utils import distribution
from recap_overhead_trace import summarize_trace

STAGES = {
    'suffix_lookup': ('suffix_lookup',),
    'tree_search_fusion': ('tree_search_fusion',),
    'packing': ('topology_packing', 'verification_packing'),
    'memory_update': ('memory_update',),
    'suffix_index': ('suffix_index',),
    'accept_commit': ('accept_commit',),
    'decode_other': ('decode_other',),
    'target_verification': ('target_verification',),
}
PREPARATION = ('suffix_lookup', 'tree_search_fusion', 'topology_packing', 'verification_packing')


def json_read(path):
    return json.loads(path.read_text())


def records_read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stage_seconds(row, keys, kind='wall_seconds'):
    return sum(row['stages'].get(k,{}).get(kind,0.0) for k in keys)


def check_gpu_activity(path, protocol, summary):
    start=datetime.strptime(protocol['started_at'],'%Y-%m-%dT%H:%M:%S%z').replace(tzinfo=None)
    end=datetime.strptime(summary['finished_at'],'%Y-%m-%dT%H:%M:%S%z').replace(tzinfo=None)
    own_pid=protocol['process_id']
    timestamps=set()
    own_active=set()
    competing=[]
    for line in path.read_text().splitlines():
        cells=line.split()
        if len(cells)<7 or not cells[0].isdigit():
            continue
        stamp=datetime.strptime(' '.join(cells[:2]),'%Y%m%d %H:%M:%S')
        if not start<=stamp<=end:
            continue
        timestamps.add(stamp)
        if 'C' not in cells[4] or not cells[5].isdigit() or int(cells[5])==0:
            continue
        if int(cells[3])==own_pid:
            own_active.add(stamp)
        else:
            competing.append({'time':stamp.isoformat(),'pid':int(cells[3]),'sm_percent':int(cells[5])})
    assert len(own_active)>=10, 'missing process-level GPU activity evidence'
    assert not competing, f'concurrent GPU compute observed: {competing[:5]}'
    return {'process_id':own_pid,'sampled_seconds':len(timestamps),
            'own_active_seconds':len(own_active),'competing_active_samples':0,
            'sampling_interval_seconds':1,
            'scope':'model loading, warmup, timing and kernel captures; idle co-resident allocations allowed'}


def write_paper_tables(directory, payload):
    labels=[('suffix_lookup','Suffix lookup'),('tree_search_fusion','Lookup, fusion, search'),
            ('packing','Packing'),('memory_update','Memory refresh'),
            ('suffix_index','Suffix index update'),('accept_commit','Token commitment'),
            ('decode_other','Other decoding'),('target_verification','Target verification')]
    lines=[r'\begin{table}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{2.5pt}',
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',
        r' & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{LLaVA} \\',
        r'Stage & Wall & CPU & Wall & CPU \\',r'\midrule']
    for key,label in labels:
        cells=[label]
        for model in ('qwen','llava'):
            s=payload['models'][model]['budgets']['63']['stages'][key]
            cells.extend(f"${s[metric]:.3f}$" for metric in
                ('pooled_wall_ms_per_iteration','pooled_process_cpu_ms_per_iteration'))
        lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular*}',
        r'\caption{Iteration-pooled phase times (ms) at the default $63$-node transition cap: $80$ runs per model over $40$ distinct inputs and two order seeds. Wall times are exclusive; process CPU time includes synchronization polling and is not added to elapsed time.}',
        r'\label{tab:overhead-stages}',r'\end{table}'])
    (directory/'latency_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table*}[t]',r'\centering',r'\small',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}',r'\toprule',
        r'Model & Cap & Mean nodes & Prep. ms & $p_{95}$ ms & Host MiB & GPU GiB \\',r'\midrule']
    for model in ('qwen','llava'):
        if model=='llava': lines.append(r'\midrule')
        for cap in ('31','63','127'):
            s=payload['models'][model]['budgets'][cap]
            cells=['Qwen' if model=='qwen' else 'LLaVA',cap,
                f"{s['actual_candidate_nodes_per_iteration']:.1f}",
                f"{s['pooled_preparation_ms_per_iteration']:.3f}",
                f"{s['request_mean_preparation_ms_per_iteration']['p95']:.3f}",
                f"{s['memory_union_host_MiB_max']:.2f}",
                f"{s['cuda_peak_allocated_GiB']:.2f}"]
            lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular*}',
        r'\caption{Candidate-budget sensitivity, with $80$ runs per model and cap. Node counts exclude the root and include suffix additions. Preparation pools retrieval, fusion, tree construction, and packing over iterations; $p_{95}$ is the 95th percentile of request-average preparation time per iteration. Host MiB is the largest post-request snapshot of proposal state; GPU GiB is peak allocated memory for the model process in the interleaved sweep, including weights, prefill, key/value state, and live workspaces.}',
        r'\label{tab:overhead-budgets}',r'\end{table*}'])
    (directory/'budget_table.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    root=args.run_root.resolve()
    payload={'complete':True,'run_root':str(root),'models':{},'source_sha256':{}}
    launch=json_read(root/'launch.json')
    activity_path=Path(launch['launcher_log']).with_name(launch['run_tag']+'.gpu-activity.log')
    for model in ('qwen','llava'):
        summary=json_read(root/model/'summary.json')
        protocol=json_read(root/model/'protocol.json')
        assert summary['complete'] and summary['kernel_traces_complete']
        assert summary['calibration']['identical_tokens_and_acceptance']==summary['calibration']['num_pairs']==80
        control_path=root/model/'unprofiled_cap63/results.jsonl'
        controls=records_read(control_path)
        control_by_id={(r['repeat_seed'],r['sample_id']):r for r in controls}
        assert len(control_by_id)==len(controls)==80
        assert all(not r['measured'] and r['transition_cap']==63 for r in controls)
        control_summary=json_read(control_path.with_name('summary.json'))
        accept=distribution(r['average_accept_length'] for r in controls)
        control_summary.update(avg_accept_length=accept['mean'],median_accept_length=accept['median'],
            max_accept_length=accept['maximum'],p90_accept_length=accept['p90'],
            avg_sample_speedup=None,median_sample_speedup=None,max_sample_speedup=None,
            sample_speedup_gt_1_ratio=None,
            accept_le_0_5_ratio=statistics.fmean(r['average_accept_length']<=.5 for r in controls),
            speedup_note='same-method control; no autoregressive comparator in this cost diagnostic')
        control_path.with_name('summary.json').write_text(json.dumps(control_summary,indent=2)+'\n')
        budgets={}
        for cap in (31,63,127):
            path=root/model/f'cap{cap}/results.jsonl'
            rows=records_read(path)
            payload['source_sha256'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            identities={(r['repeat_seed'],r['sample_id']) for r in rows}
            assert len(identities)==len(rows)==80
            assert len({r['sample_id'] for r in rows})==40
            assert {r['repeat_seed'] for r in rows}=={17,29}
            assert {r['dataset'] for r in rows}=={'ChartQA','TextVQA','OCRBench','MMT-Bench'}
            assert all(sum(r['dataset']==dataset for r in rows)==20 for dataset in {r['dataset'] for r in rows})
            assert all(r['measured'] and not r['diagnostic_kernel_trace'] for r in rows)
            assert all(r['counters']['max_candidate_nodes']<=cap+4 for r in rows)
            assert all(r['counters']['iterations']==len(r['accepted']) for r in rows)
            assert all(r['memory']['U']['cuda_tensor_storage_bytes']==0 for r in rows)
            assert all(sum(x['wall_seconds'] for x in r['stages'].values())<=r['wall_seconds']+1e-6 for r in rows)
            if cap==63:
                assert identities==set(control_by_id)
                assert all(r['output_sha256']==control_by_id[(r['repeat_seed'],r['sample_id'])]['output_sha256'] and
                    r['accepted']==control_by_id[(r['repeat_seed'],r['sample_id'])]['accepted'] for r in rows)
            elapsed=sum(r['wall_seconds'] for r in rows)
            iterations=sum(len(r['accepted']) for r in rows)
            preparation=[stage_seconds(r,PREPARATION) for r in rows]
            stage_summary={}
            for name,keys in STAGES.items():
                times=[stage_seconds(r,keys) for r in rows]
                cpu=[stage_seconds(r,keys,'process_cpu_seconds') for r in rows]
                stage_summary[name]={
                    'pooled_wall_ms_per_iteration':sum(times)/iterations*1000,
                    'pooled_process_cpu_ms_per_iteration':sum(cpu)/iterations*1000,
                    'request_mean_wall_ms_per_iteration':distribution(t/max(len(r['accepted']),1)*1000 for r,t in zip(rows,times)),
                    'fraction_of_generation_wall_time':sum(times)/elapsed}
            decode_keys=tuple(k for keys in STAGES.values() for k in keys)
            decode_seconds=sum(stage_seconds(r,decode_keys) for r in rows)
            budgets[str(cap)]={
                'num_records':len(rows),'unique_inputs':40,'iterations':iterations,
                'stages':stage_summary,
                'pooled_preparation_ms_per_iteration':sum(preparation)/iterations*1000,
                'request_mean_preparation_ms_per_iteration':distribution(
                    t/max(len(r['accepted']),1)*1000 for r,t in zip(rows,preparation)),
                'preparation_generation_fraction':sum(preparation)/elapsed,
                'preparation_decode_fraction':sum(preparation)/decode_seconds,
                'actual_candidate_nodes_per_iteration':sum(r['counters']['total_candidate_nodes'] for r in rows)/iterations,
                'cuda_peak_allocated_GiB':max(r['cuda_peak_allocated_bytes'] for r in rows)/2**30,
                'cuda_peak_reserved_GiB':max(r['cuda_peak_reserved_bytes'] for r in rows)/2**30,
                'U_host_MiB_max':max(r['memory']['U']['host_object_bytes'] for r in rows)/2**20,
                'CG_host_MiB_max':max(r['memory']['C']['host_object_bytes']+r['memory']['G']['host_object_bytes'] for r in rows)/2**20,
                'memory_union_host_MiB_max':max(r['memory_union']['host_object_bytes'] for r in rows)/2**20,
                'topology_cuda_MiB_max':max(r['memory']['topology']['cuda_tensor_storage_bytes'] for r in rows)/2**20,
                'mean_generated_tokens':statistics.fmean(r['new_tokens'] for r in rows),
                'by_seed':{str(seed):{'preparation_ms_per_iteration':
                    sum(stage_seconds(r,PREPARATION) for r in rows if r['repeat_seed']==seed)/
                    sum(len(r['accepted']) for r in rows if r['repeat_seed']==seed)*1000}
                    for seed in (17,29)}}
        trace_summary=json_read(root/model/'kernel_traces/summary.json')
        assert trace_summary['complete'] and trace_summary['num_records']==4
        diagnostic_dir=root/model/'kernel_traces/diagnostics'
        diagnostic_dir.mkdir(exist_ok=True)
        legacy_path=diagnostic_dir/'original_range_summary.json'
        if not legacy_path.exists():
            legacy_path.write_text(json.dumps(trace_summary,indent=2)+'\n')
        trace_rows=trace_summary['records']
        kernel_totals=defaultdict(float)
        for r in trace_rows:
            trace_path=root/model/'kernel_traces'/(r['dataset'].lower().replace('-','_')+'.json')
            r['kernel_stages'],r['kernel_event_accounting']=summarize_trace(trace_path)
            for name,value in r['kernel_stages'].items():
                kernel_totals[name]+=value['device_total_us']
        trace_summary['device_timing_method']='direct CUDA kernel/memcpy/memset duration sums; annotation spans excluded'
        (root/model/'kernel_traces/summary.json').write_text(json.dumps(trace_summary,indent=2)+'\n')
        denominator=sum(len(r['accepted']) for r in trace_rows)
        decoding=sum(kernel_totals[name] for keys in STAGES.values() for name in keys)
        payload['models'][model]={
            'budgets':budgets,'calibration':summary['calibration'],
            'gpu_activity':check_gpu_activity(activity_path,protocol,summary),
            'kernel_traces':{'num_records':4,'iterations':denominator,
                'device_ms_per_iteration':{k:v/denominator/1000 for k,v in kernel_totals.items()},
                'target_fraction_of_decode_device_time':kernel_totals['target_verification']/decoding,
                'note':'independent 4-input, up-to-64-token captures; kernel duration is not synchronized stage wall time'}}
    memory_path=root/'memory/summary.json'
    memory=json_read(memory_path)
    assert memory['complete'] and memory['num_records']==21
    payload['source_sha256']['memory/summary.json']=hashlib.sha256(memory_path.read_bytes()).hexdigest()
    payload['memory']=memory
    analysis=root/'analysis'
    analysis.mkdir(exist_ok=True)
    (analysis/'paper_metrics.json').write_text(json.dumps(payload,indent=2)+'\n')
    write_paper_tables(analysis,payload)
    (root/'summary.json').write_text(json.dumps({'complete':True,
        'models':['qwen','llava'],'unique_inputs_per_model':40,'repeat_seeds':[17,29],
        'transition_caps':[31,63,127],'profiled_generations':480,'unprofiled_generations':160,
        'separate_kernel_trace_generations':8,'natural_replay_requests_per_seed':10000,
        'natural_replay_unique_trajectories':400,'capacity_points':6,
        'paper_metrics_path':str(analysis/'paper_metrics.json')},indent=2)+'\n')
    for model,s in payload['models'].items():
        print(model,'calibration',s['calibration']['profile_over_unprofiled_latency']['median'])
        for cap,r in s['budgets'].items():
            print('cap',cap,'prepare_ms',round(r['pooled_preparation_ms_per_iteration'],4),
                'decode_percent',round(r['preparation_decode_fraction']*100,2),
                'peak_GiB',round(r['cuda_peak_allocated_GiB'],3))
    print('complete=true',analysis/'paper_metrics.json')


if __name__=='__main__':
    main()
