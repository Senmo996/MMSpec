"""Audit completed adaptive-fusion controls and emit traceable paper tables."""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_recap_overhead import check_gpu_activity

RULES = ('fixed', 'uniform', 'confidence', 'prior_confidence', 'confidence_gate')
DATASETS = ('ChartQA', 'TextVQA', 'OCRBench', 'MMT-Bench')
NAMES = {'fixed': 'Fixed prior', 'uniform': 'Uniform', 'confidence': 'Confidence',
         'prior_confidence': 'Prior $\\times$ confidence', 'confidence_gate': 'Confidence gate',
         'backoff': 'Specificity backoff',
         'prior_soft': 'Prior $(0.55,0.30,0.15)$', 'prior_sharp': 'Prior $(0.85,0.10,0.05)$'}


def read(path):
    return json.loads(path.read_text())


def audit_model(root, model, log):
    directory = root / model
    summary = read(directory / 'summary.json')
    protocol = read(directory / 'protocol.json')
    assert summary['complete'] and summary['num_records'] == 2000
    assert protocol['arguments']['samples_per_dataset'] == 50
    assert protocol['arguments']['repeat_seeds'] == '17,29'
    assert protocol['arguments']['max_new_tokens'] == 200
    assert tuple(protocol['online_rules']) == RULES
    records = {mode: [json.loads(line) for line in (directory / mode / 'results.jsonl').read_text().splitlines()]
               for mode in RULES}
    keys = lambda row: (row['repeat_seed'], row['sample_id'])
    by_id = {mode: {keys(row): row for row in rows} for mode, rows in records.items()}
    base = by_id['fixed']
    assert len(base) == 400 and len({k[1] for k in base}) == 200
    for mode in RULES:
        assert len(records[mode]) == len(by_id[mode]) == 400 and by_id[mode].keys() == base.keys()
        for key, row in by_id[mode].items():
            fixed = base[key]
            assert row['counters']['max_candidate_nodes'] <= 67
            assert row['stream_position'] == fixed['stream_position']
            assert row['source_index'] == fixed['source_index']
            ratio = row['new_tokens'] / row['wall_seconds'] / (fixed['new_tokens'] / fixed['wall_seconds'])
            assert abs(row['speedup'] - ratio) < 1e-12
        for dataset in DATASETS:
            rows = [r for r in records[mode] if r['dataset'] == dataset]
            assert len(rows) == 100 and len({r['sample_id'] for r in rows}) == 50
            reported = summary['rules'][mode]['by_dataset'][dataset]
            assert abs(reported['mean_speed_ratio'] - statistics.fmean(r['speedup'] for r in rows)) < 1e-12
    activity = check_gpu_activity(log, protocol, summary)
    activity['scope'] = 'model loading, warmup, and online fusion evaluations; idle co-resident allocations allowed'
    return dict(summary=summary, gpu_activity=activity,
                source_sha256=protocol['source_sha256'], manifests=read(directory / 'manifests.json'))


def tables(analysis, payload):
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}', r'\toprule',
             r'Model & Fusion rule & ChartQA & TextVQA & OCRBench & MMT-Bench & Mean & $\tau$ \\', r'\midrule']
    for model in ('qwen', 'llava'):
        if model == 'llava':
            lines.append(r'\midrule')
        for mode in RULES:
            s = payload['models'][model]['summary']['rules'][mode]
            cells = ['Qwen' if model == 'qwen' else 'LLaVA', NAMES[mode]]
            cells.extend(f"${100 * (s['by_dataset'][d]['mean_speed_ratio'] - 1):+.2f}$" for d in DATASETS)
            macro = statistics.fmean(s['by_dataset'][d]['mean_speed_ratio'] for d in DATASETS)
            cells.extend([f'${100 * (macro - 1):+.2f}$', f"${1 + s['avg_accept_length']:.2f}$"])
            lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}',
        r'\caption{Online throughput change (\%) relative to fixed-prior fusion, with mean acceptance length $\tau$. Each model uses $200$ distinct inputs ($50$ per dataset), repeated under two request orders; entries average input-paired throughput ratios. All rules share memory scope, candidate limits, and suffix insertion. Adaptation cost is included.}',
        r'\label{tab:adaptive-fusion-online}', r'\end{table*}'])
    (analysis / 'online_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{2pt}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}', r'\toprule',
             r'Fusion rule & ChartQA & TextVQA & OCR & MMT \\', r'\midrule']
    for mode in (*RULES, 'backoff', 'prior_soft', 'prior_sharp'):
        cells = [NAMES[mode]]
        if mode == 'backoff':
            lines.append(r'\midrule')
            cells.extend(f"${payload['gate_diagnostic']['datasets'][d]['all']['backoff_minus_fixed_recall_pp']:+.2f}$" for d in DATASETS)
        else:
            cells.extend(f"${100 * payload['offline']['datasets'][d]['paired_vs_fixed'][mode]['recall_delta']:+.2f}$" for d in DATASETS)
        lines.append(' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}',
        r'\caption{Change in all-query Recall@$8$ (percentage points) relative to fixed-prior fusion on $23{,}811$ saved Qwen states from $400$ distinct inputs. Uncovered queries count as misses.}',
        r'\label{tab:adaptive-fusion-offline}', r'\end{table}'])
    (analysis / 'offline_table.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    launch = read(root / 'launch.json')
    offline = read(root / 'offline/summary.json')
    assert offline['complete'] and offline['num_records'] == 400 and offline['num_states'] == 23811
    gate = read(root / 'analysis/gate_diagnostic.json')
    assert gate['complete']
    for dataset in DATASETS:
        secondary = gate['datasets'][dataset]['all']
        assert secondary['num_states'] == offline['datasets'][dataset]['num_states']
        total = secondary['gate_minus_backoff_recall_pp'] + secondary['backoff_minus_fixed_recall_pp']
        assert abs(total - 100 * offline['datasets'][dataset]['paired_vs_fixed']['confidence_gate']['recall_delta']) < 1e-12
    result = dict(complete=True, offline=offline, gate_diagnostic=gate, models={})
    for model in ('qwen', 'llava'):
        result['models'][model] = audit_model(root, model, Path(launch['gpu_activity_log']))
    qwen, llava = (result['models'][m] for m in ('qwen', 'llava'))
    assert qwen['source_sha256'] == llava['source_sha256'], 'source changed between model runs'
    assert qwen['manifests'] == llava['manifests']
    analysis = root / 'analysis'
    analysis.mkdir(exist_ok=True)
    (analysis / 'paper_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    tables(analysis, result)
    (root / 'summary.json').write_text(json.dumps(dict(complete=True,
        offline_unique_inputs=400, offline_states=23811, online_unique_inputs_per_model=200,
        online_generation_runs=4000, models=['qwen', 'llava'],
        gpu_activity_validated=True, analysis=str(analysis)), indent=2) + '\n')
    for model in ('qwen', 'llava'):
        print(model, {mode: round(100 * (result['models'][model]['summary']['rules'][mode]['avg_sample_speedup'] - 1), 3) for mode in RULES})
    print('complete=true', analysis)


if __name__ == '__main__':
    main()
