"""Secondary source-selection diagnostic after the frozen fusion comparisons.

Distinguish confidence gating from choosing the most specific single source.
This analysis uses every saved state and does not select or tune a new policy.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    source = Path(json.loads((root / 'offline/source.json').read_text())['path'])
    groups = defaultdict(lambda: defaultdict(list))
    per_input = []
    with source.open() as original, (root / 'offline/results.jsonl').open() as replay:
        for first, second in zip(original, replay, strict=True):
            raw, saved = json.loads(first), json.loads(second)
            assert (raw['dataset'], raw['question_id']) == (saved['dataset'], saved['question_id'])
            rows = {p: ids for p, ids, _ in raw['prediction_rows']}
            item = dict(dataset=raw['dataset'], question_id=raw['question_id'], groups={})
            for name in ('all', 'source_disagreement'):
                states = saved['states'] if name == 'all' else [s for s in saved['states'] if s['source_disagreement']]
                values = np.zeros(6)
                for state in states:
                    sources = state['source_positions']
                    assert all(p < state['position'] for p in sources)
                    primary = [t for t in rows[sources[0]][:8] if t != raw['blocked_token_id']] if sources else []
                    label = rows[state['position']][0]
                    backoff = label in primary
                    fixed = state['methods']['fixed']['hit']
                    gate = state['methods']['confidence_gate']['hit']
                    values += [int(gate) - int(backoff), int(backoff) - int(fixed),
                               int(gate and not backoff), int(backoff and not gate), len(primary), 1]
                groups[(raw['dataset'], name)][saved['image_cluster']].append(values)
                item['groups'][name] = values.tolist()
            per_input.append(item)
    result = {}
    for (dataset, group), clusters in groups.items():
        array = np.array([np.sum(v, axis=0) for v in clusters.values()])
        total = array.sum(axis=0)
        rng = np.random.default_rng(20260911)
        boots = []
        for _ in range(2000):
            trial = array[rng.integers(0, len(array), len(array))].sum(axis=0)
            if trial[-1]:
                boots.append(trial[:2] / trial[-1])
        ci = np.quantile(boots, [.025, .975], axis=0)
        result.setdefault(dataset, {})[group] = dict(num_states=int(total[-1]),
            gate_minus_backoff_recall_pp=100 * total[0] / total[-1],
            gate_minus_backoff_ci95_pp=(100 * ci[:, 0]).tolist(),
            backoff_minus_fixed_recall_pp=100 * total[1] / total[-1],
            backoff_minus_fixed_ci95_pp=(100 * ci[:, 1]).tolist(),
            gate_only_hits=int(total[2]), backoff_only_hits=int(total[3]),
            mean_backoff_candidates=total[4] / total[-1])
    path = root / 'analysis/gate_diagnostic.json'
    path.write_text(json.dumps(dict(complete=True, analysis_role='secondary diagnostic after primary offline results',
        definitions=['gate-minus-backoff', 'backoff-minus-fixed', 'gate-only hits', 'backoff-only hits', 'candidate sum', 'states'],
        datasets=result, per_input=per_input), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
