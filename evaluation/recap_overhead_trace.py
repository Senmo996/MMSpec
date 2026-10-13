"""Sum actual CUDA work from Kineto traces, excluding GPU annotation spans."""

from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path


DEVICE_CATEGORIES = {'kernel', 'gpu_memcpy', 'gpu_memset'}
CPU_CATEGORIES = {'cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver'}
PREFIX = 'recap_overhead::'


def summarize_events(events):
    ranges = sorted((e for e in events if e.get('cat') == 'user_annotation'
                     and e.get('ph') == 'X' and e['name'].startswith(PREFIX)),
                    key=lambda e: e['ts'])
    if not ranges:
        raise ValueError('missing CPU phase annotations')
    if any(a['ts'] + a['dur'] > b['ts'] + .01 for a, b in zip(ranges, ranges[1:])):
        raise ValueError('overlapping CPU phase annotations')
    starts = [e['ts'] for e in ranges]
    stages = defaultdict(lambda: dict(cpu_range_us=0.0, device_total_us=0.0,
        kernel_us=0.0, memcpy_us=0.0, memset_us=0.0, count=0, cuda_events=0))
    for event in ranges:
        row = stages[event['name'][len(PREFIX):]]
        row['cpu_range_us'] += event['dur']
        row['count'] += 1
    external, correlation = {}, {}
    for event in events:
        if event.get('cat') not in CPU_CATEGORIES or event.get('ph') != 'X':
            continue
        index = bisect_right(starts, event['ts']) - 1
        if index < 0 or event['ts'] >= ranges[index]['ts'] + ranges[index]['dur']:
            continue
        owner = ranges[index]['name'][len(PREFIX):]
        args = event.get('args', {})
        if 'External id' in args:
            external[args['External id']] = owner
        if event['cat'] in {'cuda_runtime', 'cuda_driver'} and 'correlation' in args:
            correlation[args['correlation']] = owner
    total_us = 0.0
    total_events = 0
    unassigned = []
    for event in events:
        category = event.get('cat')
        if category not in DEVICE_CATEGORIES or event.get('ph') != 'X':
            continue
        duration = event['dur']
        total_us += duration
        total_events += 1
        args = event.get('args', {})
        # Launch correlation avoids GPU/CPU timestamp-alignment artifacts near
        # synchronization boundaries. External IDs provide a second direct link.
        owner = correlation.get(args.get('correlation')) or external.get(args.get('External id'))
        if owner is None:
            unassigned.append(event)
            continue
        row = stages[owner]
        row['device_total_us'] += duration
        row['cuda_events'] += 1
        metric = {'kernel': 'kernel_us', 'gpu_memcpy': 'memcpy_us', 'gpu_memset': 'memset_us'}[category]
        row[metric] += duration
    if unassigned:
        raise ValueError(f'{len(unassigned)} CUDA events lack a phase owner')
    attributed = sum(row['device_total_us'] for row in stages.values())
    if abs(attributed - total_us) > max(.01, total_us * 1e-9):
        raise ValueError('CUDA duration accounting mismatch')
    return dict(stages), dict(cuda_events=total_events, device_total_us=total_us,
        unassigned_cuda_events=0,
        method='sum kernel/gpu_memcpy/gpu_memset durations, owned by CPU launch correlation or external ID; gpu_user_annotation excluded')


def summarize_trace(path):
    return summarize_events(json.loads(Path(path).read_text())['traceEvents'])
