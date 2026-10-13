"""Explicit diagnostic timing and memory accounting for RECAP overhead studies.

Stage wall times are synchronized and mutually exclusive. They are not CUDA
kernel times: optional Kineto traces supply those separately. No hooks are
installed outside the caller's context, and the normal method is unchanged.
"""

from collections import defaultdict
from contextlib import contextmanager
import functools
import math
import os
import statistics
import sys
import time

import torch


class StageProfiler:
    def __init__(self, synchronize=None, annotate=False):
        self.synchronize = synchronize or torch.cuda.synchronize
        self.annotate = annotate
        self.active = None
        self.annotation = None
        self.in_decode = False
        self.forward_calls = 0
        self.memory_objects = {}
        self.rows = defaultdict(lambda: dict(wall_seconds=0.0, process_cpu_seconds=0.0, segments=0))

    def _stop(self):
        if self.active is None:
            return
        self.synchronize()
        wall = time.perf_counter() - self.wall_start
        cpu = time.process_time() - self.cpu_start
        row = self.rows[self.active]
        row['wall_seconds'] += wall
        row['process_cpu_seconds'] += cpu
        row['segments'] += 1
        if self.annotation is not None:
            self.annotation.__exit__(None, None, None)
            self.annotation = None
        self.active = None

    def switch(self, name):
        self._stop()
        if name is None:
            return
        if self.annotate:
            self.annotation = torch.profiler.record_function('recap_overhead::' + name)
            self.annotation.__enter__()
        self.active = name
        self.cpu_start = time.process_time()
        self.wall_start = time.perf_counter()

    @contextmanager
    def region(self, name):
        parent = self.active
        self.switch(name)
        try:
            yield
        finally:
            self.switch(parent)

    def wrap_updates(self, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            name = 'memory_update' if self.in_decode else 'memory_initialization'
            with self.region(name):
                return function(*args, **kwargs)
        return wrapped

    def finish(self):
        self.switch(None)
        return dict(self.rows)


@contextmanager
def instrument_model(model, profiler):
    """Install bounded, process-local wrappers and restore them on every exit."""
    from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel
    restorations = []

    def install(owner, attribute, value):
        own = attribute in vars(owner)
        original = vars(owner).get(attribute)
        setattr(owner, attribute, value)
        restorations.append((owner, attribute, own, original))

    original_forward = model.base_model.forward

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        name = 'prefill' if profiler.forward_calls == 0 else 'target_verification'
        profiler.forward_calls += 1
        with profiler.region(name):
            return original_forward(*args, **kwargs)

    original_build = TreeRecyclingSpecModel._build_tree

    @functools.wraps(original_build)
    def build(*args, **kwargs):
        kwargs['build_profiler'] = profiler
        with profiler.region('tree_search_fusion'):
            return original_build(*args, **kwargs)

    def wrap(owner, attribute, label):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            name = label
            if label == 'suffix_index' and not profiler.in_decode:
                name = 'suffix_initialization'
            with profiler.region(name):
                return original(*args, **kwargs)
        install(owner, attribute, wrapped)

    try:
        install(model.base_model, 'forward', forward)
        install(TreeRecyclingSpecModel, '_build_tree', staticmethod(build))
        for name in ('visual', 'vision_tower'):
            visual = getattr(model.base_model, name, None)
            if isinstance(visual, torch.nn.Module):
                wrap(visual, 'forward', 'vision_encoding')
        for name in ('lookup_continuation_tokens', 'lookup_tokens', 'lookup'):
            if hasattr(model.draft, name):
                wrap(model.draft, name, 'suffix_lookup')
        wrap(model.draft, 'update_tokens', 'suffix_index')
        yield
    finally:
        profiler.finish()
        for owner, attribute, own, original in reversed(restorations):
            if own:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)


def object_footprint(value):
    """Owned reachable object bytes, deduplicating shared objects and storage."""
    seen, storage_seen = set(), set()
    totals = dict(host_object_bytes=0, cpu_tensor_storage_bytes=0, cuda_tensor_storage_bytes=0)

    def visit(item):
        if id(item) in seen or item is None:
            return
        seen.add(id(item))
        totals['host_object_bytes'] += sys.getsizeof(item)
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr())
            if key not in storage_seen:
                storage_seen.add(key)
                metric = 'cuda_tensor_storage_bytes' if item.is_cuda else 'cpu_tensor_storage_bytes'
                totals[metric] += storage.nbytes()
        elif isinstance(item, dict):
            for k, v in item.items():
                visit(k)
                visit(v)
        elif isinstance(item, (list, tuple, set, frozenset)):
            for v in item:
                visit(v)
        elif hasattr(item, '__dict__') and not isinstance(item, (type, torch.nn.Module)):
            visit(vars(item))
        elif hasattr(type(item), '__slots__'):
            for slot in type(item).__slots__:
                if hasattr(item, slot):
                    visit(getattr(item, slot))
    visit(value)
    return totals


def rss_bytes():
    with open('/proc/self/statm') as handle:
        resident = int(handle.read().split()[1])
    return resident * os.sysconf('SC_PAGE_SIZE')


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    left = math.floor(position)
    right = math.ceil(position)
    return values[left] * (right - position) + values[right] * (position - left) if left != right else values[left]


def distribution(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.fmean(values) if values else None,
                median=statistics.median(values) if values else None,
                p90=quantile(values, .9), p95=quantile(values, .95),
                maximum=max(values) if values else None)


def kernel_stage_summary(trace_path):
    """Sum CUDA events; GPU annotation spans also contain launch/wait gaps."""
    from evaluation.recap_overhead_trace import summarize_trace
    stages, _ = summarize_trace(trace_path)
    return stages
