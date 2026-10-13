"""Timing must preserve tree semantics and avoid double-counting memory/time."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.recap_overhead_utils import StageProfiler, object_footprint
from method.sam_grounded.tree_recycling_model import TreeRecyclingSpecModel as Model
import torch
import pytest


def test_instrumented_tree_is_identical():
    kwargs = dict(root_token=1, transitions=torch.empty(0,dtype=torch.long),
        transition_valid=torch.empty(0,dtype=torch.bool), width=2, depth=5,
        node_budget=8, blocked_token_id=-1, host_transitions={1:[2,3],2:[3,4],3:[4,2]},
        host_transition_scores={1:[.7,.3],2:[.6,.4],3:[.8,.2]},
        host_context_transitions={(1,2):[4,3]}, host_context_transition_scores={(1,2):[.75,.25]},
        context_candidate_mode='fusion', score_priority_layout=True, reserved_path_tokens=[4,3])
    a = Model._build_tree(**kwargs)
    p = StageProfiler(synchronize=lambda: None)
    with p.region('tree_search_fusion'):
        b = Model._build_tree(**kwargs, build_profiler=p)
    p.finish()
    assert a[0] == b[0] and a[3] == b[3]
    assert torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
    assert p.rows['tree_search_fusion']['wall_seconds'] > 0
    assert p.rows['topology_packing']['wall_seconds'] > 0


def test_exclusive_stages_restore_parent():
    p=StageProfiler(synchronize=lambda: None)
    start=time.perf_counter()
    p.switch('outer')
    with p.region('child'):
        p.switch('child_second_part')
        sum(i*i for i in range(10000))
    assert p.active == 'outer'
    p.finish()
    assert sum(r['wall_seconds'] for r in p.rows.values()) <= time.perf_counter()-start


def test_footprint_deduplicates_shared_rows_and_storage():
    row=[1000,1001,1002]
    a=object_footprint([row,row])['host_object_bytes']
    b=object_footprint([row,list(row)])['host_object_bytes']
    assert b-a == sys.getsizeof(row)
    tensor=torch.ones(100)
    shared=object_footprint([tensor,tensor[:50]])
    assert shared['cpu_tensor_storage_bytes'] == 400
    assert shared['cuda_tensor_storage_bytes'] == 0


@pytest.mark.parametrize('competing', [False, True])
def test_gpu_activity_rejects_concurrent_compute(tmp_path, competing):
    from evaluation.summarize_recap_overhead import check_gpu_activity
    rows=[]
    for second in range(10,20):
        rows.append(f'20260910 15:00:{second} 2 100 C 75 20 - - - - python')
        utilization='5' if competing else '-'
        rows.append(f'20260910 15:00:{second} 2 200 C {utilization} - - - - - idle_model')
    # Other activity before this run must not invalidate it.
    rows.append('20260910 14:00:00 2 200 C 99 20 - - - - python')
    path=tmp_path/'activity.log'
    path.write_text('\n'.join(rows))
    protocol={'process_id':100,'started_at':'2026-09-10T15:00:00+0800'}
    summary={'finished_at':'2026-09-10T15:00:30+0800'}
    if competing:
        with pytest.raises(AssertionError,match='concurrent GPU compute'):
            check_gpu_activity(path,protocol,summary)
    else:
        result=check_gpu_activity(path,protocol,summary)
        assert result['own_active_seconds']==10
        assert result['competing_active_samples']==0


@pytest.mark.parametrize('unassigned', [False, True])
def test_cuda_event_sum_excludes_annotation_spans(unassigned):
    from evaluation.recap_overhead_trace import summarize_events
    events=[
        dict(ph='X',cat='user_annotation',name='recap_overhead::packing',ts=10,dur=100),
        dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',ts=20,dur=1,
             args={'correlation':9}),
        # GPU timestamps may cross a CPU boundary. The launch identifies ownership.
        dict(ph='X',cat='kernel',name='kernel',ts=109,dur=3,args={'correlation':9}),
        dict(ph='X',cat='gpu_user_annotation',name='recap_overhead::packing',ts=20,dur=90),
    ]
    if unassigned:
        events.append(dict(ph='X',cat='kernel',name='unknown',ts=50,dur=2,args={}))
        with pytest.raises(ValueError,match='lack a phase owner'):
            summarize_events(events)
    else:
        stages,accounting=summarize_events(events)
        assert stages['packing']['device_total_us']==3
        assert accounting['cuda_events']==1
        assert accounting['unassigned_cuda_events']==0
