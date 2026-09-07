import pytest

from evaluation.analyze_frozen_selective_conflict_regime import (
    add_visual_features,
    prepare,
)


def make_row(
    question_id,
    output_position,
    *,
    drop,
    full_logprob,
    jsd,
    iteration=0,
):
    return {
        "analysis_split": "all",
        "benchmark": "ScienceQA",
        "cluster_id": f"ScienceQA:{question_id}",
        "question_id": question_id,
        "choice_index": 0,
        "turn_index": 0,
        "iteration": iteration,
        "output_position": output_position,
        "visual_probe_max_target_logprob_drop": drop,
        "visual_probe_full_target_logprob": full_logprob,
        "visual_probe_jsd": jsd,
        "u_available": False,
        "gc_available": False,
        "u_root_candidate_token_ids": [],
        "gc_root_candidate_token_ids": [],
        "target_token_id": 7,
    }


def test_local2_visual_features_follow_each_trajectory_without_crossing_samples():
    rows = [
        make_row("a", 3, drop=0.0, full_logprob=-1.0, jsd=0.1, iteration=1),
        make_row("b", 1, drop=3.0, full_logprob=-1.0, jsd=0.9),
        make_row("a", 1, drop=1.0, full_logprob=-1.0, jsd=0.5),
    ]

    add_visual_features(rows)

    first_a = next(row for row in rows if row["question_id"] == "a" and row["output_position"] == 1)
    last_a = next(row for row in rows if row["question_id"] == "a" and row["output_position"] == 3)
    only_b = next(row for row in rows if row["question_id"] == "b")
    assert first_a["visual_target_drop_fraction"] == pytest.approx(0.5)
    assert first_a["visual_local2_mean_target_drop_fraction"] == pytest.approx(0.25)
    assert first_a["visual_local2_mean_jsd"] == pytest.approx(0.3)
    assert last_a["visual_local2_mean_target_drop_fraction"] == pytest.approx(0.0)
    assert only_b["visual_local2_mean_target_drop_fraction"] == pytest.approx(0.75)
    assert only_b["visual_local2_mean_jsd"] == pytest.approx(0.9)


def test_prepare_records_offline_visual_score_and_rejects_unknown_metric():
    rows = [
        make_row("a", 1, drop=1.0, full_logprob=-1.0, jsd=0.5),
        make_row("a", 2, drop=0.0, full_logprob=-1.0, jsd=0.1, iteration=1),
    ]

    prepare(rows, visual_metric="local2_mean_target_drop_fraction")

    assert rows[0]["frozen_visual_score"] == pytest.approx(0.25)
    assert rows[1]["frozen_visual_score"] == pytest.approx(0.0)
    assert all("frozen_visual_percentile" in row for row in rows)
    with pytest.raises(ValueError, match="unsupported visual metric"):
        prepare(rows, visual_metric="not-a-score")


def test_prepare_uses_exact_generated_token_span_metric_when_recorded():
    rows = [
        make_row("a", 1, drop=1.0, full_logprob=-1.0, jsd=0.5),
        make_row("b", 1, drop=0.0, full_logprob=-1.0, jsd=0.1),
    ]
    rows[0]["visual_probe_span2_mean_target_drop_fraction"] = 0.8
    rows[1]["visual_probe_span2_mean_target_drop_fraction"] = 0.2

    prepare(rows, visual_metric="span2_mean_target_drop_fraction")

    assert rows[0]["frozen_visual_score"] == pytest.approx(0.8)
    assert rows[1]["frozen_visual_score"] == pytest.approx(0.2)


def test_prepare_rejects_unrecorded_exact_span_metric():
    rows = [make_row("a", 1, drop=1.0, full_logprob=-1.0, jsd=0.5)]

    with pytest.raises(ValueError, match="is absent from 1/1 states"):
        prepare(rows, visual_metric="span2_mean_target_drop_fraction")
