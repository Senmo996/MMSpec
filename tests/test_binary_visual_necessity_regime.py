from evaluation.analyze_binary_visual_necessity_regime import (
    assign_binary_label,
    clustered_bootstrap,
    summarize_binary_benchmark,
)


def row(*, changed, eligible=True, u_hit=0, gc_hit=0):
    return {
        "visual_probe_full_top1_matches_target": True,
        "visual_probe_top1_disagreement_rate": float(changed),
        "frozen_eligible": eligible,
        "u_available": True,
        "gc_available": True,
        "frozen_u_hit": u_hit,
        "frozen_gc_hit": gc_hit,
        "frozen_delta": gc_hit - u_hit,
    }


def test_binary_label_is_counterfactual_top1_change():
    rows = [row(changed=0), row(changed=1)]

    audit = assign_binary_label(rows)

    assert [item["binary_visual_label"] for item in rows] == ["low", "high"]
    assert audit["low_visual_states"] == 1
    assert audit["high_visual_states"] == 1


def test_summary_recovers_directional_crossover():
    rows = [
        row(changed=0, u_hit=1, gc_hit=0),
        row(changed=0, u_hit=1, gc_hit=0),
        row(changed=1, u_hit=0, gc_hit=1),
        row(changed=1, u_hit=0, gc_hit=1),
    ]
    assign_binary_label(rows)

    result = summarize_binary_benchmark(rows, minimum_stratum_states=2)

    assert result is not None
    assert result["low_delta"] == -1.0
    assert result["high_delta"] == 1.0
    assert result["interaction"] == 2.0


def test_full_view_mismatch_is_not_labeled():
    rows = [row(changed=0)]
    rows[0]["visual_probe_full_top1_matches_target"] = False

    audit = assign_binary_label(rows)

    assert rows[0]["binary_visual_label"] is None
    assert audit["full_view_top1_mismatch"] == 1


def test_development_bootstrap_can_report_sparse_draws():
    rows = []
    for benchmark in (
        "ChartQA",
        "MMSpec",
        "MMT-Bench",
        "MathVista",
        "OCRBench",
        "SEEDBench",
        "ScienceQA",
        "TextVQA",
    ):
        for index, changed in enumerate((0, 0, 0, 0, 0, 1, 1, 1, 1, 1)):
            current = row(changed=changed, u_hit=1 - changed, gc_hit=changed)
            current.update(
                {
                    "benchmark": benchmark,
                    "cluster_id": f"{benchmark}:{index}",
                }
            )
            rows.append(current)
    assign_binary_label(rows)

    intervals, valid_draws = clustered_bootstrap(
        rows,
        resamples=200,
        seed=7,
        minimum_valid_fraction=0.0,
    )

    assert valid_draws >= 100
    assert intervals["low_delta"][1] < 0.0
    assert intervals["high_delta"][0] > 0.0
