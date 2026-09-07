from evaluation.evaluate_continuous_visual_regime_development_gate import (
    EXPECTED_BENCHMARKS,
    EXPECTED_BOOTSTRAP_RESAMPLES,
    EXPECTED_BOOTSTRAP_SEED,
    EXPECTED_PROTOCOL,
    EXPECTED_REFERENCE_POPULATION,
    EXPECTED_VISUAL_METRIC,
    evaluate,
)


def valid_summary():
    support = {name: 8 for name in EXPECTED_BENCHMARKS}
    fits = {
        name: {
            "delta_slope": 0.2,
            "delta_p10": -0.1,
            "delta_p90": 0.1,
        }
        for name in EXPECTED_BENCHMARKS
    }
    return {
        "analysis_role": "development",
        "analysis_split": "all",
        "benchmarks": sorted(EXPECTED_BENCHMARKS),
        "excluded_benchmarks": ["MME"],
        "visual_probe_protocols": [EXPECTED_PROTOCOL],
        "probe_audit": {
            "same_text_trajectory_ratio": 1.0,
            "recomputed_vision_encoder_ratio": 1.0,
        },
        "frozen_rule": {
            "visual_metric": EXPECTED_VISUAL_METRIC,
            "visual_percentile_reference_population": (
                EXPECTED_REFERENCE_POPULATION
            ),
        },
        "visual_percentile_audit": {
            "reference_population": EXPECTED_REFERENCE_POPULATION,
            "uses_source_hit_outcomes": False,
            "states_by_benchmark": support,
            "num_reference_states": sum(support.values()),
        },
        "continuous_point_estimates": {
            "delta_slope": 0.2,
            "delta_p10": -0.1,
            "delta_p90": 0.1,
        },
        "continuous_by_benchmark": fits,
        "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
    }


def test_gate_accepts_complete_directional_development_crossover():
    result = evaluate(valid_summary())

    assert result["decision"] == "go"
    assert result["positive_slope_benchmarks"] == 8
    assert result["point_crossover_benchmarks"] == 8
    assert result["failures"] == []


def test_gate_rejects_nonnegative_low_endpoint():
    summary = valid_summary()
    summary["continuous_point_estimates"]["delta_p10"] = 0.01

    result = evaluate(summary)

    assert result["decision"] == "no_go"
    assert (
        "development percentile-0.10 endpoint does not favor U"
        in result["failures"]
    )


def test_gate_rejects_outcome_based_ranks_and_missing_benchmark():
    summary = valid_summary()
    summary["visual_percentile_audit"]["uses_source_hit_outcomes"] = True
    summary["continuous_by_benchmark"].pop("OCRBench")

    result = evaluate(summary)

    assert result["decision"] == "no_go"
    assert "visual percentile assignment used a source-hit outcome" in result["failures"]
    assert "continuous fit is missing one or more benchmarks" in result["failures"]


def test_gate_rejects_wrong_probe_and_bootstrap_contract():
    summary = valid_summary()
    summary["visual_probe_protocols"] = ["teacher_forced_mean_patch_occlusion_v1"]
    summary["bootstrap_resamples"] = 10

    result = evaluate(summary)

    assert result["decision"] == "no_go"
    assert "whole-image content-ablation protocol mismatch" in result["failures"]
    assert "development bootstrap resample count mismatch" in result["failures"]


def test_gate_can_predeclare_span2_jsd_fallback():
    summary = valid_summary()
    summary["frozen_rule"]["visual_metric"] = "span2_mean_jsd"

    result = evaluate(summary, expected_visual_metric="span2_mean_jsd")

    assert result["decision"] == "go"
    assert result["expected_visual_metric"] == "span2_mean_jsd"
