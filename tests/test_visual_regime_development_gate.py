from evaluation.evaluate_visual_regime_development_gate import (
    EXPECTED_BENCHMARKS,
    EXPECTED_PROTOCOL,
    EXPECTED_VISUAL_METRIC,
    evaluate,
)


def valid_summary():
    by_benchmark = {}
    for index, name in enumerate(sorted(EXPECTED_BENCHMARKS)):
        by_benchmark[name] = {
            "low_eligible_states": 4,
            "high_eligible_states": 4,
            "low_delta": -0.2 if index < 4 else 0.0,
            "high_delta": 0.2,
            "interaction": 0.4 if index < 4 else 0.2,
        }
    return {
        "analysis_role": "development",
        "analysis_split": "all",
        "benchmarks": sorted(EXPECTED_BENCHMARKS),
        "excluded_benchmarks": ["MME"],
        "visual_probe_protocols": [EXPECTED_PROTOCOL],
        "frozen_rule": {
            "visual_metric": EXPECTED_VISUAL_METRIC,
            "uses_future_generated_token": True,
            "online_routing_feature": False,
        },
        "probe_audit": {
            "same_text_trajectory_ratio": 1.0,
            "recomputed_vision_encoder_ratio": 1.0,
        },
        "point_estimates": {
            "low_delta": -0.1,
            "high_delta": 0.2,
            "interaction": 0.3,
        },
        "by_benchmark": by_benchmark,
    }


def test_development_gate_launches_only_on_broad_point_crossover():
    result = evaluate(valid_summary())

    assert result["decision"] == "go"
    assert result["num_valid_benchmarks"] == 8
    assert result["positive_interaction_benchmarks"] == 8
    assert result["crossover_benchmarks"] == 4
    assert result["failures"] == []


def test_development_gate_rejects_nonnegative_low_tail():
    summary = valid_summary()
    summary["point_estimates"]["low_delta"] = 0.01

    result = evaluate(summary)

    assert result["decision"] == "no_go"
    assert "development low tail does not favor U" in result["failures"]


def test_development_gate_rejects_wrong_metric_and_sparse_tails():
    summary = valid_summary()
    summary["frozen_rule"]["visual_metric"] = "visual_target_drop_fraction"
    for row in summary["by_benchmark"].values():
        row["low_eligible_states"] = 1

    result = evaluate(summary)

    assert result["decision"] == "no_go"
    assert "exact two-token visual metric mismatch" in result["failures"]
    assert "low tail has only 8 eligible states" in result["failures"]

