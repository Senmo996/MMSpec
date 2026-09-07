from evaluation.audit_selective_validation_run import content_span_audit


def wrap_trace(trace):
    return [{"choices": [{"policy_trace": [[trace]]}]}]


def valid_trace(span_tokens=2):
    return {
        "visual_probe_protocol": "teacher_forced_whole_image_mean_ablation_v1",
        "visual_probe_span2_num_tokens": span_tokens,
        "visual_probe_span2_mean_jsd": 0.2,
        "visual_probe_span2_mean_target_logprob_drop": 0.3,
        "visual_probe_span2_mean_target_drop_fraction": 0.4,
        "visual_probe_span2_same_text_trajectory": True,
        "visual_probe_span2_uses_future_tokens": span_tokens == 2,
    }


def test_content_span_audit_accepts_two_token_and_terminal_one_token_states():
    rows = [
        {
            "choices": [
                {
                    "policy_trace": [
                        [valid_trace(2), valid_trace(1)],
                    ]
                }
            ]
        }
    ]

    report = content_span_audit(rows)

    assert report["protocol_states"] == 2
    assert report["valid_span_states"] == 2
    assert report["two_token_states"] == 1
    assert report["one_token_terminal_states"] == 1
    assert report["missing_or_invalid_span_states"] == 0


def test_content_span_audit_rejects_inconsistent_future_flag():
    trace = valid_trace(1)
    trace["visual_probe_span2_uses_future_tokens"] = True

    report = content_span_audit(wrap_trace(trace))

    assert report["valid_span_states"] == 0
    assert report["missing_or_invalid_span_states"] == 1
