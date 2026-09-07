from evaluation.audit_candidate_alignment_validation import (
    CANDIDATE_NUMERIC_FIELDS,
    audit_candidate_fields,
)
from evaluation.audit_consensus_counterfactual_run import EXPECTED_PROTOCOL


def _result(*, uses_target_outcome: bool = False) -> dict:
    trace = {
        "selective_reuse_diagnostics": True,
        "visual_probe_candidate_alignment_available": True,
        "visual_probe_candidate_alignment_budget": 2,
        "visual_probe_candidate_alignment_uses_target_outcome": (
            uses_target_outcome
        ),
        "visual_probe_u_candidate_consensus_support": 0.25,
        "visual_probe_gc_candidate_consensus_support": 0.75,
        "visual_probe_gc_minus_u_candidate_visual_support": 0.50,
    }
    for field in CANDIDATE_NUMERIC_FIELDS:
        trace.setdefault(field, -0.5)
    return {
        "choices": [
            {
                "selective_probe_metadata": [
                    {
                        "protocol": EXPECTED_PROTOCOL,
                        "candidate_set_visual_alignment": True,
                        "candidate_set_visual_alignment_uses_target_outcome": False,
                    }
                ],
                "policy_trace": [[trace]],
            }
        ]
    }


def test_candidate_alignment_validation_audit_accepts_consistent_fields():
    report = audit_candidate_fields([_result()])

    assert report["metadata_records"] == 1
    assert report["valid_metadata_records"] == 1
    assert report["diagnostic_states"] == 1
    assert report["states_with_candidate_flag"] == 1
    assert report["available_candidate_states"] == 1
    assert report["valid_available_candidate_states"] == 1
    assert report["states_using_target_outcome"] == 0


def test_candidate_alignment_validation_audit_flags_target_outcome_use():
    report = audit_candidate_fields([_result(uses_target_outcome=True)])

    assert report["states_using_target_outcome"] == 1
