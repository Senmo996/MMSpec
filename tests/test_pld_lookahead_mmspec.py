"""Check that paired PLD/Lookahead runs preserve the native multi-turn protocol."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from evaluation import eval_sam_grounded_mmspec as evaluation


@pytest.mark.parametrize("engine", ["pld", "lookahead"])
def test_paired_multiturn_history_and_compact_output(engine, monkeypatch, tmp_path):
    base_model = object()
    calls, prompts = [], []

    class Candidate:
        def __init__(self, *args, **kwargs):
            self.base_model = base_model
            self.lookahead_cache = dict(warmup_only=True)

        @classmethod
        def from_pretrained(cls, **kwargs):
            assert kwargs["attn_implementation"] == "eager"
            return cls()

        def get_tokenizer(self):
            return None

        def eval(self):
            return self

        def specgenerate(self, input_ids, **kwargs):
            assert kwargs["return_policy_trace"] is False
            if isinstance(self, Target):
                assert kwargs["draft_policy"] == "target"
                policy, token = "target", 80
            else:
                policy, token = engine, 90
                if engine == "pld":
                    assert (kwargs["pld_ngram"], kwargs["pld_n_pred"]) == (4, 10)
                else:
                    assert (kwargs["decoding_length"], kwargs["branch_length"]) == (64, 12)
                    assert "warmup_only" not in self.lookahead_cache
            calls.append(policy)
            return torch.cat([input_ids, torch.tensor([[token, token]])], dim=1), 2, 1, [0, 0]

    class Target(Candidate):
        def __init__(self, shared_model, *args, **kwargs):
            assert shared_model is base_model
            super().__init__()

    def build_prompt(sample, args, turn_idx=0, conversation_history=None):
        prompts.append((turn_idx, list(conversation_history or [])))
        return {"input_ids": torch.tensor([[1, turn_idx + 2]])}

    tracker = SimpleNamespace(
        reset=lambda: None, snapshot=lambda: (0.0, 0.0),
        install_phase_fallback_hooks=lambda model: None,
    )
    monkeypatch.setattr(evaluation, "PLDSpecModel", Candidate)
    monkeypatch.setattr(evaluation, "LookaheadSpecModel", Candidate)
    monkeypatch.setattr(evaluation, "TreeRecyclingSpecModel", Target)
    monkeypatch.setattr(evaluation, "build_time_breakdown_tracker", lambda model: tracker)
    monkeypatch.setattr(evaluation, "load_mmspec_data", lambda folder: [
        {"id": "example", "topic": "multi-turn conversation", "turns": ["q1", "q2", "q3"]}
    ])
    monkeypatch.setattr(evaluation, "build_prompt", build_prompt)
    monkeypatch.setattr(evaluation, "process_output", lambda ids, tokenizer, length: str(ids[0, length].item()))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "eval", "--base-model-path", "fake-model", "--data-folder", "unused",
        "--output-root", str(tmp_path), "--draft-engine", engine,
        "--policies", f"target,{engine}", "--attn-implementation", "eager",
        "--warmup-runs", "0", "--max-new-token", "2",
    ])
    evaluation.main()
    assert calls == ["target"] * 3 + [engine] * 3
    assert prompts == [
        (0, []), (1, [("q1", "80")]), (2, [("q1", "80"), ("q2", "80")]),
        (0, []), (1, [("q1", "90")]), (2, [("q1", "90"), ("q2", "90")]),
    ]
    for policy in ("target", engine):
        record = json.loads((tmp_path / policy / "results.jsonl").read_text())
        choice = record["choices"][0]
        assert choice["new_tokens"] == [2, 2, 2]
        assert len(choice["wall_time"]) == len(choice["output_hashes"]) == 3
        assert "turns" not in choice and "policy_trace" not in choice
