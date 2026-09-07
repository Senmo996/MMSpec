import json

from evaluation.analyze_policy_pairs import compare


def _write_result(path, question_id, new_tokens, wall_time):
    row = {
        "question_id": question_id,
        "topic": "test",
        "choices": [
            {
                "index": 0,
                "output_hashes": [question_id],
                "new_tokens": [new_tokens],
                "wall_time": [wall_time],
            }
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_compare_skips_zero_token_reference(tmp_path):
    policy_path = tmp_path / "fixed.jsonl"
    reference_path = tmp_path / "target.jsonl"
    _write_result(reference_path, "zero", 0, 1.0)
    _write_result(policy_path, "zero", 1, 1.0)
    _write_result(reference_path, "valid", 2, 2.0)
    _write_result(policy_path, "valid", 2, 1.0)

    result = compare(
        policy_path,
        reference_path,
        bootstrap_samples=10,
    )

    assert result["num_paired_samples"] == 1
    assert result["mean_speed_ratio"] == 2.0
