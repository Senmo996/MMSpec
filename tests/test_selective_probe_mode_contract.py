from method.sam_grounded.tree_recycling_model import SELECTIVE_REUSE_PROBE_MODES


def test_offline_content_ablation_mode_is_accepted_by_tree_model_contract():
    assert SELECTIVE_REUSE_PROBE_MODES == {
        "packed-attention",
        "teacher-forced-pixel",
        "teacher-forced-content-ablation",
    }
