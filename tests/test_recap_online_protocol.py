from unittest.mock import patch

from PIL import Image

from evaluation.build_disjoint_selective_manifests import _pixel_identity
from evaluation import eval_sam_grounded_fixed_multi as fixed_eval


def test_pixel_identity_tracks_content_and_geometry_not_image_mode():
    image = Image.new("RGB", (2, 3), (40, 50, 60))
    assert _pixel_identity(image) == _pixel_identity(image.convert("RGBA"))
    assert _pixel_identity(image) != _pixel_identity(Image.new("RGB", (3, 2), (40, 50, 60)))
    changed = image.copy()
    changed.putpixel((0, 0), (40, 50, 61))
    assert _pixel_identity(image) != _pixel_identity(changed)


def test_compact_timing_summary_distinguishes_instrumented_runs():
    args = fixed_eval.build_parser().parse_args([
        "--base-model-path", "unused", "--output-root", "unused",
        "--manifest-dir", "unused", "--no-policy-trace",
    ])
    args.datasets = ["ChartQA"]
    with patch.object(fixed_eval, "_write_json") as write:
        fixed_eval._write_run_summary("unused", {}, args.candidate_policy, args)
        config = write.call_args.args[1]["config"]
        assert not config["forward_timing_hooks_enabled"]
        assert config["diagnostic_timing_valid"]
        args.no_policy_trace = False
        fixed_eval._write_run_summary("unused", {}, args.candidate_policy, args)
        config = write.call_args.args[1]["config"]
        assert config["forward_timing_hooks_enabled"]
        assert not config["diagnostic_timing_valid"]
