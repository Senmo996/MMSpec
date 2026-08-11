"""Lazy loader for the optional C++ GWTR score-priority tree builder."""

import os
from pathlib import Path


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("fast_tree_builder.cpp")
    project_root = source.parents[2]
    build_root = Path(
        os.environ.get(
            "GWTR_TREE_BUILDER_CACHE",
            str(project_root.parent / "outputs" / "cache" / "gwtr_tree_builder"),
        )
    )
    build_root.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name="gwtr_tree_builder_v1",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_root),
        verbose=False,
    )
    return _EXTENSION


def build_score_priority_nodes(**kwargs):
    """Return ``(token, parent, depth, rank)`` rows from the C++ builder."""

    extension = _load_extension()
    return extension.build_score_priority_nodes(**kwargs)
