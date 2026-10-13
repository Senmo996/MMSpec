"""MMSpec baselines, with access to repository methods for legacy imports."""

from pathlib import Path

# Evaluators put MMSpec first on sys.path and import this as ``method``.
# Let that same package resolve the shared DREAM and FOVEA implementations.
if __name__ == "method":
    _repository_methods = Path(__file__).resolve().parents[2] / "method"
    if _repository_methods.is_dir():
        __path__.append(str(_repository_methods))
