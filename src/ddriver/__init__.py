"""
ddriver package holds reusable scripts, such as
data loading, models, training, eval.
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("ddriver")
except Exception:
    __version__ = "0.0.0"

__all__ = ["config", "data", "models", "train", "eval"]


