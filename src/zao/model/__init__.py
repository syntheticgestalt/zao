"""The network, its architecture and its weights.

``ZAOModel`` is loaded on first use, so reading ``ZAO_CONFIG`` or
``DEFAULT_WEIGHTS`` does not import PyTorch.
"""

from .config import DEFAULT_WEIGHTS, ZAO_CONFIG
from .weights import load_model, resolve_weights

__all__ = ["DEFAULT_WEIGHTS", "ZAO_CONFIG", "ZAOModel", "load_model", "resolve_weights"]


def __getattr__(name: str):
    if name == "ZAOModel":
        from .zao import ZAOModel

        return ZAOModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(__all__)
