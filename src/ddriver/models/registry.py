"""
Model registry utilities.

This keeps a mapping from a string name → callable that builds a torch.nn.Module.
You can register custom builders with the decorator or helper functions below.
"""

from __future__ import annotations

from typing import Callable, Dict, List

MODEL_REGISTRY: Dict[str, Callable[..., object]] = {}


def _normalize_name(name: str) -> str:
    """Helper to normalize model names (case-insensitive, trimmed)."""
    if not isinstance(name, str):
        raise TypeError("Model name must be a string")
    return name.strip().lower()


def register_model(name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """
    Decorator to register a model builder function.

    Example:
        @register_model("my_cool_model")
        def build_model(num_classes: int):
            ...
            return model
    """

    normalized_name = _normalize_name(name)

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        if normalized_name in MODEL_REGISTRY:
            raise ValueError(f"Model '{normalized_name}' is already registered")
        MODEL_REGISTRY[normalized_name] = fn
        return fn

    return decorator


def build_model(name: str, *args, **kwargs):
    """
    Instantiate a model by name.

    Args:
        name: String key used during registration.
        *args/**kwargs: Forwarded to the registered builder.
    """

    normalized_name = _normalize_name(name)
    if normalized_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[normalized_name](*args, **kwargs)


def available_models() -> List[str]:
    """List sorted names of registered models."""
    return sorted(MODEL_REGISTRY.keys())


def register_timm_backbone(model_name: str) -> None:
    """
    Convenience helper to register a timm backbone by name.

    Usage:
        register_timm_backbone("resnet18")
        model = build_model("resnet18", pretrained=True, num_classes=10)
    """

    normalized_name = _normalize_name(model_name)

    if normalized_name in MODEL_REGISTRY:
        return  # already registered, nothing to do

    try:
        import timm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "timm is required to register timm backbones. Install via `pip install timm`."
        ) from exc

    @register_model(normalized_name)
    def _builder(**kwargs):
        return timm.create_model(model_name, **kwargs)


__all__ = [
    "register_model",
    "register_timm_backbone",
    "build_model",
    "available_models",
    "MODEL_REGISTRY",
]

