from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base import EmbeddingResult, FoundationModelWrapper

if TYPE_CHECKING:
    pass

MODEL_REGISTRY: dict[str, type[FoundationModelWrapper]] = {}


def register_model(name: str):
    """Class decorator that registers a FoundationModelWrapper subclass."""

    def decorator(cls: type[FoundationModelWrapper]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(cfg: DictConfig) -> FoundationModelWrapper:
    """Factory: instantiate a registered model wrapper from Hydra config.

    Expects ``cfg.model.name`` to match a key in MODEL_REGISTRY.
    """
    model_name = cfg.model.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](cfg)


# Import submodules so @register_model decorators execute at import time.
from . import geneformer, scgpt  # noqa: E402, F401

__all__ = [
    "EmbeddingResult",
    "FoundationModelWrapper",
    "MODEL_REGISTRY",
    "get_model",
    "register_model",
]
