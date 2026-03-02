from __future__ import annotations

import importlib
import logging
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
# Conditional imports: model packages may not be installed in the main env
# when using subprocess-based isolation. Gracefully skip unavailable models.
_log = logging.getLogger(__name__)

for _mod_name in ["scgpt", "geneformer", "cancerfoundation", "scfoundation"]:
    try:
        importlib.import_module(f".{_mod_name}", package=__name__)
    except ImportError:
        _log.debug("Skipping model '%s': dependencies not installed", _mod_name)
    except Exception:
        _log.warning("Failed to import model '%s'", _mod_name, exc_info=True)

__all__ = [
    "EmbeddingResult",
    "FoundationModelWrapper",
    "MODEL_REGISTRY",
    "get_model",
    "register_model",
]
