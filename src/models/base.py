from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import Dataset


@dataclass
class EmbeddingResult:
    """Container for foundation model embedding outputs."""

    cell_embeddings: np.ndarray  # (n_cells, embedding_dim)
    gene_embeddings: np.ndarray | None
    embedding_dim: int


class FoundationModelWrapper(ABC):
    """Abstract base class for single-cell foundation model wrappers.

    Wraps (but does NOT subclass) an nn.Module. Provides a uniform interface
    for tokenization, embedding extraction, and differentiable forward passes
    across different foundation model architectures.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._model: nn.Module | None = None
        self._device: torch.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """Return the underlying nn.Module. Raises if not loaded."""
        if self._model is None:
            raise RuntimeError(f"{self.name}: model not loaded. Call load_pretrained() first.")
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str | torch.device) -> "FoundationModelWrapper":
        """Move the underlying model to *device* and return self."""
        self._device = torch.device(device)
        if self._model is not None:
            self._model = self._model.to(self._device)
        return self

    # ------------------------------------------------------------------
    # Abstract – must be implemented per model
    # ------------------------------------------------------------------

    @abstractmethod
    def load_pretrained(self) -> None:
        """Load checkpoint and vocabulary from paths specified in config."""

    @abstractmethod
    def tokenize(self, adata) -> dict[str, Tensor]:
        """Convert an AnnData object into model-ready tensors.

        Returns a dict of tensors (e.g. gene_ids, expression_values, padding_mask).
        """

    @abstractmethod
    def create_dataset(self, adata) -> Dataset:
        """Wrap tokenize() output as a per-cell PyTorch Dataset."""

    @abstractmethod
    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        """High-level: AnnData in -> embeddings out.

        Handles batching and torch.no_grad() internally.
        """

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Low-level differentiable forward: tokenized batch -> cell embeddings.

        For use in custom finetuning loops where the caller computes the loss.
        """

    @abstractmethod
    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return (native_loss, cell_embeddings) for the model's pretraining objective.

        E.g. masked language model loss for scGPT / Geneformer.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name, e.g. 'scGPT'."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of cell embeddings."""

    @property
    @abstractmethod
    def gene_vocabulary(self) -> list[str]:
        """List of gene names this model knows about."""
