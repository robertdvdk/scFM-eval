"""Geneformer foundation model wrapper (stub).

Geneformer uses rank-value encoding — genes are ranked by their expression
relative to median expression across a reference corpus. The model is a
HuggingFace BertForMaskedLM under the hood, and cell embeddings are
typically extracted from the CLS token.

The heavy methods in this stub raise NotImplementedError — they will be
filled in once the Geneformer package is installed and integrated.
"""

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from . import register_model
from .base import EmbeddingResult, FoundationModelWrapper


@register_model("geneformer")
class GeneformerWrapper(FoundationModelWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._embedding_dim: int = 0  # Set after loading pretrained model
        self._gene_vocab: list[str] = []

    # ------------------------------------------------------------------
    # Abstract method implementations (stubs)
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        raise NotImplementedError(
            "GeneformerWrapper.load_pretrained() requires the geneformer package. "
            "Install it and implement model loading from "
            f"cfg.model.pretrained_path={self.cfg.model.pretrained_path}"
        )

    def tokenize(self, adata) -> dict[str, Tensor]:
        raise NotImplementedError(
            "GeneformerWrapper.tokenize() requires the geneformer package. "
            "Tokenization: rank genes by expression/median ratio, encode as token IDs."
        )

    def create_dataset(self, adata) -> Dataset:
        raise NotImplementedError("GeneformerWrapper.create_dataset() requires the geneformer package.")

    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        raise NotImplementedError("GeneformerWrapper.embed() requires the geneformer package.")

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("GeneformerWrapper.forward() requires the geneformer package.")

    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "GeneformerWrapper.compute_native_loss() requires the geneformer package. "
            "Native objective: masked language model loss."
        )

    @property
    def name(self) -> str:
        return "Geneformer"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def gene_vocabulary(self) -> list[str]:
        return self._gene_vocab
