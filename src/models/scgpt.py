"""scGPT foundation model wrapper (stub).

scGPT uses masked gene expression prediction as its pretraining objective.
Tokenization maps gene names to vocabulary indices and bins expression values.
Cell embeddings are extracted from the CLS token.

The heavy methods in this stub raise NotImplementedError — they will be
filled in once the scGPT package is installed and integrated.
"""

from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from . import register_model
from .base import EmbeddingResult, FoundationModelWrapper


@register_model("scgpt")
class ScGPTWrapper(FoundationModelWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._embedding_dim: int = cfg.model.d_model
        self._gene_vocab: list[str] = []

    # ------------------------------------------------------------------
    # Abstract method implementations (stubs)
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        raise NotImplementedError(
            "ScGPTWrapper.load_pretrained() requires the scgpt package. "
            "Install it and implement model/vocab loading from "
            f"cfg.model.pretrained_path={self.cfg.model.pretrained_path}"
        )

    def tokenize(self, adata) -> dict[str, Tensor]:
        raise NotImplementedError(
            "ScGPTWrapper.tokenize() requires the scgpt package. "
            "Tokenization maps gene names -> vocab indices and bins expression values."
        )

    def create_dataset(self, adata) -> Dataset:
        raise NotImplementedError("ScGPTWrapper.create_dataset() requires the scgpt package.")

    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        raise NotImplementedError("ScGPTWrapper.embed() requires the scgpt package.")

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("ScGPTWrapper.forward() requires the scgpt package.")

    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "ScGPTWrapper.compute_native_loss() requires the scgpt package. "
            "Native objective: masked gene expression prediction."
        )

    @property
    def name(self) -> str:
        return "scGPT"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def gene_vocabulary(self) -> list[str]:
        return self._gene_vocab
