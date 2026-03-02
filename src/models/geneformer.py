"""Geneformer V2 foundation model wrapper.

Geneformer uses rank-value encoding — genes are ranked by their expression
relative to median expression across a reference corpus. The model is a
HuggingFace BertForMaskedLM under the hood, and cell embeddings are
typically extracted from the CLS token.

This wrapper delegates to Geneformer's ``TranscriptomeTokenizer`` and
``EmbExtractor`` for zero-shot embedding extraction. The heavy lifting
(tokenization + inference) is file-based: adata is written to disk,
tokenized into a HuggingFace Dataset, and then fed through the model.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from . import register_model
from .base import EmbeddingResult, FoundationModelWrapper

log = logging.getLogger(__name__)


@register_model("geneformer")
class GeneformerWrapper(FoundationModelWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._embedding_dim: int = 0
        self._gene_vocab: list[str] = []
        self._token_dict: dict[str, int] | None = None
        self._gene_median_dict: dict[str, float] | None = None
        self._gene_name_id_dict: dict[str, str] | None = None  # symbol -> Ensembl

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dict(cfg_path: str | None, bundled_filename: str) -> dict:
        """Load a pickle dictionary from config path or geneformer's bundled default."""
        if cfg_path is not None:
            path = Path(cfg_path)
        else:
            import geneformer

            path = Path(geneformer.__file__).parent / bundled_filename
        if not path.is_file():
            raise FileNotFoundError(f"Dictionary not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        from transformers import AutoConfig

        cfg = self.cfg

        # Load bundled dictionaries
        self._token_dict = self._load_dict(
            cfg.model.get("token_dictionary_path"),
            "token_dictionary_gc104M.pkl",
        )
        self._gene_median_dict = self._load_dict(
            cfg.model.get("median_dictionary_path"),
            "gene_median_dictionary_gc104M.pkl",
        )
        self._gene_name_id_dict = self._load_dict(
            cfg.model.get("gene_name_id_dict_path"),
            "gene_name_id_dict_gc104M.pkl",
        )

        # Get embedding dim from model config (does not download full weights)
        model_config = AutoConfig.from_pretrained(cfg.model.pretrained_path)
        self._embedding_dim = model_config.hidden_size

        # Gene vocabulary: Ensembl IDs in the token dictionary (skip special tokens)
        special_tokens = {"<pad>", "<mask>", "<cls>", "<eos>"}
        self._gene_vocab = [k for k in self._token_dict if k not in special_tokens]

        log.info(
            "Loaded Geneformer dictionaries: token_dict=%d, gene_median=%d, gene_name_id=%d, embedding_dim=%d",
            len(self._token_dict),
            len(self._gene_median_dict),
            len(self._gene_name_id_dict),
            self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        from geneformer import EmbExtractor, TranscriptomeTokenizer

        cfg = self.cfg
        gene_col = cfg.model.get("gene_col", "index")

        # Work on a copy to avoid mutating caller's data
        adata = adata.copy()

        # Resolve gene names from the configured column
        gene_names = adata.var_names.tolist() if gene_col == "index" else adata.var[gene_col].tolist()

        # Map gene symbols -> Ensembl IDs
        name_to_ensembl = self._gene_name_id_dict
        mapped = {g: name_to_ensembl[g] for g in gene_names if g in name_to_ensembl}

        log.info(
            "Mapped %d/%d gene symbols to Ensembl IDs (%d in vocabulary)",
            len(mapped),
            len(gene_names),
            sum(1 for eid in mapped.values() if eid in self._token_dict),
        )
        if not mapped:
            raise ValueError(
                "No gene symbols matched the Geneformer gene_name_id dictionary. "
                f"Check that gene names (from adata.var['{gene_col}']) use standard symbols."
            )

        # Update var_names to Ensembl IDs (unmapped genes keep original names,
        # which the tokenizer will simply ignore)
        new_names = [name_to_ensembl.get(g, g) for g in gene_names]
        adata.var_names = new_names
        adata.var_names_make_unique()

        # Ensure raw counts are present (Geneformer expects unnormalized counts).
        # The .X matrix should be counts; we trust the caller for this.
        # Add n_counts obs column if missing (required by TranscriptomeTokenizer)
        if "n_counts" not in adata.obs.columns:
            import scipy.sparse

            x = adata.X
            if scipy.sparse.issparse(x):
                adata.obs["n_counts"] = np.array(x.sum(axis=1)).flatten()
            else:
                adata.obs["n_counts"] = x.sum(axis=1)

        n_cells = adata.n_obs
        tmp_dir = Path(tempfile.mkdtemp(prefix="geneformer_"))

        try:
            # Write adata to temp file
            data_dir = tmp_dir / "data"
            data_dir.mkdir()
            adata.write_h5ad(data_dir / "input.h5ad")

            # Resolve dictionary file paths
            import geneformer as gf_pkg

            pkg_dir = Path(gf_pkg.__file__).parent
            token_dict_file = (
                Path(cfg.model.token_dictionary_path)
                if cfg.model.get("token_dictionary_path")
                else pkg_dir / "token_dictionary_gc104M.pkl"
            )
            median_dict_file = (
                Path(cfg.model.median_dictionary_path)
                if cfg.model.get("median_dictionary_path")
                else pkg_dir / "gene_median_dictionary_gc104M.pkl"
            )
            gene_mapping_file = (
                Path(cfg.model.gene_name_id_dict_path)
                if cfg.model.get("gene_name_id_dict_path")
                else pkg_dir / "ensembl_mapping_dict_gc104M.pkl"
            )

            # Tokenize
            tok_output_dir = tmp_dir / "tokenized"
            tok_output_dir.mkdir()

            tokenizer = TranscriptomeTokenizer(
                model_input_size=cfg.model.get("max_input_size", 4096),
                special_token=True,
                use_h5ad_index=True,
                model_version=cfg.model.get("model_version", "V2"),
                token_dictionary_file=token_dict_file,
                gene_median_file=median_dict_file,
                gene_mapping_file=gene_mapping_file,
            )
            tokenizer.tokenize_data(
                data_directory=data_dir,
                output_directory=tok_output_dir,
                output_prefix="tokenized",
                file_format="h5ad",
            )

            # Find the tokenized dataset
            tokenized_dataset = tok_output_dir / "tokenized.dataset"
            if not tokenized_dataset.exists():
                # Some versions may output without .dataset suffix
                candidates = list(tok_output_dir.iterdir())
                raise FileNotFoundError(
                    f"Tokenized dataset not found at {tokenized_dataset}. Contents of output dir: {candidates}"
                )

            # Extract embeddings
            emb_output_dir = tmp_dir / "embeddings"
            emb_output_dir.mkdir()

            emb_mode = cfg.model.get("embedding_strategy", "cls")
            emb_layer = cfg.model.get("layer_to_quant", -1)

            embex = EmbExtractor(
                model_type="Pretrained",
                emb_mode=emb_mode,
                emb_layer=emb_layer,
                max_ncells=None,
                forward_batch_size=batch_size,
                model_version=cfg.model.get("model_version", "V2"),
                token_dictionary_file=str(token_dict_file),
            )

            embs_df = embex.extract_embs(
                model_directory=cfg.model.pretrained_path,
                input_data_file=tokenized_dataset,
                output_directory=emb_output_dir,
                output_prefix="embs",
            )

            # Convert DataFrame to numpy — columns are 0..hidden_size-1
            # (no label columns since emb_label=None)
            cell_embeddings = embs_df.iloc[:, : self._embedding_dim].values.astype(np.float32)

            if cell_embeddings.shape[0] != n_cells:
                log.warning(
                    "Embedding count (%d) differs from input cell count (%d). "
                    "EmbExtractor may have filtered or downsampled cells.",
                    cell_embeddings.shape[0],
                    n_cells,
                )

            log.info(
                "Geneformer embeddings: shape %s, embedding_dim %d",
                cell_embeddings.shape,
                self._embedding_dim,
            )

            return EmbeddingResult(
                cell_embeddings=cell_embeddings,
                gene_embeddings=None,
                embedding_dim=self._embedding_dim,
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Finetuning methods (not yet implemented)
    # ------------------------------------------------------------------

    def tokenize(self, adata) -> dict[str, Tensor]:
        raise NotImplementedError(
            "GeneformerWrapper.tokenize() is not implemented for zero-shot evaluation. "
            "Use geneformer.TranscriptomeTokenizer for file-based tokenization."
        )

    def create_dataset(self, adata) -> Dataset:
        raise NotImplementedError(
            "GeneformerWrapper.create_dataset() is not implemented for zero-shot evaluation. "
            "Use geneformer.TranscriptomeTokenizer to produce HuggingFace Datasets."
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "GeneformerWrapper.forward() is not implemented for zero-shot evaluation. Use EmbExtractor for inference."
        )

    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "GeneformerWrapper.compute_native_loss() is not implemented for zero-shot "
            "evaluation. Native objective: masked language model loss."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Geneformer"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def gene_vocabulary(self) -> list[str]:
        return self._gene_vocab
