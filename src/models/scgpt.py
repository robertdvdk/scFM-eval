"""scGPT foundation model wrapper.

scGPT uses masked gene expression prediction as its pretraining objective.
Tokenization maps gene names to vocabulary indices and bins expression values.
Cell embeddings are extracted from the CLS token.

This wrapper delegates to scGPT's ``get_batch_cell_embeddings()`` for
zero-shot embedding extraction. Finetuning methods (tokenize, create_dataset,
forward, compute_native_loss) are not yet implemented.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from . import register_model
from .base import EmbeddingResult, FoundationModelWrapper

log = logging.getLogger(__name__)


@register_model("scgpt")
class ScGPTWrapper(FoundationModelWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._embedding_dim: int = cfg.model.d_model
        self._gene_vocab: list[str] = []
        self._vocab = None  # GeneVocab, set by load_pretrained
        self._model_configs: dict | None = None  # args.json contents

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained

        cfg = self.cfg
        model_dir = Path(cfg.model.pretrained_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        vocab_file = model_dir / "vocab.json"
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        for f in (vocab_file, model_config_file, model_file):
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")

        # Vocabulary
        vocab = GeneVocab.from_file(vocab_file)
        for s in ("<pad>", "<cls>", "<eoc>"):
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])

        # Model config from args.json
        with open(model_config_file) as f:
            model_configs = json.load(f)

        use_fast_transformer = cfg.model.use_fast_transformer

        # Construct model (mirrors scgpt.tasks.cell_emb.embed_data)
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,
            vocab=vocab,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=use_fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        load_pretrained(
            model,
            torch.load(model_file, map_location=self._device),
            verbose=False,
        )
        model.to(self._device)
        model.eval()

        self._model = model
        self._vocab = vocab
        self._model_configs = model_configs
        self._embedding_dim = model_configs["embsize"]
        self._gene_vocab = [t for t in vocab.get_itos() if t not in ("<pad>", "<cls>", "<eoc>")]

        log.info(
            "Loaded scGPT: d_model=%d, vocab_size=%d, genes=%d",
            self._embedding_dim,
            len(vocab),
            len(self._gene_vocab),
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        from scgpt.tasks.cell_emb import get_batch_cell_embeddings

        cfg = self.cfg
        vocab = self._vocab
        gene_col = cfg.model.gene_col

        # Resolve gene names
        genes = adata.var_names.tolist() if gene_col == "index" else adata.var[gene_col].tolist()

        # Map genes to vocab indices (GeneVocab has no .get(); must check membership)
        gene_ids_in_vocab = [vocab[g] if g in vocab else -1 for g in genes]  # noqa: SIM401
        matched = sum(1 for x in gene_ids_in_vocab if x >= 0)
        log.info(
            "Matched %d/%d genes in vocabulary of size %d",
            matched,
            len(genes),
            len(vocab),
        )
        if matched == 0:
            raise ValueError(
                "No genes matched the scGPT vocabulary. Check that gene names "
                f"(from adata.var['{gene_col}']) match the vocabulary."
            )

        # Filter to matched genes
        mask = np.array([x >= 0 for x in gene_ids_in_vocab])
        adata_filtered = adata[:, mask].copy()

        # Get vocab indices for matched genes only
        matched_genes = (
            adata_filtered.var_names.tolist() if gene_col == "index" else adata_filtered.var[gene_col].tolist()
        )
        gene_ids = np.array(vocab(matched_genes), dtype=int)

        # HVG selection to avoid random truncation in DataCollator
        n_top_genes = cfg.model.get("n_top_genes", 2000)
        if adata_filtered.n_vars > n_top_genes:
            sc.pp.highly_variable_genes(adata_filtered, n_top_genes=n_top_genes)
            hvg_mask = adata_filtered.var["highly_variable"].values
            adata_filtered = adata_filtered[:, hvg_mask].copy()
            gene_ids = gene_ids[hvg_mask]
            log.info("Selected %d HVGs from %d vocab-matched genes", adata_filtered.n_vars, matched)

        cell_embeddings = get_batch_cell_embeddings(
            adata_filtered,
            cell_embedding_mode="cls",
            model=self._model,
            vocab=self._vocab,
            max_length=cfg.model.max_seq_len,
            batch_size=batch_size,
            model_configs=self._model_configs,
            gene_ids=gene_ids,
            use_batch_labels=False,
        )

        return EmbeddingResult(
            cell_embeddings=cell_embeddings,
            gene_embeddings=None,
            embedding_dim=self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Finetuning methods (not yet implemented)
    # ------------------------------------------------------------------

    def tokenize(self, adata) -> dict[str, Tensor]:
        raise NotImplementedError(
            "ScGPTWrapper.tokenize() is not implemented for zero-shot evaluation. "
            "For finetuning, see scgpt.tokenizer and scgpt.data_collator."
        )

    def create_dataset(self, adata) -> Dataset:
        raise NotImplementedError(
            "ScGPTWrapper.create_dataset() is not implemented for zero-shot evaluation. "
            "For finetuning, see scgpt.tasks.cell_emb.get_batch_cell_embeddings "
            "for the internal Dataset pattern."
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "ScGPTWrapper.forward() is not implemented for zero-shot evaluation. "
            "For finetuning, use model._encode() directly."
        )

    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "ScGPTWrapper.compute_native_loss() is not implemented for zero-shot "
            "evaluation. Native objective: masked gene expression prediction."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "scGPT"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def gene_vocabulary(self) -> list[str]:
        return self._gene_vocab
