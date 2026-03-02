"""CancerFoundation model wrapper.

CancerFoundation is a Transformer-based foundation model for single-cell
RNA-seq expression modeling. It uses masked language modeling with optional
generative training, domain adversarial training, and MVC objectives.

Cell embeddings are extracted from the CLS token after preprocessing that
includes gene intersection with vocabulary, HVG selection, and expression
value binning.

The embed() method replicates the same preprocessing and forward-pass logic
used by finetune_cancerfoundation.py so that zero-shot results exactly match
finetuning with 0 epochs.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from omegaconf import DictConfig
from scipy.sparse import issparse
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from . import register_model
from .base import EmbeddingResult, FoundationModelWrapper

log = logging.getLogger(__name__)


def _encode_cls(
    module: torch.nn.Module,
    gene_ids: torch.Tensor,
    values: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    conditions: dict | None,
) -> torch.Tensor:
    """Run only the encoder and return the CLS token embedding.

    This skips the decoder, MVC decoder, and DAT discriminator — they don't
    affect the cell embedding but consume significant VRAM.  Works for both
    generative (CFGenerator) and perceptual (TransformerEncoder) variants.
    """
    # Gene + value embeddings
    gene_embs = module.gene_encoder(gene_ids) if hasattr(module, "gene_encoder") else module.encoder(gene_ids)
    value_embs = module.value_encoder(values)
    total_embs = gene_embs + value_embs

    # Run through the transformer encoder
    if module.use_generative_training:
        seq_len = gene_ids.shape[1]
        # Handle conditions prepended to the sequence
        if conditions is not None and module.where_condition == "begin" and module.conditions:
            for cond_name, cond_values in conditions.items():
                cond_emb = module.condition_encoders[cond_name](cond_values)
            cond_emb = cond_emb.unsqueeze(1)
            total_embs = torch.cat([total_embs[:, :1, :], cond_emb, total_embs[:, 1:, :]], dim=1)
            src_key_padding_mask = torch.nn.functional.pad(src_key_padding_mask, (1, 0), "constant", False)
            seq_len += 1

        pcpt_output, _ = module.transformer_encoder(
            pcpt_total_embs=total_embs,
            gen_total_embs=None,
            src_key_padding_mask=src_key_padding_mask,
            attn_mask=torch.zeros((seq_len, seq_len), device=gene_ids.device),
        )
        return pcpt_output[:, 0, :]
    else:
        # Perceptual / standard TransformerEncoder
        if conditions and hasattr(module, "condition_encoders"):
            cond_emb = module.condition_encoders["technology"](conditions["technology"])
            value_embs[:, 1, :] = cond_emb
            total_embs = gene_embs + value_embs

        output = module.transformer_encoder(total_embs, src_key_padding_mask=src_key_padding_mask)
        return output[:, 0, :]


@register_model("cancerfoundation")
class CancerFoundationWrapper(FoundationModelWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._embedding_dim: int = cfg.model.embsize
        self._gene_vocab: list[str] = []
        self._vocab: dict[str, int] | None = None  # token -> id mapping
        self._cf_model = None  # CancerFoundation Lightning module

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        cfg = self.cfg

        # Add CancerFoundation source repo to sys.path so the package is importable
        # without installation (e.g. from ~/CancerFoundation).
        source_path = cfg.model.get("source_path", None)
        if source_path is not None:
            source_path = str(Path(source_path).resolve())
            if source_path not in sys.path:
                sys.path.insert(0, source_path)

        # The cancerfoundation package's __init__.py imports SingleCellDataset,
        # which in turn imports bionemo (an NVIDIA-only package not needed for
        # inference). Stub it out so the import chain doesn't fail.
        if "bionemo" not in sys.modules:
            import types

            bionemo = types.ModuleType("bionemo")
            bionemo.scdl = types.ModuleType("bionemo.scdl")
            bionemo.scdl.io = types.ModuleType("bionemo.scdl.io")
            bionemo.scdl.io.single_cell_memmap_dataset = types.ModuleType("bionemo.scdl.io.single_cell_memmap_dataset")
            bionemo.scdl.io.single_cell_memmap_dataset.SingleCellMemMapDataset = None
            sys.modules["bionemo"] = bionemo
            sys.modules["bionemo.scdl"] = bionemo.scdl
            sys.modules["bionemo.scdl.io"] = bionemo.scdl.io
            sys.modules["bionemo.scdl.io.single_cell_memmap_dataset"] = bionemo.scdl.io.single_cell_memmap_dataset

        from cancerfoundation.loss import LossType
        from cancerfoundation.model.model import CancerFoundation

        # PyTorch 2.6+ defaults to weights_only=True; allowlist checkpoint types.
        torch.serialization.add_safe_globals([LossType])

        model_dir = Path(cfg.model.pretrained_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        vocab_file = model_dir / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

        # Find checkpoint file
        checkpoint_path = cfg.model.get("checkpoint_filename", None)
        if checkpoint_path:
            ckpt = model_dir / checkpoint_path
        else:
            # Auto-detect: look for .ckpt files
            ckpts = list(model_dir.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt files found in {model_dir}")
            ckpt = ckpts[0]
            if len(ckpts) > 1:
                log.warning(
                    "Multiple checkpoints found in %s, using %s. Set model.checkpoint_filename to specify.",
                    model_dir,
                    ckpt.name,
                )

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        # Load vocabulary
        with open(vocab_file) as f:
            self._vocab = json.load(f)

        # Load model from checkpoint
        self._cf_model = CancerFoundation.load_from_checkpoint(
            str(ckpt),
            vocab=self._vocab,
            strict=True,
        )
        self._cf_model.eval()

        # Extract model properties
        self._embedding_dim = self._cf_model.embsize

        # Build gene vocabulary (exclude special tokens)
        special_tokens = {"<pad>", "<cls>", "<unk>", "<mask>", "<eos>", "<eoc>"}
        self._gene_vocab = [k for k in self._vocab if k not in special_tokens]

        log.info(
            "Loaded CancerFoundation: embsize=%d, vocab_size=%d, genes=%d, checkpoint=%s",
            self._embedding_dim,
            len(self._vocab),
            len(self._gene_vocab),
            ckpt.name,
        )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, adata, batch_size: int = 64) -> EmbeddingResult:
        """Embed cells using the same pipeline as finetune_cancerfoundation.

        This replicates the preprocessing (vocab intersection, HVG, binning)
        and forward-pass logic from finetune_cancerfoundation.py so that
        zero-shot results exactly match finetuning with 0 epochs.
        """
        from cancerfoundation.data.preprocess import binning as cf_binning

        cfg = self.cfg
        gene_col = cfg.model.get("gene_col", "index")
        model = self._cf_model
        vocab = self._vocab

        # Move model to device
        model.to(self._device)
        model.eval()

        # Work on a copy
        data = adata.copy()
        if gene_col != "index":
            data.var_names = data.var[gene_col].tolist()
            data.var_names_make_unique()

        # --- Preprocessing (matches finetune_cancerfoundation.preprocess_data) ---

        # Gene intersection with vocabulary
        data.var["gene_name"] = data.var.index.tolist()
        data.var["id_in_vocab"] = [1 if g in vocab else -1 for g in data.var["gene_name"]]
        n_matched = (data.var["id_in_vocab"] >= 0).sum()
        log.info("Matched %d/%d genes in vocabulary of size %d", n_matched, len(data.var), len(vocab))
        if n_matched == 0:
            raise ValueError("No genes matched the CancerFoundation vocabulary.")
        data = data[:, data.var["id_in_vocab"] >= 0].copy()

        # HVG selection
        n_top_genes = cfg.model.get("n_top_genes", 2000)
        sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes)
        data = data[:, data.var["highly_variable"]].copy()
        log.info("Selected %d highly variable genes", data.n_vars)

        # Expression binning
        n_bins = model.n_bins
        normalise_bins = model.normalise_bins
        X = data.X.toarray() if issparse(data.X) else data.X.copy()
        X_binned = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_binned[i] = cf_binning(X[i], n_bins)
            if normalise_bins:
                X_binned[i] = X_binned[i] / n_bins

        # Gene IDs
        genes = data.var["gene_name"].tolist()
        gene_ids = np.array([vocab[g] for g in genes])

        # --- Tokenize ---
        # Sequence length = n_genes + 1 (CLS). We don't use model.max_seq_len
        # because the transformer has no positional encoding and can handle any
        # length; truncating to the training-time max_seq_len would silently
        # discard genes the user explicitly selected via n_top_genes.
        pad_value = model.pad_value
        cls_token_id = vocab["<cls>"]
        pad_token_id = vocab["<pad>"]

        n_cells = X_binned.shape[0]
        all_gene_ids_np = np.concatenate([[cls_token_id], gene_ids])
        all_genes = np.tile(all_gene_ids_np, (n_cells, 1))
        cls_col = np.full((n_cells, 1), pad_value)
        all_values = np.concatenate([cls_col, X_binned], axis=1)

        all_gene_ids = torch.from_numpy(all_genes).long()
        all_values = torch.from_numpy(all_values).float()

        # --- Forward pass (encoder-only, skips decoder/MVC/DAT for speed + VRAM) ---
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, batch_size), desc="Embedding", leave=False):
                batch_genes = all_gene_ids[i : i + batch_size].to(self._device)
                batch_values = all_values[i : i + batch_size].to(self._device)
                src_key_padding_mask = batch_genes.eq(pad_token_id)
                current_batch_size = batch_genes.shape[0]

                # Conditions (dummy zeros if model was trained with conditions)
                conditions = None
                if model.model.conditions is not None:
                    conditions = {
                        cond_name: torch.zeros(current_batch_size, dtype=torch.long, device=self._device)
                        for cond_name in model.model.conditions
                    }

                cell_emb = _encode_cls(model.model, batch_genes, batch_values, src_key_padding_mask, conditions)
                embeddings.append(cell_emb.float().cpu().numpy())

        cell_embeddings = np.concatenate(embeddings, axis=0)

        if cfg.model.get("l2_normalize", False):
            norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
            cell_embeddings = cell_embeddings / norms
            log.info("Applied L2 normalization to cell embeddings")

        log.info(
            "CancerFoundation embeddings: shape %s, embedding_dim %d",
            cell_embeddings.shape,
            self._embedding_dim,
        )

        return EmbeddingResult(
            cell_embeddings=cell_embeddings,
            gene_embeddings=None,
            embedding_dim=self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> CancerFoundationWrapper:
        self._device = torch.device(device)
        if self._cf_model is not None:
            self._cf_model = self._cf_model.to(self._device)
        return self

    # ------------------------------------------------------------------
    # Finetuning methods (not yet implemented)
    # ------------------------------------------------------------------

    def tokenize(self, adata) -> dict[str, Tensor]:
        raise NotImplementedError(
            "CancerFoundationWrapper.tokenize() is not implemented for zero-shot evaluation. "
            "For finetuning, see cancerfoundation.data.data_collator.AnnDataCollator."
        )

    def create_dataset(self, adata) -> Dataset:
        raise NotImplementedError(
            "CancerFoundationWrapper.create_dataset() is not implemented for zero-shot evaluation. "
            "For finetuning, see cancerfoundation.data.dataset.SingleCellDataset."
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "CancerFoundationWrapper.forward() is not implemented for zero-shot evaluation. "
            "For finetuning, use the CancerFoundation Lightning module directly."
        )

    def compute_native_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "CancerFoundationWrapper.compute_native_loss() is not implemented for zero-shot "
            "evaluation. Native objective: masked gene expression prediction."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "CancerFoundation"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def gene_vocabulary(self) -> list[str]:
        return self._gene_vocab
