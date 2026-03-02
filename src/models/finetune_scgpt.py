"""Finetune scGPT for batch integration via MLM + DAT.

Uses an external AdversarialDiscriminator with configurable gradient-reversal
scale (``dat_scale``) instead of scGPT's built-in discriminator (which
hardcodes scale=1.0). The TransformerModel is constructed with ``do_dab=False``
and pretrained weights are loaded with ``strict=False``.

Local training loop avoids the wandb dependency in ``scgpt.trainer``.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# External discriminator with configurable gradient-reversal scale
# ---------------------------------------------------------------------------


class AdversarialDiscriminator(nn.Module):
    """DAT discriminator with configurable gradient-reversal scale."""

    def __init__(self, d_model: int, n_cls: int, scale: float = 1.0, nlayers: int = 3):
        super().__init__()
        from scgpt.model.grad_reverse import grad_reverse

        self._grad_reverse = grad_reverse
        self.scale = scale
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.LeakyReLU())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x):
        x = self._grad_reverse(x, lambd=self.scale)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ScGPTFinetuneDataset(Dataset):
    """Per-cell dataset that yields gene IDs, expression values, and batch labels."""

    def __init__(self, count_matrix: np.ndarray, gene_ids: np.ndarray, batch_ids: np.ndarray, vocab):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]

        # Prepend <cls> token
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, 0.0)  # pad_value for CLS position

        return {
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
            "batch_labels": self.batch_ids[idx],
        }


# ---------------------------------------------------------------------------
# Collator with masking
# ---------------------------------------------------------------------------


class FinetuneCollator:
    """Pad, truncate, bin, and optionally mask expression values.

    Handles ``batch_labels`` passthrough (scGPT's DataCollator does not).
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_value: float,
        max_length: int,
        do_mlm: bool = True,
        mask_ratio: float = 0.4,
        mask_value: float = -1.0,
        n_bins: int = 51,
    ):
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.max_length = max_length
        self.do_mlm = do_mlm
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.n_bins = n_bins

    @staticmethod
    def _binning(row: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Bin non-zero expression values into n_bins categories."""
        nonzero = row != 0
        if nonzero.sum() == 0:
            return row
        vals = row[nonzero].float()
        bins = torch.quantile(vals, torch.linspace(0, 1, n_bins - 1, device=vals.device))
        binned = torch.bucketize(vals, bins)
        out = row.clone()
        out[nonzero] = binned.to(row.dtype)
        return out

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_ori_len = max(len(ex["genes"]) for ex in examples)
        max_length = min(self.max_length, max_ori_len)

        padded_genes = []
        padded_expr = []
        batch_labels = []

        for ex in examples:
            genes = ex["genes"]
            expr = ex["expressions"]

            # Bin expression values (skip CLS at position 0)
            expr[1:] = self._binning(expr[1:], self.n_bins)

            # Truncate by random sampling (keep CLS at position 0)
            if len(genes) > max_length:
                keep_idx = torch.randperm(len(genes) - 1)[: max_length - 1] + 1
                keep_idx = torch.cat([torch.tensor([0]), keep_idx.sort().values])
                genes = genes[keep_idx]
                expr = expr[keep_idx]

            # Pad
            pad_len = max_length - len(genes)
            if pad_len > 0:
                genes = torch.cat([genes, torch.full((pad_len,), self.pad_token_id, dtype=genes.dtype)])
                expr = torch.cat([expr, torch.full((pad_len,), self.pad_value, dtype=expr.dtype)])

            padded_genes.append(genes)
            padded_expr.append(expr)
            batch_labels.append(ex["batch_labels"])

        gene_tensor = torch.stack(padded_genes)
        expr_tensor = torch.stack(padded_expr)
        batch_tensor = torch.tensor(batch_labels, dtype=torch.long)

        # Masking for MLM
        if self.do_mlm:
            masked_expr = expr_tensor.clone()
            prob_matrix = torch.full(expr_tensor.shape, self.mask_ratio)
            prob_matrix[expr_tensor == self.pad_value] = 0
            prob_matrix[:, 0] = 0  # Never mask CLS
            mask = torch.bernoulli(prob_matrix).bool()
            masked_expr[mask] = self.mask_value
        else:
            masked_expr = expr_tensor

        return {
            "gene": gene_tensor,
            "expr": expr_tensor,
            "masked_expr": masked_expr,
            "batch_labels": batch_tensor,
        }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _train_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    mask_value: float,
    pad_token_id: int,
    discriminator=None,
    dat_weight: float = 2.0,
    ecs_enabled: bool = False,
    amp: bool = False,
) -> float:
    """One training epoch: MLM + DAT (+ optional ECS)."""
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_mlm = 0.0
    total_dat = 0.0
    total_ecs = 0.0
    num_batches = len(train_loader)

    for data_dict in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
        gene_ids = data_dict["gene"].to(device)
        expr = data_dict["expr"].to(device)
        masked_expr = data_dict["masked_expr"].to(device)
        batch_labels = data_dict["batch_labels"].to(device)

        src_key_padding_mask = gene_ids.eq(pad_token_id)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            output_dict = model(
                gene_ids,
                masked_expr,
                src_key_padding_mask,
                MVC=False,
                ECS=ecs_enabled,
            )

            # MLM loss on masked positions
            mlm_pred = output_dict["mlm_output"]  # (batch, seq_len)
            masked_positions = masked_expr.eq(mask_value)
            mlm_loss = nn.functional.mse_loss(mlm_pred[masked_positions], expr[masked_positions])

            loss = mlm_loss

            # DAT loss via external discriminator
            if discriminator is not None:
                cell_emb = output_dict["cell_emb"]
                dat_pred = discriminator(cell_emb.float())
                dat_loss = nn.functional.cross_entropy(dat_pred, batch_labels)
                loss = loss + dat_weight * dat_loss

            # ECS loss (computed by model when ECS=True)
            if ecs_enabled and "loss_ecs" in output_dict:
                loss = loss + output_dict["loss_ecs"]

        total_mlm += mlm_loss.item()
        if discriminator is not None:
            total_dat += dat_loss.item()
        if ecs_enabled and "loss_ecs" in output_dict:
            total_ecs += output_dict["loss_ecs"].item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()
    n = num_batches
    avg_mlm = total_mlm / n
    avg_dat = total_dat / n
    avg_ecs = total_ecs / n
    # ECS is added unweighted (weight=1) in scGPT; DAT uses dat_weight
    total = avg_mlm + dat_weight * avg_dat + avg_ecs

    if discriminator is not None:
        dat_str = f"DAT={avg_dat:.4f} (x{dat_weight}={dat_weight * avg_dat:.4f})"
    else:
        dat_str = "DAT=inactive"
    ecs_str = f"ECS={avg_ecs:.4f}" if ecs_enabled else "ECS=inactive"

    log.info(
        "Epoch %d train | total=%.4f | MLM=%.4f | %s | %s | lr=%.6f",
        epoch,
        total,
        avg_mlm,
        dat_str,
        ecs_str,
        scheduler.get_last_lr()[0],
    )
    return total_mlm / num_batches


def _evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
    mask_value: float,
    pad_token_id: int,
    amp: bool = False,
) -> float:
    """Evaluate MLM loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for data_dict in tqdm(data_loader, desc="Validation", leave=False):
            gene_ids = data_dict["gene"].to(device)
            expr = data_dict["expr"].to(device)
            masked_expr = data_dict["masked_expr"].to(device)

            src_key_padding_mask = gene_ids.eq(pad_token_id)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output_dict = model(
                    gene_ids,
                    masked_expr,
                    src_key_padding_mask,
                    MVC=False,
                    ECS=False,
                )

                mlm_pred = output_dict["mlm_output"]
                masked_positions = masked_expr.eq(mask_value)
                loss = nn.functional.mse_loss(mlm_pred[masked_positions], expr[masked_positions])

            total_loss += loss.item() * len(gene_ids)
            total_num += len(gene_ids)

    return total_loss / total_num


def _get_embeddings(
    model,
    adata,
    gene_ids: np.ndarray,
    vocab,
    model_configs: dict,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """Get normalized CLS cell embeddings for ALL cells using scGPT's utility."""
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings

    return get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def finetune_scgpt(adata, cfg):
    """Finetune scGPT on adata and return embeddings for ALL cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Full dataset (train/val split is done internally).
    cfg : DictConfig
        Full Hydra config with ``cfg.model`` and ``cfg.task.finetune``.

    Returns
    -------
    FinetuneResult
    """
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    from models.base import FinetuneResult

    ft_cfg = cfg.task.finetune
    seed = ft_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Finetuning scGPT on device: %s", device)

    # --- Load vocabulary and model config ---
    model_dir = Path(cfg.model.pretrained_path)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    vocab = GeneVocab.from_file(vocab_file)
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    with open(model_config_file) as f:
        model_configs = json.load(f)

    pad_token = model_configs["pad_token"]
    pad_value = model_configs["pad_value"]
    embsize = model_configs["embsize"]
    max_seq_len = cfg.model.get("max_seq_len", 1200)
    n_bins = cfg.model.get("n_bins", 51)

    # --- Preprocess: gene intersection + HVG ---
    gene_col = cfg.model.get("gene_col", "index")
    genes = adata.var_names.tolist() if gene_col == "index" else adata.var[gene_col].tolist()
    gene_ids_in_vocab = [vocab[g] if g in vocab else -1 for g in genes]  # noqa: SIM401
    mask = np.array([x >= 0 for x in gene_ids_in_vocab])
    n_matched = mask.sum()
    log.info("Matched %d/%d genes in scGPT vocabulary", n_matched, len(genes))
    if n_matched == 0:
        raise ValueError("No genes matched the scGPT vocabulary.")

    adata = adata[:, mask].copy()
    matched_genes = adata.var_names.tolist() if gene_col == "index" else adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(matched_genes), dtype=int)

    # HVG selection
    n_hvg = ft_cfg.get("n_hvg", cfg.model.get("n_top_genes", 2000))
    if adata.n_vars > n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
        hvg_mask = adata.var["highly_variable"].values
        adata = adata[:, hvg_mask].copy()
        gene_ids = gene_ids[hvg_mask]
        log.info("Selected %d HVGs", adata.n_vars)

    # --- Batch labels ---
    batch_key = cfg.task.metadata.batch_key
    batch_ids = adata.obs[batch_key].astype("category").cat.codes.values
    n_batches = len(np.unique(batch_ids))

    # Store batch_id in adata.obs for get_batch_cell_embeddings
    adata.obs["batch_id"] = batch_ids

    # Get count matrix
    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    # --- Train/val split ---
    val_fraction = ft_cfg.get("val_fraction", 0.1)
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=val_fraction, shuffle=True, random_state=seed)
    log.info("Train: %d cells, Valid: %d cells", len(train_idx), len(val_idx))

    # --- Build model (no built-in DAB — we use an external discriminator) ---
    do_dat = ft_cfg.get("do_dat", True)
    ecs_threshold = ft_cfg.get("ecs_threshold", 0.0)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=cfg.model.get("use_fast_transformer", False),
        fast_transformer_backend="flash",
        pre_norm=False,
        ecs_threshold=ecs_threshold if ecs_threshold > 0 else 0.3,
    )

    # Load pretrained weights (strict=False: new DAB params init randomly)
    pretrained_params = torch.load(model_file, map_location=device)
    load_pretrained(model, pretrained_params, verbose=False)
    model.to(device)

    amp = ft_cfg.get("amp", False)
    if amp:
        log.info("Using bfloat16 autocast (activations in bf16, weights in fp32)")

    # --- Setup external DAT discriminator ---
    discriminator = None
    if do_dat:
        dat_scale = ft_cfg.get("dat_scale", 1.0)
        discriminator = AdversarialDiscriminator(d_model=embsize, n_cls=n_batches, scale=dat_scale).to(device)
        log.info("DAT enabled: %d batches, scale=%.1f", n_batches, dat_scale)

    log.info("Model loaded: embsize=%d, n_batches=%d, do_dat=%s", embsize, n_batches, do_dat)

    # --- Datasets and loaders ---
    mask_ratio = ft_cfg.get("mask_ratio", 0.4)
    mask_value = -1.0
    batch_size = ft_cfg.get("batch_size", 32)

    pad_token_id = vocab[pad_token]

    train_dataset = ScGPTFinetuneDataset(X[train_idx], gene_ids, batch_ids[train_idx], vocab)
    val_dataset = ScGPTFinetuneDataset(X[val_idx], gene_ids, batch_ids[val_idx], vocab)

    train_collator = FinetuneCollator(
        pad_token_id=pad_token_id,
        pad_value=pad_value,
        max_length=max_seq_len,
        do_mlm=True,
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        n_bins=n_bins,
    )
    val_collator = FinetuneCollator(
        pad_token_id=pad_token_id,
        pad_value=pad_value,
        max_length=max_seq_len,
        do_mlm=True,
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        n_bins=n_bins,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collator, num_workers=4, pin_memory=True
    )

    # --- Optimizer and scheduler ---
    lr = ft_cfg.get("lr", 1e-4)
    schedule_ratio = ft_cfg.get("schedule_ratio", 0.9)
    params = list(model.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=schedule_ratio)

    # --- Training loop ---
    epochs = ft_cfg.get("epochs", 15)
    dat_weight = ft_cfg.get("dat_weight", 2.0)
    ecs_enabled = ecs_threshold > 0

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            mask_value=mask_value,
            pad_token_id=pad_token_id,
            discriminator=discriminator,
            dat_weight=dat_weight,
            ecs_enabled=ecs_enabled,
            amp=amp,
        )

        val_loss = _evaluate(model, val_loader, device, mask_value=mask_value, pad_token_id=pad_token_id, amp=amp)
        log.info("Epoch %d val loss: %.4f (best: %.4f at epoch %d)", epoch, val_loss, best_val_loss, best_epoch)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model_state = copy.deepcopy(model.state_dict())
        #     best_epoch = epoch
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     best_model_state = copy.deepcopy(model.state_dict())
        #     if patience_counter >= patience:
        #         log.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
        #         break
        best_model_state = copy.deepcopy(model.state_dict())

    # --- Restore best model and get embeddings for ALL cells ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    log.info("Computing embeddings for all %d cells using best model (epoch %d)", adata.n_obs, best_epoch)
    embeddings = _get_embeddings(
        model,
        adata,
        gene_ids,
        vocab,
        model_configs,
        batch_size=batch_size,
        max_length=max_seq_len,
        device=device,
    )

    return FinetuneResult(
        cell_embeddings=embeddings,
        embedding_dim=embsize,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
    )
