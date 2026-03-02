"""Finetune CancerFoundation for batch integration via MLM + DAT.

Extracted from /CancerFoundation/finetune.py. Receives an AnnData and config
from the subprocess worker, trains the model, and returns embeddings for ALL
cells as a FinetuneResult.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
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
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers (from /CancerFoundation/finetune.py)
# ---------------------------------------------------------------------------


class FinetuneDataset(Dataset):
    """Dataset for finetuning with masked value prediction."""

    def __init__(self, gene_ids: torch.Tensor, values: torch.Tensor, batch_labels: torch.Tensor):
        self.gene_ids = gene_ids
        self.values = values
        self.batch_labels = batch_labels

    def __len__(self):
        return self.gene_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "gene_ids": self.gene_ids[idx],
            "values": self.values[idx],
            "batch_labels": self.batch_labels[idx],
        }


def random_mask_value(
    values: torch.Tensor,
    mask_ratio: float,
    mask_value: float,
    pad_value: float,
) -> torch.Tensor:
    """Apply random masking to expression values."""
    masked = values.clone()
    prob_matrix = torch.full(values.shape, mask_ratio)
    prob_matrix[values == pad_value] = 0
    mask = torch.bernoulli(prob_matrix).bool()
    masked[mask] = mask_value
    return masked


def ecs_loss(cell_emb: torch.Tensor, ecs_threshold: float = 0.8) -> torch.Tensor:
    """Elastic Cell Similarity loss."""
    cell_emb_normed = nn.functional.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
    mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
    cos_sim = cos_sim[mask]
    loss = -torch.pow(cos_sim - ecs_threshold, 2)
    return loss.mean()


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: dict,
    pad_value: float,
    cls_token_id: int,
    include_zero_gene: bool = True,
) -> dict[str, torch.Tensor]:
    """Tokenize and pad a batch of expression data."""
    batch_size = data.shape[0]
    genes_list = []
    values_list = []

    for i in range(batch_size):
        row = data[i]
        if include_zero_gene:
            genes = gene_ids
            values = row
        else:
            nonzero_mask = row > 0
            genes = gene_ids[nonzero_mask]
            values = row[nonzero_mask]

        # Truncate (keep first max_len - 1 to leave room for CLS)
        if len(genes) > max_len - 1:
            idx = np.random.permutation(len(genes))[: max_len - 1]
            genes = genes[idx]
            values = values[idx]

        # Add CLS token at beginning
        genes = np.concatenate([[cls_token_id], genes])
        values = np.concatenate([[pad_value], values])

        # Pad to max_len
        pad_len = max_len - len(genes)
        if pad_len > 0:
            pad_token_id = vocab["<pad>"]
            genes = np.concatenate([genes, np.full(pad_len, pad_token_id)])
            values = np.concatenate([values, np.full(pad_len, pad_value)])

        genes_list.append(genes)
        values_list.append(values)

    return {
        "genes": torch.from_numpy(np.array(genes_list)).long(),
        "values": torch.from_numpy(np.array(values_list)).float(),
    }


def preprocess_data(
    adata: sc.AnnData,
    vocab: dict,
    n_hvg: int = 1200,
    n_bins: int = 51,
    normalise_bins: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocess AnnData for finetuning: gene intersection, HVG, binning."""
    from cancerfoundation.data.preprocess import binning as cf_binning

    adata.var["gene_name"] = adata.var.index.tolist()
    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
    n_matched = (adata.var["id_in_vocab"] >= 0).sum()
    log.info("Matched %d/%d genes in vocabulary of size %d", n_matched, len(adata.var), len(vocab))
    adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    adata = adata[:, adata.var["highly_variable"]].copy()
    log.info("Selected %d highly variable genes", adata.n_vars)

    X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()

    X_binned = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_binned[i] = cf_binning(X[i], n_bins)
        if normalise_bins:
            X_binned[i] = X_binned[i] / n_bins

    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array([vocab[g] for g in genes])
    return X_binned, gene_ids, genes


def _get_dummy_conditions(model, batch_size: int, device: torch.device) -> dict | None:
    """Create dummy conditions dict if model was trained with conditions."""
    if model.model.conditions is None:
        return None
    return {cond_name: torch.zeros(batch_size, dtype=torch.long, device=device) for cond_name in model.model.conditions}


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


def _model_forward(model, gene_ids, values, src_key_padding_mask, conditions) -> dict:
    """Forward pass handling both generative and perceptual CancerFoundation variants."""
    if model.model.use_generative_training:
        batch_size, seq_len = gene_ids.shape
        output_dict = model.model.generative_forward(
            pcpt_genes=gene_ids,
            pcpt_values=values,
            pcpt_key_padding_mask=src_key_padding_mask,
            gen_genes=None,
            gen_key_padding_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            attn_mask=torch.zeros((seq_len, seq_len), device=gene_ids.device),
            conditions=conditions,
        )
        return {
            "mlm_output": output_dict["pcpt_preds"],
            "cell_emb": output_dict["cell_emb"],
            "mvc_output": output_dict.get("mvc_output"),
        }
    else:
        return model.model.perceptual_forward(
            src=gene_ids,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            conditions=conditions,
        )


def wasserstein_condition_loss(scores, labels):
    """
    scores: (batch, n_cls) — raw critic outputs
    labels: (batch,) — integer batch/condition labels

    For each sample from batch k, we want score_k to be high
    and all other scores to be low.
    """
    # Gather the score for the correct class
    correct_scores = scores.gather(1, labels.unsqueeze(1)).squeeze(1)  # (batch,)

    # Mean of scores for incorrect classes
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    incorrect_scores = (scores * mask).sum(1) / mask.sum(1)  # (batch,)

    # Critic wants to maximize this gap
    # Encoder (via grad reversal) wants to minimize it
    loss = -(correct_scores - incorrect_scores).mean()

    return loss


def _train_epoch(
    model,
    train_loader: DataLoader,
    opt_model,
    opt_disc,
    scheduler,
    device: torch.device,
    mask_ratio: float,
    mask_value: float,
    pad_value: float,
    epoch: int,
    discriminator=None,
    dat_weight: float = 1.0,
    n_critic: int = 1,
    ecs_threshold: float = 0.0,
    ecs_weight: float = 10.0,
    amp: bool = False,
) -> float:
    """Train for one epoch with optional DAT and ECS losses.

    When n_critic > 1, trains the discriminator for n_critic steps per encoder
    step (WGAN-style) using detached embeddings.
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    total_mlm = 0.0
    total_mvc = 0.0
    total_dat = 0.0
    total_dat_critic = 0.0
    total_ecs = 0.0
    num_batches = len(train_loader)

    for batch_data in tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False):
        gene_ids = batch_data["gene_ids"].to(device)
        values = batch_data["values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        batch_size = gene_ids.shape[0]

        masked_values = random_mask_value(values, mask_ratio, mask_value, pad_value)
        src_key_padding_mask = gene_ids.eq(model.pad_token_id)
        conditions = _get_dummy_conditions(model, batch_size, device)

        # --- n_critic discriminator-only steps (WGAN-style) ---
        if discriminator is not None and n_critic > 1:
            for _ in range(n_critic):
                opt_disc.zero_grad()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    output_dict_d = _model_forward(model, gene_ids, masked_values, src_key_padding_mask, conditions)
                    cell_emb_d = output_dict_d["cell_emb"]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    batch_pred_d = discriminator(cell_emb_d.detach().float())
                    dat_loss_d = wasserstein_condition_loss(batch_pred_d, batch_labels)
                dat_loss_d.backward()
                opt_disc.step()
            total_dat_critic += dat_loss_d.item()

        # --- Full encoder + discriminator step ---
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            output_dict = _model_forward(model, gene_ids, masked_values, src_key_padding_mask, conditions)
            output_values = output_dict["mlm_output"]

            target_values = values.clone()
            masked_positions = masked_values == mask_value
            mlm_loss = nn.functional.mse_loss(output_values[masked_positions], target_values[masked_positions])
            loss = mlm_loss

            # MVC loss if enabled
            if model.do_mvc and output_dict.get("mvc_output") is not None:
                mvc_out = output_dict["mvc_output"]
                mvc_loss = nn.functional.mse_loss(mvc_out[masked_positions], target_values[masked_positions])
                loss = loss + mvc_loss

            # Domain adversarial training (Wasserstein + grad_reverse in discriminator)
            cell_emb = output_dict["cell_emb"]
            if discriminator is not None:
                batch_pred = discriminator(cell_emb.float())
                dat_loss = wasserstein_condition_loss(batch_pred, batch_labels)
                loss = loss - dat_weight * dat_loss

            # Elastic cell similarity
            if ecs_threshold > 0:
                loss_ecs = ecs_loss(cell_emb.float(), ecs_threshold)
                loss = loss + ecs_weight * loss_ecs

        total_mlm += mlm_loss.item()
        if model.do_mvc and output_dict.get("mvc_output") is not None:
            total_mvc += mvc_loss.item()
        if discriminator is not None:
            total_dat += dat_loss.item()
        if ecs_threshold > 0:
            total_ecs += loss_ecs.item()

        opt_model.zero_grad()
        loss.backward()
        opt_model.step()

    scheduler.step()
    n = num_batches
    avg_mlm = total_mlm / n
    avg_mvc = total_mvc / n
    avg_dat = total_dat / n
    avg_dat_critic = total_dat_critic / n if (discriminator is not None and n_critic > 1) else 0.0
    avg_ecs = total_ecs / n
    recon = avg_mlm + avg_mvc

    mvc_str = f"MVC={avg_mvc:.4f}" if total_mvc > 0 else "MVC=inactive"
    if discriminator is not None:
        dat_str = f"DAT={-avg_dat:.4f} (x{dat_weight}={-dat_weight * avg_dat:.4f})"
        if n_critic > 1:
            dat_str += f" | critic={-avg_dat_critic:.4f}"
    else:
        dat_str = "DAT=inactive"
    ecs_str = f"ECS={avg_ecs:.4f} (x{ecs_weight}={ecs_weight * avg_ecs:.4f})" if ecs_threshold > 0 else "ECS=inactive"

    log.info(
        "Epoch %d train | recon=%.4f | MLM=%.4f | %s | %s | %s | lr=%.6f",
        epoch,
        recon,
        avg_mlm,
        mvc_str,
        dat_str,
        ecs_str,
        scheduler.get_last_lr()[0],
    )
    metrics = {
        "train/recon_loss": recon,
        "train/mlm_loss": avg_mlm,
        "train/mvc_loss": avg_mvc,
        "train/dat_loss": -avg_dat,
        "train/ecs_loss": avg_ecs,
        "train/lr": scheduler.get_last_lr()[0],
    }
    if discriminator is not None and n_critic > 1:
        metrics["train/dat_critic_loss"] = -avg_dat_critic
    return metrics


def _evaluate(model, data_loader: DataLoader, device, mask_ratio, mask_value, pad_value, amp: bool = False) -> float:
    """Evaluate the model on validation data (MLM loss only)."""
    model.eval()
    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Validation", leave=False):
            gene_ids = batch_data["gene_ids"].to(device)
            values = batch_data["values"].to(device)
            target_values = values.clone()
            batch_size = gene_ids.shape[0]

            masked_values = random_mask_value(values, mask_ratio, mask_value, pad_value)
            src_key_padding_mask = gene_ids.eq(model.pad_token_id)
            conditions = _get_dummy_conditions(model, batch_size, device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output_dict = _model_forward(model, gene_ids, masked_values, src_key_padding_mask, conditions)
                output_values = output_dict["mlm_output"]
                masked_positions = masked_values == mask_value
                loss = nn.functional.mse_loss(output_values[masked_positions], target_values[masked_positions])

            total_loss += loss.item() * len(gene_ids)
            total_num += len(gene_ids)

    return total_loss / total_num


def _get_embeddings(
    model,
    all_counts,
    gene_ids,
    vocab,
    batch_size,
    device,
    max_seq_len,
    pad_value,
    amp: bool = False,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Get cell embeddings from the model for ALL cells."""
    model.eval()
    embeddings = []

    tokenized = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_value=pad_value,
        cls_token_id=vocab["<cls>"],
        include_zero_gene=True,
    )

    all_gene_ids = tokenized["genes"]
    all_values = tokenized["values"]

    with torch.no_grad():
        for i in tqdm(range(0, len(all_gene_ids), batch_size), desc="Embedding", leave=False):
            batch_genes = all_gene_ids[i : i + batch_size].to(device)
            batch_values = all_values[i : i + batch_size].to(device)
            src_key_padding_mask = batch_genes.eq(vocab["<pad>"])
            current_batch_size = batch_genes.shape[0]

            conditions = _get_dummy_conditions(model, current_batch_size, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                cell_emb = _encode_cls(model.model, batch_genes, batch_values, src_key_padding_mask, conditions)
            embeddings.append(cell_emb.float().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    if l2_normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def finetune_cancerfoundation(adata, cfg):
    """Finetune CancerFoundation on adata and return embeddings for ALL cells.

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
    from models.base import FinetuneResult

    ft_cfg = cfg.task.finetune
    seed = ft_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Finetuning CancerFoundation on device: %s", device)

    # --- Load CancerFoundation model and vocab ---
    source_path = cfg.model.get("source_path", None)
    if source_path is not None:
        source_path = str(Path(source_path).resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    # Stub bionemo
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
    from cancerfoundation.model.module import WassersteinDiscriminator

    torch.serialization.add_safe_globals([LossType])

    model_dir = Path(cfg.model.pretrained_path)
    vocab_file = model_dir / "vocab.json"
    checkpoint_filename = cfg.model.get("checkpoint_filename", None)
    if checkpoint_filename:
        ckpt = model_dir / checkpoint_filename
    else:
        ckpts = list(model_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {model_dir}")
        ckpt = ckpts[0]

    with open(vocab_file) as f:
        vocab = json.load(f)

    model = CancerFoundation.load_from_checkpoint(str(ckpt), vocab=vocab, strict=True)
    model = model.to(device)
    amp = ft_cfg.get("amp", False)
    if amp:
        log.info("Using bfloat16 autocast (activations in bf16, weights in fp32)")

    max_seq_len = model.max_seq_len
    n_bins = model.n_bins
    normalise_bins = model.normalise_bins
    pad_value = model.pad_value
    mask_value = model.mask_value
    d_model = model.model.d_model

    log.info("Model loaded: max_seq_len=%d, n_bins=%d, d_model=%d", max_seq_len, n_bins, d_model)

    # --- Preprocess ---
    n_hvg = ft_cfg.get("n_hvg", cfg.model.get("n_top_genes", 2000))
    X_binned, gene_ids, genes = preprocess_data(adata, vocab, n_hvg=n_hvg, n_bins=n_bins, normalise_bins=normalise_bins)

    # Sequence length = n_genes + 1 (CLS). Use actual gene count instead of
    # model.max_seq_len so all selected HVGs are kept (no random truncation).
    actual_seq_len = len(gene_ids) + 1

    # Batch labels from metadata
    batch_key = cfg.task.metadata.batch_key
    batch_ids = adata.obs[batch_key].astype("category").cat.codes.values

    # --- Train/val split ---
    val_fraction = ft_cfg.get("val_fraction", 0.1)
    train_data, valid_data, train_batch_labels, valid_batch_labels = train_test_split(
        X_binned, batch_ids, test_size=val_fraction, shuffle=True, random_state=seed
    )
    log.info("Train: %d cells, Valid: %d cells", len(train_data), len(valid_data))

    # --- Tokenize ---
    tokenized_train = tokenize_and_pad_batch(
        train_data, gene_ids, max_len=actual_seq_len, vocab=vocab, pad_value=pad_value, cls_token_id=vocab["<cls>"]
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data, gene_ids, max_len=actual_seq_len, vocab=vocab, pad_value=pad_value, cls_token_id=vocab["<cls>"]
    )

    train_dataset = FinetuneDataset(
        tokenized_train["genes"], tokenized_train["values"], torch.from_numpy(train_batch_labels).long()
    )
    valid_dataset = FinetuneDataset(
        tokenized_valid["genes"], tokenized_valid["values"], torch.from_numpy(valid_batch_labels).long()
    )

    batch_size = ft_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Setup DAT discriminator ---
    discriminator = None
    do_dat = ft_cfg.get("do_dat", True)
    if do_dat:
        n_batches = len(np.unique(batch_ids))
        dat_scale = ft_cfg.get("dat_scale", 1.0)
        discriminator = WassersteinDiscriminator(d_model=d_model, n_cls=n_batches, scale=dat_scale).to(device)
        log.info("DAT enabled: %d batches, scale=%.1f", n_batches, dat_scale)

    # --- Optimizers and scheduler ---
    lr = ft_cfg.get("lr", 1e-4)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    schedule_ratio = ft_cfg.get("schedule_ratio", 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(opt_model, 1, gamma=schedule_ratio)

    opt_disc = None
    if discriminator is not None:
        opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr, eps=1e-8)

    # --- Training loop ---
    epochs = ft_cfg.get("epochs", 15)
    mask_ratio = ft_cfg.get("mask_ratio", 0.4)
    dat_weight = ft_cfg.get("dat_weight", 2.0)
    n_critic = ft_cfg.get("n_critic", 1)
    ecs_threshold = ft_cfg.get("ecs_threshold", 0.0)
    ecs_weight = ft_cfg.get("ecs_weight", 10.0)
    patience = ft_cfg.get("patience", 5)

    # --- Wandb ---
    use_wandb = ft_cfg.get("wandb", False)
    if use_wandb:
        try:
            import wandb
        except ImportError:
            log.warning("wandb requested but not installed in this environment; disabling")
            use_wandb = False
    if use_wandb:
        wandb_project = ft_cfg.get("wandb_project", "scFM-eval")
        wandb_name = ft_cfg.get("wandb_name", None)
        if not wandb_name:
            dataset = cfg.task.get("dataset_name", "unknown")
            parts = [f"cf-ft_{dataset}"]
            if do_dat:
                dat_part = f"dat-w{dat_weight}-s{ft_cfg.get('dat_scale', 1.0)}"
                if n_critic > 1:
                    dat_part += f"-nc{n_critic}"
                parts.append(dat_part)
            if ecs_threshold > 0:
                parts.append(f"ecs-t{ecs_threshold}-w{ecs_weight}")
            parts.append(f"lr{lr}_ep{epochs}_bs{batch_size}")
            wandb_name = "_".join(parts)
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "model": cfg.model.get("name", "cancerfoundation"),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "mask_ratio": mask_ratio,
                "n_hvg": n_hvg,
                "do_dat": do_dat,
                "dat_weight": dat_weight,
                "n_critic": n_critic,
                "ecs_threshold": ecs_threshold,
                "ecs_weight": ecs_weight,
                "patience": patience,
                "val_fraction": val_fraction,
                "schedule_ratio": schedule_ratio,
                "amp": amp,
                "n_train": len(train_data),
                "n_valid": len(valid_data),
                "n_total": len(X_binned),
                "d_model": d_model,
                "seq_len": actual_seq_len,
            },
        )

    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            opt_model,
            opt_disc,
            scheduler,
            device,
            mask_ratio,
            mask_value,
            pad_value,
            epoch,
            discriminator=discriminator,
            dat_weight=dat_weight,
            n_critic=n_critic,
            ecs_threshold=ecs_threshold,
            ecs_weight=ecs_weight,
            amp=amp,
        )

        val_loss = _evaluate(model, valid_loader, device, mask_ratio, mask_value, pad_value, amp=amp)
        log.info("Epoch %d val loss: %.4f", epoch, val_loss)

        if use_wandb:
            wandb.log({**train_metrics, "val/mlm_loss": val_loss, "epoch": epoch})

    wandb_run_id = None
    if use_wandb:
        wandb_run_id = wandb.run.id
        wandb.finish()

    # --- Get embeddings for ALL cells using final model ---
    log.info("Computing embeddings for all %d cells using final model (epoch %d)", len(X_binned), epochs)
    l2_normalize = cfg.model.get("l2_normalize", False)
    embeddings = _get_embeddings(
        model,
        X_binned,
        gene_ids,
        vocab,
        batch_size,
        device,
        actual_seq_len,
        pad_value,
        amp=amp,
        l2_normalize=l2_normalize,
    )

    return FinetuneResult(
        cell_embeddings=embeddings,
        embedding_dim=d_model,
        best_val_loss=val_loss,
        best_epoch=epochs,
        wandb_run_id=wandb_run_id,
        wandb_project=wandb_project if use_wandb else None,
    )
