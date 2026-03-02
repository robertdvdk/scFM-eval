"""CLI entry point executed inside an isolated model venv.

The main process spawns this script via subprocess. It loads config and input
data from disk, runs the model wrapper to compute embeddings, and writes
results back to an npz file.

Usage (invoked by subprocess_embed, not directly)::

    {env_path}/bin/python -m models.subprocess_worker \
        --input /tmp/scfm_xxx/input.h5ad \
        --config /tmp/scfm_xxx/config.yaml \
        --output /tmp/scfm_xxx/result.npz \
        --mode embed
"""

from __future__ import annotations

import argparse
import logging
import sys

import anndata
import numpy as np
from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="scFM-eval subprocess worker")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--config", required=True, help="Path to config .yaml file")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    parser.add_argument("--mode", default="embed", choices=["embed", "finetune"])
    args = parser.parse_args()

    # Configure logging so finetune progress is visible
    if args.mode == "finetune":
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Load config and data
    cfg = OmegaConf.load(args.config)
    adata = anndata.read_h5ad(args.input)

    if args.mode == "finetune":
        _run_finetune(cfg, adata, args.output)
    else:
        _run_embed(cfg, adata, args.output)


def _run_embed(cfg, adata, output_path: str) -> None:
    """Zero-shot embedding mode."""
    from models import get_model

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    wrapper = get_model(cfg)
    wrapper.load_pretrained()
    wrapper.to(device)
    result = wrapper.embed(adata, batch_size=cfg.model.batch_size)

    save_kwargs = {
        "cell_embeddings": result.cell_embeddings,
        "embedding_dim": np.array(result.embedding_dim),
    }
    if result.gene_embeddings is not None:
        save_kwargs["gene_embeddings"] = result.gene_embeddings

    np.savez(output_path, **save_kwargs)


def _run_finetune(cfg, adata, output_path: str) -> None:
    """Finetuning mode: train model on adata, embed ALL cells, save results."""
    model_name = cfg.model.name

    if model_name == "cancerfoundation":
        from models.finetune_cancerfoundation import finetune_cancerfoundation

        result = finetune_cancerfoundation(adata, cfg)
    elif model_name in ("scgpt", "scgpt_human", "scgpt_pancancer"):
        from models.finetune_scgpt import finetune_scgpt

        result = finetune_scgpt(adata, cfg)
    else:
        raise ValueError(
            f"Finetuning not implemented for model '{model_name}'. "
            f"Supported: cancerfoundation, scgpt, scgpt_human, scgpt_pancancer"
        )

    save_kwargs = {
        "cell_embeddings": result.cell_embeddings,
        "embedding_dim": np.array(result.embedding_dim),
        "best_val_loss": np.array(result.best_val_loss),
        "best_epoch": np.array(result.best_epoch),
    }
    if result.wandb_run_id is not None:
        save_kwargs["wandb_run_id"] = np.array(result.wandb_run_id)
    if result.wandb_project is not None:
        save_kwargs["wandb_project"] = np.array(result.wandb_project)
    np.savez(output_path, **save_kwargs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"subprocess_worker error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
