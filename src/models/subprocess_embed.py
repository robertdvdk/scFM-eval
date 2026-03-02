"""Subprocess-based embedding orchestration.

Serializes AnnData to disk, spawns a Python subprocess inside the model's
isolated venv, and reads back the embedding results. This allows models with
conflicting dependencies to run without polluting the main environment.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .base import EmbeddingResult, FinetuneResult

log = logging.getLogger(__name__)

# Project src/ directory — added to PYTHONPATH so the subprocess can
# ``from models import get_model`` without installing the package.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)


def subprocess_embed(
    adata,
    cfg: DictConfig,
    env_path: str,
    timeout: int = 3600,
) -> EmbeddingResult:
    """Compute embeddings by spawning a subprocess in an isolated venv.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data to embed.
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.model``).
    env_path : str
        Path to the isolated venv (e.g. ``/app/envs/scgpt/.venv``).
        Must contain ``bin/python``.
    timeout : int
        Maximum seconds to wait for the subprocess (default 3600).

    Returns
    -------
    EmbeddingResult
        Cell (and optionally gene) embeddings produced by the model.

    Raises
    ------
    FileNotFoundError
        If *env_path* does not contain a usable Python interpreter.
    RuntimeError
        If the subprocess exits with a non-zero code or times out.
    """
    env_path = Path(env_path)
    python_bin = env_path / "bin" / "python"

    if not python_bin.is_file():
        raise FileNotFoundError(
            f"Python interpreter not found at {python_bin}. Create the environment first: bash envs/setup_envs.sh"
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"scfm_{uuid.uuid4().hex[:8]}_"))
    input_path = tmp_dir / "input.h5ad"
    config_path = tmp_dir / "config.yaml"
    output_path = tmp_dir / "result.npz"

    try:
        # Serialize inputs
        log.info("Writing subprocess inputs to %s", tmp_dir)
        adata.write_h5ad(input_path)
        OmegaConf.save(cfg, str(config_path))

        # Build environment: inherit parent env, override PYTHONPATH and PATH
        env = os.environ.copy()
        env["PYTHONPATH"] = _SRC_DIR
        env["PATH"] = str(env_path / "bin") + os.pathsep + env.get("PATH", "")

        cmd = [
            str(python_bin),
            "-m",
            "models.subprocess_worker",
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--mode",
            "embed",
        ]

        log.info("Spawning subprocess: %s", " ".join(cmd[:4]) + " ...")
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            timeout=timeout,
        )

        if proc.returncode != 0:
            stdout_tail = proc.stdout[-2000:] if proc.stdout else "(no output)"
            raise RuntimeError(
                f"Subprocess exited with code {proc.returncode}.\nstdout (last 2000 chars):\n{stdout_tail}"
            )

        if proc.stdout:
            log.debug("Subprocess stdout:\n%s", proc.stdout[-500:])

        # Read results
        if not output_path.is_file():
            raise RuntimeError(f"Subprocess exited successfully but output file is missing: {output_path}")

        data = np.load(output_path)
        cell_embeddings = data["cell_embeddings"]
        gene_embeddings = data.get("gene_embeddings")
        embedding_dim = int(data["embedding_dim"])

        log.info(
            "Subprocess completed: cell_embeddings shape %s, embedding_dim %d",
            cell_embeddings.shape,
            embedding_dim,
        )

        return EmbeddingResult(
            cell_embeddings=cell_embeddings,
            gene_embeddings=gene_embeddings,
            embedding_dim=embedding_dim,
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Subprocess timed out after {timeout}s. "
            f"The model may have run out of memory, or you can increase "
            f"model.subprocess_timeout in your config."
        ) from exc

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def subprocess_finetune(
    adata,
    cfg: DictConfig,
    env_path: str,
    timeout: int = 7200,
) -> FinetuneResult:
    """Finetune a model by spawning a subprocess in an isolated venv.

    Same pattern as :func:`subprocess_embed` but passes ``--mode finetune``
    and expects ``FinetuneResult`` fields in the output .npz.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data (all cells). The subprocess receives the full dataset;
        train/val splitting is handled inside the finetune function.
    cfg : DictConfig
        Full Hydra config (must contain ``cfg.model`` and ``cfg.task.finetune``).
    env_path : str
        Path to the isolated venv (e.g. ``/app/envs/cancerfoundation/.venv``).
    timeout : int
        Maximum seconds to wait for the subprocess (default 7200).

    Returns
    -------
    FinetuneResult
        Cell embeddings for ALL cells plus training metadata.
    """
    env_path = Path(env_path)
    python_bin = env_path / "bin" / "python"

    if not python_bin.is_file():
        raise FileNotFoundError(
            f"Python interpreter not found at {python_bin}. Create the environment first: bash envs/setup_envs.sh"
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"scfm_ft_{uuid.uuid4().hex[:8]}_"))
    input_path = tmp_dir / "input.h5ad"
    config_path = tmp_dir / "config.yaml"
    output_path = tmp_dir / "result.npz"

    try:
        log.info("Writing finetune subprocess inputs to %s", tmp_dir)
        adata.write_h5ad(input_path)
        OmegaConf.save(cfg, str(config_path))

        env = os.environ.copy()
        env["PYTHONPATH"] = _SRC_DIR
        env["PATH"] = str(env_path / "bin") + os.pathsep + env.get("PATH", "")

        cmd = [
            str(python_bin),
            "-m",
            "models.subprocess_worker",
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--mode",
            "finetune",
        ]

        log.info("Spawning finetune subprocess: %s", " ".join(cmd[:4]) + " ...")
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=None,  # inherit parent stdout so progress is visible
            stderr=None,
            timeout=timeout,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"Finetune subprocess exited with code {proc.returncode}.")

        if not output_path.is_file():
            raise RuntimeError(f"Finetune subprocess exited successfully but output file is missing: {output_path}")

        data = np.load(output_path, allow_pickle=True)
        cell_embeddings = data["cell_embeddings"]
        embedding_dim = int(data["embedding_dim"])
        best_val_loss = float(data["best_val_loss"])
        best_epoch = int(data["best_epoch"])
        wandb_run_id = str(data["wandb_run_id"]) if "wandb_run_id" in data else None
        wandb_project = str(data["wandb_project"]) if "wandb_project" in data else None

        log.info(
            "Finetune subprocess completed: cell_embeddings shape %s, "
            "embedding_dim %d, best_val_loss %.4f, best_epoch %d",
            cell_embeddings.shape,
            embedding_dim,
            best_val_loss,
            best_epoch,
        )

        return FinetuneResult(
            cell_embeddings=cell_embeddings,
            embedding_dim=embedding_dim,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            wandb_run_id=wandb_run_id,
            wandb_project=wandb_project,
        )

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Finetune subprocess timed out after {timeout}s. "
            f"The model may need more time; increase timeout in your config."
        ) from exc

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
