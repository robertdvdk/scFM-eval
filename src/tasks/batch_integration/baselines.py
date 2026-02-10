import logging

import harmonypy as hm
import numpy as np
import scanpy as sc
import scvi
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def run_harmony(
    adata: sc.AnnData, batch_key: str, n_pcs: int = 30, n_hvgs: int = 2000, **kwargs
) -> tuple[str, np.ndarray]:
    """Run Harmony batch correction on pre-normalized, log-transformed data.

    Performs HVG selection, scaling, and PCA, then applies Harmony integration.
    Assumes adata.X is already normalized and log1p-transformed.
    """
    adata = adata.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, batch_key=batch_key)
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)

    # Call harmonypy directly to avoid scanpy wrapper incompatibility with PyTorch backend
    # Prevent duplicate logs: harmonypy's own handler + Hydra's root handler both fire
    logging.getLogger("harmonypy").propagate = False
    harmony_out = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key)
    Z = harmony_out.Z_corr
    if hasattr(Z, "detach"):  # PyTorch tensor (possibly on GPU)
        Z = Z.detach().cpu().numpy()
    Z = np.asarray(Z)
    # Z_corr is stored as (n_pcs, n_cells); transpose to (n_cells, n_pcs)
    if Z.shape[0] != adata.n_obs:
        Z = Z.T

    return ("Harmony", Z)


def run_scvi(
    adata: sc.AnnData, batch_key: str, label_key: str, n_hvgs: int = 2000, n_latent: int = 30, max_epochs: int = 400
) -> tuple[str, np.ndarray]:
    """Run scVI batch integration.

    Trains a variational autoencoder on raw counts with batch as a covariate.
    Assumes adata.X is log-normalized and adata.layers['counts'] has raw counts.
    """
    adata = adata.copy()

    # Select HVGs on log-normalized data
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, batch_key=batch_key)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Swap to raw counts for scVI
    adata.X = adata.layers["counts"]

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, layer=None)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs)

    return ("scVI", model.get_latent_representation())


def run_scanvi(
    adata: sc.AnnData,
    batch_key: str,
    label_key: str,
    n_hvgs: int = 2000,
    n_latent: int = 30,
    max_epochs_scvi: int = 400,
    max_epochs_scanvi: int = 20,
) -> tuple[str, np.ndarray]:
    """Run scANVI semi-supervised batch integration.

    First trains scVI, then initializes scANVI from it with cell type labels.
    Assumes adata.X is log-normalized and adata.layers['counts'] has raw counts.
    """
    adata = adata.copy()

    # Select HVGs on log-normalized data
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, batch_key=batch_key)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Swap to raw counts for scVI
    adata.X = adata.layers["counts"]

    # Train scVI as initialization
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, layer=None)
    scvi_model = scvi.model.SCVI(adata, n_latent=n_latent)
    scvi_model.train(max_epochs=max_epochs_scvi)

    # Initialize and train scANVI from the scVI model
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, labels_key=label_key, unlabeled_category="__unlabeled__"
    )
    scanvi_model.train(max_epochs=max_epochs_scanvi)

    return ("scANVI", scanvi_model.get_latent_representation())


BASELINE_REGISTRY: dict[str, callable] = {
    "harmony": run_harmony,
    "scvi": run_scvi,
    "scanvi": run_scanvi,
}


def run_baselines(adata: sc.AnnData, baseline_configs: list[DictConfig], batch_key: str, label_key: str) -> list[str]:
    """Run all configured baseline methods and add their embeddings to adata.obsm.

    Returns the list of obsm keys that were added.
    """
    added_keys = []

    for baseline_cfg in baseline_configs:
        method = baseline_cfg["method"]
        if method not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline method: {method}. Available: {list(BASELINE_REGISTRY.keys())}")

        # Extract kwargs (everything except 'method')
        kwargs = {k: v for k, v in baseline_cfg.items() if k != "method"}

        log.info(f"Running baseline: {method} with params: {kwargs}")
        name, embedding = BASELINE_REGISTRY[method](adata, batch_key=batch_key, label_key=label_key, **kwargs)

        obsm_key = name
        adata.obsm[obsm_key] = embedding
        added_keys.append(obsm_key)
        log.info(f"Added baseline embedding: {obsm_key} with shape {embedding.shape}")

    return added_keys
