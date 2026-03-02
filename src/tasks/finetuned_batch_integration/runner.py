"""Runner for finetuned batch integration evaluation.

Foundation models are finetuned (MLM + DAT) on the evaluation data via
subprocess isolation, then evaluated alongside trained baselines (scVI,
scANVI) using scib-metrics. Evaluation is transductive: embeddings are
computed for ALL cells.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

from tasks.batch_integration.baselines import run_baselines

log = logging.getLogger(__name__)


class FinetunedBatchIntegrationRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _load_data(self):
        ground_truth_path = hydra.utils.to_absolute_path(
            self.cfg.task.data.data_root + self.cfg.task.data.ground_truth + ".h5ad"
        )

        log.info(f"Loading ground truth from: {ground_truth_path}")
        self.adata = sc.read_h5ad(ground_truth_path)
        dataset_name = self.cfg.task.dataset_name

        self.embedding_obsm_keys = []

        # Collect foundation models to finetune
        fm_entries: list[tuple[str, DictConfig]] = []

        if getattr(self.cfg, "model", None) is not None:
            fm_entries.append((self.cfg.model.name, self.cfg))

        for fm_name in self.cfg.task.get("foundation_models", []):
            model_cfg_path = hydra.utils.to_absolute_path(f"configs/model/{fm_name}.yaml")
            model_cfg = OmegaConf.load(model_cfg_path)
            base_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
            merged_cfg = OmegaConf.merge(base_cfg, model_cfg)
            fm_entries.append((fm_name, merged_cfg))

        # Finetune each foundation model via subprocess
        self._wandb_run_id = None
        self._wandb_project = None
        for fm_name, run_cfg in fm_entries:
            env_path = run_cfg.model.get("env_path", None)

            if env_path is None:
                raise RuntimeError(
                    f"Finetuning requires subprocess isolation (env_path must be set in "
                    f"configs/model/{fm_name}.yaml). In-process finetuning is not supported."
                )

            from models.subprocess_embed import subprocess_finetune

            result = subprocess_finetune(
                adata=self.adata,
                cfg=run_cfg,
                env_path=env_path,
                timeout=run_cfg.model.get("subprocess_timeout", 7200),
            )

            obsm_key = f"{fm_name}_{dataset_name}"
            self.adata.obsm[obsm_key] = result.cell_embeddings
            self.embedding_obsm_keys.append(obsm_key)
            log.info(
                f"Finetuned {fm_name}: embeddings shape {result.cell_embeddings.shape}, "
                f"best_val_loss={result.best_val_loss:.4f}, best_epoch={result.best_epoch}"
            )

            if result.wandb_run_id is not None and self._wandb_run_id is None:
                self._wandb_run_id = result.wandb_run_id
                self._wandb_project = result.wandb_project

        # Run trained baselines (scVI, scANVI)
        baselines_cfg = self.cfg.task.get("baselines", None)
        if baselines_cfg:
            batch_key = self.cfg.task.metadata.batch_key
            label_key = self.cfg.task.metadata.label_key
            baseline_keys = run_baselines(self.adata, baselines_cfg, batch_key, label_key)
            self.embedding_obsm_keys.extend(baseline_keys)

        log.info(f"Data shape: {self.adata.shape}")
        log.info(f"Evaluating {len(self.embedding_obsm_keys)} submissions: {self.embedding_obsm_keys}")

    def _plot_umaps(self, output_dir: Path, wandb_run=None):
        """Compute and save UMAP plots for each embedding."""
        batch_key = self.cfg.task.metadata.batch_key
        label_key = self.cfg.task.metadata.label_key
        dataset_name = self.cfg.task.dataset_name

        for obsm_key in self.embedding_obsm_keys:
            model_type = obsm_key.removesuffix(f"_{dataset_name}")
            log.info(f"Computing UMAP for {model_type}")

            self.adata.obsm["X_emb"] = self.adata.obsm[obsm_key]
            sc.pp.neighbors(self.adata, use_rep="X_emb")
            sc.tl.umap(self.adata)

            fig = sc.pl.umap(self.adata, color=[batch_key, label_key], return_fig=True, show=False)
            fig_path = output_dir / f"{dataset_name}_{model_type}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved UMAP plot: {fig_path}")

            if wandb_run is not None:
                import wandb

                wandb_run.log({f"umap/{model_type}": wandb.Image(str(fig_path))})

    def run(self) -> dict[str, pd.DataFrame]:
        log.info(f"Running task: {self.cfg.task.name}")

        self._load_data()

        bio_conservation = BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
        batch_correction = BatchCorrection(pcr_comparison=False)

        bm = Benchmarker(
            self.adata,
            batch_key=self.cfg.task.metadata.batch_key,
            label_key=self.cfg.task.metadata.label_key,
            embedding_obsm_keys=self.embedding_obsm_keys,
            n_jobs=self.cfg.task.get("n_jobs", 6),
            bio_conservation_metrics=bio_conservation,
            batch_correction_metrics=batch_correction,
            solver="randomized",
        )

        bm.benchmark()
        result = bm.get_results(min_max_scale=False)
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        bm.plot_results_table(min_max_scale=False, save_dir=output_dir)

        # Resume wandb run from finetuning subprocess (if available)
        wandb_run = None
        if self._wandb_run_id is not None:
            try:
                import wandb

                wandb_run = wandb.init(
                    id=self._wandb_run_id,
                    project=self._wandb_project,
                    resume="must",
                )
                log.info(f"Resumed wandb run {self._wandb_run_id}")
            except Exception as e:
                log.warning(f"Failed to resume wandb run: {e}")

        self._plot_umaps(output_dir, wandb_run=wandb_run)

        if wandb_run is not None:
            # Log benchmark metrics
            for col in result.columns:
                for idx in result.index:
                    wandb_run.log({f"benchmark/{col}/{idx}": result.loc[idx, col]})
            wandb_run.finish()

        log.info(f"Benchmark results:\n{result.to_string()}")
        return {"results": result}
