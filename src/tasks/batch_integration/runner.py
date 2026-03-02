import logging
import os
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

from .baselines import run_baselines

log = logging.getLogger(__name__)


class BatchIntegrationRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _load_data(self):
        # Resolve paths to absolute
        ground_truth_path = hydra.utils.to_absolute_path(
            self.cfg.task.data.data_root + self.cfg.task.data.ground_truth + ".h5ad"
        )

        log.info(f"Loading ground truth from: {ground_truth_path}")

        # Load data
        self.adata = sc.read_h5ad(ground_truth_path)

        # Build submission paths from model_name list and dataset_name
        model_names = self.cfg.task.model_name
        if model_names is None:
            model_names = []
        elif not isinstance(model_names, (list, ListConfig)):
            model_names = [model_names]
        else:
            model_names = [m for m in model_names if m is not None]
        dataset_name = self.cfg.task.dataset_name

        # Track embedding keys for benchmarker
        self.embedding_obsm_keys = []

        # Collect foundation models to evaluate.
        # 1) Single model from CLI: model=scgpt_human
        # 2) Multiple models from task config: task.foundation_models=[scgpt_human, ...]
        fm_entries: list[tuple[str, DictConfig]] = []

        if getattr(self.cfg, "model", None) is not None:
            fm_entries.append((self.cfg.model.name, self.cfg))

        for fm_name in self.cfg.task.get("foundation_models", []):
            model_cfg_path = hydra.utils.to_absolute_path(f"configs/model/{fm_name}.yaml")
            model_cfg = OmegaConf.load(model_cfg_path)
            base_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
            merged_cfg = OmegaConf.merge(base_cfg, model_cfg)
            fm_entries.append((fm_name, merged_cfg))

        # Compute embeddings for each foundation model
        for fm_name, run_cfg in fm_entries:
            env_path = run_cfg.model.get("env_path", None)

            if env_path is not None:
                from models.subprocess_embed import subprocess_embed

                result = subprocess_embed(
                    adata=self.adata,
                    cfg=run_cfg,
                    env_path=env_path,
                    timeout=run_cfg.model.get("subprocess_timeout", 3600),
                )
            else:
                from models import get_model

                device = "cuda" if torch.cuda.is_available() else "cpu"
                wrapper = get_model(run_cfg)
                wrapper.load_pretrained()
                wrapper.to(device)
                result = wrapper.embed(self.adata, batch_size=run_cfg.model.batch_size)

            obsm_key = f"{fm_name}"
            if fm_name == "cancerfoundation":
                obsm_key = "CancerFoundation"
            elif fm_name == "scgpt_human":
                obsm_key = "scGPT_Human"
            elif fm_name == "scgpt_pancancer":
                obsm_key = "scGPT_PanCancer"
            elif fm_name == "scfoundation":
                obsm_key = "scFoundation"
            else:
                raise ValueError
            self.adata.obsm[obsm_key] = result.cell_embeddings
            self.embedding_obsm_keys.append(obsm_key)
            log.info(f"Computed embeddings: {obsm_key} shape {result.cell_embeddings.shape}")

        # Load all submissions and add their embeddings to adata
        for model_name in model_names:
            submission_path = hydra.utils.to_absolute_path(
                self.cfg.task.data.data_root + f"submission/{model_name}_{dataset_name}.csv"
            )
            submission_name = f"{model_name}_{dataset_name}"

            log.info(f"Loading submission: {submission_name}")
            log.info(f"From: {submission_path}")

            submission_df = pd.read_csv(submission_path, index_col=0)

            # Validate alignment (fail fast)
            if not self.adata.obs_names.equals(pd.Index(submission_df.index.astype(str))):
                raise ValueError(
                    f"Cell ID mismatch in {submission_name}: Submission cell IDs must match Ground Truth order."
                )

            # Add embedding with unique key
            self.adata.obsm[submission_name] = submission_df.values
            self.embedding_obsm_keys.append(submission_name)

        # Run classical baselines if configured
        baselines_cfg = self.cfg.task.get("baselines", None)
        if baselines_cfg:
            batch_key = self.cfg.task.metadata.batch_key
            label_key = self.cfg.task.metadata.label_key
            baseline_keys = run_baselines(self.adata, baselines_cfg, batch_key, label_key)
            self.embedding_obsm_keys.extend(baseline_keys)

        log.info(f"Data shape: {self.adata.shape}")
        log.info(f"Evaluating {len(self.embedding_obsm_keys)} submissions: {self.embedding_obsm_keys}")

    def _plot_raw_umap(self, output_dir: Path):
        """Plot UMAP of unintegrated data (PCA on raw expression) for comparison."""
        batch_key = self.cfg.task.metadata.batch_key
        label_key = self.cfg.task.metadata.label_key
        dataset_name = self.cfg.task.dataset_name

        log.info("Computing UMAP for unintegrated data")
        sc.pp.neighbors(self.adata)
        sc.tl.umap(self.adata)

        fig = sc.pl.umap(self.adata, color=[batch_key, label_key], return_fig=True, show=False, wspace=0.5)
        fig.savefig(output_dir / f"{dataset_name}_unintegrated.png", dpi=150, bbox_inches="tight")
        log.info("Saved unintegrated UMAP plot")

    def _plot_umaps(self, output_dir: Path):
        """Compute and save UMAP plots for each embedding, colored by batch and cell type."""
        batch_key = self.cfg.task.metadata.batch_key
        label_key = self.cfg.task.metadata.label_key
        dataset_name = self.cfg.task.dataset_name

        for obsm_key in self.embedding_obsm_keys:
            model_type = obsm_key.removesuffix(f"_{dataset_name}")
            log.info(f"Computing UMAP for {model_type}")

            self.adata.obsm["X_emb"] = self.adata.obsm[obsm_key]
            sc.pp.neighbors(self.adata, use_rep="X_emb")
            sc.tl.umap(self.adata)

            fig = sc.pl.umap(self.adata, color=[batch_key, label_key], return_fig=True, show=False, wspace=0.5)
            fig_path = output_dir / f"{dataset_name}_{model_type}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            log.info(f"Saved UMAP plot: {fig_path}")

    def _plot_results_table(self, bm: Benchmarker, save_dir: Path):
        """Custom results table with a wider Method column so long names fit."""
        _METRIC_TYPE = "Metric Type"
        _AGGREGATE_SCORE = "Aggregate score"

        num_embeds = len(bm._embedding_obsm_keys)

        def cmap_fn(col_data):
            return normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)

        df = bm.get_results(min_max_scale=False)
        plot_df = df.drop(_METRIC_TYPE, axis=0)

        if bm._batch_correction_metrics is not None and bm._bio_conservation_metrics is not None:
            sort_col = "Total"
        elif bm._batch_correction_metrics is not None:
            sort_col = "Batch correction"
        else:
            sort_col = "Bio conservation"
        plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index

        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]

        column_definitions = [
            ColumnDefinition("Method", width=2.5, textprops={"ha": "left", "weight": "bold", "fontsize": 8}),
        ]
        column_definitions += [
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for col in other_cols
        ]
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]

        with mpl.rc_context({"svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
            Table(
                plot_df,
                cell_kw={"linewidth": 0, "edgecolor": "k"},
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": 10, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)
        plt.close(fig)

    def run(self) -> dict[str, pd.DataFrame]:
        log.info(f"Running task: {self.cfg.task.name}")

        self._load_data()

        # Configure metrics
        bio_conservation = BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
        batch_correction = BatchCorrection(pcr_comparison=False)

        # Run benchmark on ALL embeddings at once
        bm = Benchmarker(
            self.adata,
            batch_key=self.cfg.task.metadata.batch_key,
            label_key=self.cfg.task.metadata.label_key,
            embedding_obsm_keys=self.embedding_obsm_keys,  # Pass all embedding keys
            n_jobs=self.cfg.task.get("n_jobs", 6),
            bio_conservation_metrics=bio_conservation,
            batch_correction_metrics=batch_correction,
            solver="randomized",
        )

        bm.benchmark()
        result = bm.get_results(min_max_scale=False)
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        self._plot_results_table(bm, output_dir)

        self._plot_raw_umap(output_dir)
        self._plot_umaps(output_dir)

        log.info(f"Benchmark results:\n{result.to_string()}")

        # Return structured results
        return {"results": result}
