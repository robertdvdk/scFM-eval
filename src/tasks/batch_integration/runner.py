import logging
from pathlib import Path

import hydra
import pandas as pd
import scanpy as sc
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

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

        # Handle both single file and list of files
        submissions = self.cfg.task.data.submissions
        if not isinstance(submissions, (list, ListConfig)):
            submissions = [submissions]

        # Extract embedding key from config
        embedding_key = self.cfg.task.metadata.embedding_key

        # Track embedding keys for benchmarker
        self.embedding_obsm_keys = []

        # Load all submissions and add their embeddings to adata
        for submission_path in submissions:
            submission_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + submission_path + ".h5ad")
            submission_name = Path(submission_path).stem

            log.info(f"Loading submission: {submission_name}")
            log.info(f"From: {submission_path}")

            submission = sc.read_h5ad(submission_path)

            # Validate alignment (fail fast)
            if not self.adata.obs_names.equals(submission.obs_names):
                raise ValueError(
                    f"Cell ID mismatch in {submission_name}: Submission obs_names must match Ground Truth order."
                )

            # Check embedding exists
            if embedding_key not in submission.obsm:
                raise KeyError(f"Embedding key '{embedding_key}' not found in {submission_name}.obsm")

            # Add embedding with unique key
            self.adata.obsm[submission_name] = submission.obsm[embedding_key]
            self.embedding_obsm_keys.append(submission_name)

        log.info(f"Data shape: {self.adata.shape}")
        log.info(f"Evaluating {len(self.embedding_obsm_keys)} submissions: {self.embedding_obsm_keys}")

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
        bm.plot_results_table(min_max_scale=False, save_dir=Path(HydraConfig.get().runtime.output_dir))

        log.info(f"Benchmark results:\n{result}")

        # Return structured results
        return {"results": result}
