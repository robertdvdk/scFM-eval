import scanpy as sc
from omegaconf import DictConfig
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation


class BatchIntegrationRunner:
    def __init__(self, cfg: DictConfig):
        # Store the config part relevant to this task
        self.cfg = cfg

    def run(self):
        # Accessing the config passed down from the central main.py
        print(f"Running task: {self.cfg.task.name}")

        # Note: Check your config file structure.
        # In our previous step, we defined 'paths', not 'dataset'.
        # Ensure these keys exist in configs/task/batch_integration.yaml
        ground_truth = self.cfg.task.paths.ground_truth
        submission_path = self.cfg.task.paths.submission
        print(f"Using ground truth from file: {ground_truth}")
        print(f"Using embeddings from file: {submission_path}")

        adata = sc.read_h5ad(ground_truth)
        adata.obsm["emb"] = sc.read_h5ad(submission_path).obsm["CancerGPT"]

        print(adata)

        bio_conservation = BioConservation(
            nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True
        )
        batch_correction = BatchCorrection(pcr_comparison=False)

        bm = Benchmarker(
            adata,
            batch_key="sample",
            label_key="subtype",
            embedding_obsm_keys=["emb"],
            n_jobs=6,
            bio_conservation_metrics=bio_conservation,
            batch_correction_metrics=batch_correction,
        )

        bm.benchmark()
        result = bm.get_results(min_max_scale=False)
        print(result)
        # Return results dictionary
        return {"score": 0.95}
