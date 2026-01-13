# Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics
# by Mejia et al., 2025

import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig

from metrics import mse, r2_score_on_deltas, wmse

log = logging.getLogger(__name__)


class GenePerturbationRunner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.adata = None
        self.weights_vsrest = None
        self.weights_vscontrol = None

    def _load_data(self) -> None:
        """Load the ground truth dataset and precomputed weights."""
        # Load processed norman19 dataset
        data_path = hydra.utils.to_absolute_path(self.cfg.task.paths.ground_truth)
        log.info(f"Loading ground truth data from {data_path}")
        self.adata = sc.read_h5ad(data_path)

        # Validate required columns
        if "condition" not in self.adata.obs.columns:
            raise ValueError("Ground truth data must have 'condition' column in obs")

        log.info(f"Loaded {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        log.info(f"Perturbations: {self.adata.obs['condition'].nunique()}")

        # Load precomputed weights
        weights_dir = Path(data_path).parent

        vsrest_path = weights_dir / "norman19_weights_vsrest.pkl"
        vscontrol_path = weights_dir / "norman19_weights_vscontrol.pkl"

        if vsrest_path.exists():
            log.info(f"Loading vsrest weights from {vsrest_path}")
            with open(vsrest_path, "rb") as f:
                self.weights_vsrest = pickle.load(f)
        else:
            log.warning(f"vsrest weights not found at {vsrest_path}")

        if vscontrol_path.exists():
            log.info(f"Loading vscontrol weights from {vscontrol_path}")
            with open(vscontrol_path, "rb") as f:
                self.weights_vscontrol = pickle.load(f)
        else:
            log.warning(f"vscontrol weights not found at {vscontrol_path}")

    def _load_submission(self) -> sc.AnnData:
        """Load and validate the submission dataset."""
        submission_path = hydra.utils.to_absolute_path(self.cfg.task.paths.submission)
        log.info(f"Loading submission data from {submission_path}")
        submission = sc.read_h5ad(submission_path)

        # Validate alignment with ground truth
        if not self.adata.obs_names.equals(submission.obs_names):
            raise ValueError("Submission cell IDs must match ground truth order exactly")

        if not self.adata.var_names.equals(submission.var_names):
            raise ValueError("Submission gene names must match ground truth order exactly")

        log.info("Submission data validated successfully")
        return submission

    def _compute_metrics_for_perturbation(
        self, pert: str, submission: sc.AnnData, baseline_mode: str = "control"
    ) -> dict:
        """
        Compute metrics for a single perturbation.

        Args:
            pert: Perturbation name
            submission: Submission AnnData object
            baseline_mode: Either 'control' or 'mean' for baseline calculation

        Returns:
            Dictionary with metric values
        """
        # Get ground truth mean for this perturbation
        pert_mask = self.adata.obs["condition"] == pert
        gt_pert_mean = self.adata[pert_mask].X.mean(axis=0).A1

        # Get submission mean for this perturbation
        sub_pert_mean = submission[pert_mask].X.mean(axis=0).A1

        # Calculate baseline (control or dataset mean)
        if baseline_mode == "control":
            control_mask = self.adata.obs["condition"] == "control"
            baseline_mean = self.adata[control_mask].X.mean(axis=0).A1
            weight_key = self.weights_vscontrol
        else:  # mean
            all_pert_means = []
            for p in self.adata.obs["condition"].unique():
                if p != "control":
                    p_mask = self.adata.obs["condition"] == p
                    p_mean = self.adata[p_mask].X.mean(axis=0).A1
                    all_pert_means.append(p_mean)
            baseline_mean = np.mean(all_pert_means, axis=0)
            weight_key = self.weights_vsrest

        # Get weights for this perturbation
        weights = None
        if weight_key is not None and pert in weight_key:
            weights = weight_key[pert].values

        # Calculate deltas
        delta_gt = gt_pert_mean - baseline_mean
        delta_sub = sub_pert_mean - baseline_mean

        # Calculate metrics
        metrics = {
            "perturbation": pert,
            "n_cells": pert_mask.sum(),
            "mse": mse(gt_pert_mean, sub_pert_mean),
            "r2_delta": r2_score_on_deltas(delta_gt, delta_sub),
        }

        if weights is not None:
            metrics["wmse"] = wmse(gt_pert_mean, sub_pert_mean, weights)
            metrics["wr2_delta"] = r2_score_on_deltas(delta_gt, delta_sub, weights)
        else:
            log.warning(f"No weights found for perturbation {pert}")
            metrics["wmse"] = np.nan
            metrics["wr2_delta"] = np.nan

        return metrics

    def run(self) -> dict:
        """Run the gene perturbation evaluation."""
        log.info("Starting gene perturbation evaluation")

        # Load data
        self._load_data()
        submission = self._load_submission()

        # Get perturbations to evaluate (exclude control)
        perturbations = [p for p in self.adata.obs["condition"].unique() if p != "control"]
        log.info(f"Evaluating {len(perturbations)} perturbations")

        # Compute metrics for each perturbation
        results = []
        for pert in perturbations:
            log.info(f"Evaluating {pert}")
            metrics = self._compute_metrics_for_perturbation(pert, submission, baseline_mode="mean")
            results.append(metrics)

        # Create results dataframe
        results_df = pd.DataFrame(results)

        # Compute aggregate metrics (mean across perturbations)
        aggregate_metrics = {
            "mean_mse": results_df["mse"].mean(),
            "mean_wmse": results_df["wmse"].mean(),
            "mean_r2_delta": results_df["r2_delta"].mean(),
            "mean_wr2_delta": results_df["wr2_delta"].mean(),
            "median_mse": results_df["mse"].median(),
            "median_wmse": results_df["wmse"].median(),
            "median_r2_delta": results_df["r2_delta"].median(),
            "median_wr2_delta": results_df["wr2_delta"].median(),
        }

        log.info("Evaluation complete")
        log.info(f"Mean MSE: {aggregate_metrics['mean_mse']:.6f}")
        log.info(f"Mean WMSE: {aggregate_metrics['mean_wmse']:.6f}")
        log.info(f"Mean R² (delta): {aggregate_metrics['mean_r2_delta']:.6f}")
        log.info(f"Mean Weighted R² (delta): {aggregate_metrics['mean_wr2_delta']:.6f}")

        return {
            "aggregate": aggregate_metrics,
            "per_perturbation": results_df.to_dict(orient="records"),
        }
