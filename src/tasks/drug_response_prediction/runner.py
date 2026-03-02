import glob
import logging
import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from loaders import get_drp_dataloaders
from loaders.drug_response_prediction import DataManager

from .model import DrugMLP, DualStreamModel

log = logging.getLogger(__name__)


class DrugResponsePredictionRunner:
    """Runner for drug response prediction evaluation task."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._validate_config()
        self._set_device()
        self._seed_everything()

    def _validate_config(self):
        required_paths = [
            ("task", "data", "drug_emb_path"),
            ("task", "data", "dose_response_path"),
            ("task", "data", "metadata"),
        ]
        for path_parts in required_paths:
            try:
                value = self.cfg
                for part in path_parts:
                    value = value[part]
                if value is None:
                    if path_parts[-1] == "metadata":
                        continue
                    raise ValueError(f"Missing config: {'.'.join(path_parts)}")
            except (KeyError, AttributeError) as err:
                if path_parts[-1] != "metadata":
                    raise ValueError(f"Missing config: {'.'.join(path_parts)}") from err

        has_submission = self.cfg.task.data.get("submission") is not None
        has_fm = len(self.cfg.task.get("foundation_models", [])) > 0
        has_baselines = self.cfg.task.get("run_baselines", False)
        if not has_submission and not has_fm and not has_baselines:
            raise ValueError(
                "At least one of task.data.submission, task.foundation_models, or task.run_baselines must be set"
            )

    def _set_device(self):
        gpu_id = self.cfg.task.get("gpu_id", -1)
        if gpu_id == -1 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            log.info("Using CPU")
        else:
            self.device = torch.device(f"cuda:{gpu_id}")
            log.info(f"Using GPU: {gpu_id}")

    def _seed_everything(self):
        seed = self.cfg.task.get("model_seed", 42)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        log.info(f"Random seed set to {seed}")

    def _create_model(self, cell_dim: int, drug_dim: int, is_graph: bool, model_cls=DualStreamModel):
        import inspect

        sig = inspect.signature(model_cls)
        if "is_graph" in sig.parameters:
            model = model_cls(cell_dim=cell_dim, drug_dim=drug_dim, is_graph=is_graph).to(self.device)
        else:
            model = model_cls(cell_dim=cell_dim, drug_dim=drug_dim).to(self.device)
        log.info(
            f"Model initialized ({model_cls.__name__}). "
            f"Cell Dim: {cell_dim}, Drug Dim: {drug_dim}, Graph Mode: {is_graph}"
        )
        return model

    def _train_epoch(self, train_loader, model, optimizer, criterion):
        model.train()
        running_loss = 0.0
        for cell_vec, drug_vec, target, _, _ in train_loader:
            cell_vec = cell_vec.to(self.device)
            target = target.to(self.device)

            # drug_vec is either Tensor or PyG Batch
            drug_vec = drug_vec.to(self.device)

            optimizer.zero_grad()
            out = model(cell_vec, drug_vec)
            loss = criterion(out.view(-1), target.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def _evaluate(self, data_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_drug_ids = []
        all_cell_ids = []

        with torch.no_grad():
            for cell_vec, drug_vec, target, d_idx, c_idx in data_loader:
                cell_vec = cell_vec.to(self.device)
                drug_vec = drug_vec.to(self.device)
                target = target.to(self.device)

                out = model(cell_vec, drug_vec)
                loss = criterion(out.view(-1), target.view(-1))
                total_loss += loss.item()

                all_preds.extend(out.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                all_drug_ids.extend(d_idx.cpu().numpy().flatten())
                all_cell_ids.extend(c_idx.cpu().numpy().flatten())

        df_res = pd.DataFrame({
            "pred": all_preds,
            "target": all_targets,
            "drug_idx": all_drug_ids,
            "cell_idx": all_cell_ids,
        })

        # Calculate Per-Drug Pearson
        drug_correlations = []
        for _, group in df_res.groupby("drug_idx"):
            if len(group) > 5:  # TODO remove?
                if group["pred"].std() < 1e-9 or group["target"].std() < 1e-9:  # TODO remove?
                    continue
                r, _ = pearsonr(group["pred"], group["target"])
                drug_correlations.append(r)
        mean_per_drug_r = np.mean(drug_correlations) if drug_correlations else 0.0

        # Calculate Per-Cell Pearson
        cell_correlations = []
        for _, group in df_res.groupby("cell_idx"):
            if len(group) > 5:
                if group["pred"].std() < 1e-9 or group["target"].std() < 1e-9:
                    continue
                r, _ = pearsonr(group["pred"], group["target"])
                cell_correlations.append(r)
        mean_per_cell_r = np.mean(cell_correlations) if cell_correlations else 0.0

        # Calculate Global Pearson
        if len(all_targets) > 1 and np.std(np.array(all_preds)) > 1e-9:
            global_r, _ = pearsonr(all_targets, all_preds)
        else:
            global_r = 0.0

        return (
            total_loss / len(data_loader),
            mean_per_drug_r,
            mean_per_cell_r,
            np.array(all_targets),
            np.array(all_preds),
            global_r,
        )

    def _run_single_split(
        self,
        cell_path: str,
        test_drugs: list[str] | None = None,
        val_drugs: list[str] | None = None,
        test_cells: list[str] | None = None,
        val_cells: list[str] | None = None,
        model_label: str = "",
        model_cls=DualStreamModel,
    ) -> dict[str, Any]:

        # Determine Mode for Logging
        split_mode = self.cfg.task.get("split_mode", "cancer_stratified")

        split_name = f"split_{split_mode}"

        # Pull Configs
        test_tissue = self.cfg.task.get("test_tissue")
        drug_prop = self.cfg.task.get("drug_proportion")

        log.info(f"Running split: {split_name} (Mode: {split_mode})")

        # Initialize Loader
        train_loader, val_loader, test_loader, dims = get_drp_dataloaders(
            cell_path=cell_path,
            drug_path=hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.drug_emb_path),
            matrix_path=hydra.utils.to_absolute_path(
                self.cfg.task.data.data_root + self.cfg.task.data.dose_response_path
            ),
            metadata_path=hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.metadata)
            if self.cfg.task.data.metadata
            else None,
            split_mode=split_mode,
            test_drugs=test_drugs,
            val_drugs=val_drugs,
            test_cells=test_cells,
            val_cells=val_cells,
            holdout_tissue=test_tissue,
            drug_prop=drug_prop,
            batch_size=self.cfg.task.get("batch_size", 256),
            num_workers=self.cfg.task.get("num_workers", 4),
            data_seed=self.cfg.task.get("data_seed", 42),
        )

        model = self._create_model(dims["cell_dim"], dims["drug_dim"], dims["is_graph"], model_cls=model_cls)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.task.get("learning_rate", 0.003))
        criterion = nn.MSELoss()
        eval_criterion = nn.L1Loss()

        epochs = self.cfg.task.get("epochs", 500)
        patience = self.cfg.task.get("patience", 10)
        best_val_metric = -float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, model, optimizer, criterion)
            val_mae, val_per_drug, val_per_cell, _, _, val_global = self._evaluate(val_loader, model, eval_criterion)

            current_metric = val_per_drug if split_mode == "cold_drug" else val_global

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                log.info(
                    f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} |"
                    f" Val Per-Drug PCC: {val_per_drug:.4f} | Val Per-Cell PCC: {val_per_cell:.4f}"
                    f" | Val Global PCC: {val_global:.4f}"
                )

            if patience_counter >= patience:
                log.info(
                    f"Early stopping at epoch {epoch + 1} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} |"
                    f" Val Per-Drug PCC: {val_per_drug:.4f} | Val Per-Cell PCC: {val_per_cell:.4f}"
                    f" | Val Global PCC: {val_global:.4f}"
                )
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        test_mae, test_pd_pcc, test_pc_pcc, test_t, test_p, test_g_pcc = self._evaluate(
            test_loader, model, eval_criterion
        )
        log.info(
            f"Test MAE: {test_mae:.3f} | Test Per-Drug PCC: {test_pd_pcc:.4f} |"
            f" Test Per-Cell PCC: {test_pc_pcc:.4f} | Test Global PCC: {test_g_pcc:.4f}"
        )

        if self.cfg.task.get("save_predictions", True):
            self._save_predictions(test_p, test_t, test_pd_pcc, split_name, test_g_pcc, model_label=model_label)

        return {
            "test_mae": float(test_mae),
            "test_per_drug_pcc": float(test_pd_pcc),
            "test_per_cell_pcc": float(test_pc_pcc),
            "test_global_pcc": float(test_g_pcc),
        }

    def _save_predictions(self, preds, targets, per_drug_pcc, split_name, global_pcc, model_label: str = ""):
        if HydraConfig.initialized():
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
        if model_label:
            output_dir = output_dir / model_label
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"prediction": preds.flatten(), "ground_truth": targets.flatten()})

        counter = 0
        for file in output_dir.iterdir():
            if file.name.startswith(f"pred_{split_name}"):
                counter += 1
        save_name = f"pred_{split_name}_{counter}.csv"
        df.to_csv(output_dir / save_name, index=False)
        with open(output_dir / "metrics.csv", "a") as f:
            f.write(f"{split_name},{per_drug_pcc},{global_pcc}\n")

    def _compute_fm_embeddings(self) -> list[tuple[str, str]]:
        """Compute embeddings for each foundation model and save as CSV.

        Returns list of (model_name, csv_path) tuples.
        """
        h5ad_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.ground_truth_h5ad)
        log.info(f"Loading ground truth h5ad for FM embeddings: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

        if HydraConfig.initialized():
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for fm_name in self.cfg.task.foundation_models:
            log.info(f"Computing embeddings for foundation model: {fm_name}")
            model_cfg_path = hydra.utils.to_absolute_path(f"configs/model/{fm_name}.yaml")
            model_cfg = OmegaConf.load(model_cfg_path)
            base_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
            merged_cfg = OmegaConf.merge(base_cfg, model_cfg)

            env_path = merged_cfg.model.get("env_path", None)

            if env_path is not None:
                from models.subprocess_embed import subprocess_embed

                result = subprocess_embed(
                    adata=adata,
                    cfg=merged_cfg,
                    env_path=env_path,
                    timeout=merged_cfg.model.get("subprocess_timeout", 3600),
                )
            else:
                from models import get_model

                device = "cuda" if torch.cuda.is_available() else "cpu"
                wrapper = get_model(merged_cfg)
                wrapper.load_pretrained()
                wrapper.to(device)
                result = wrapper.embed(adata, batch_size=merged_cfg.model.batch_size)

            # Save as CSV with COSMIC_ID index to match existing submission format
            emb_df = pd.DataFrame(
                result.cell_embeddings,
                index=adata.obs_names,
            )
            csv_path = str(output_dir / f"{fm_name}_cell_embeddings.csv")
            emb_df.to_csv(csv_path)
            log.info(f"Saved {fm_name} embeddings: shape {result.cell_embeddings.shape} -> {csv_path}")
            results.append((fm_name, csv_path))

        return results

    def _generate_splits(self) -> list[dict]:
        """Generate train/val/test splits (independent of cell embeddings)."""
        split_mode = self.cfg.task.get("split_mode", "cancer_stratified")
        k_fold = self.cfg.task.get("k_fold", 0)

        # Load Indices
        drug_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.drug_emb_path)
        meta_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.metadata)

        # --- Detect Directory vs CSV for drug IDs ---
        if os.path.isdir(drug_path):
            log.info(f"Detecting Drug IDs from directory: {drug_path}")
            hkl_files = glob.glob(os.path.join(drug_path, "*.hkl"))
            drugs = [os.path.splitext(os.path.basename(f))[0] for f in hkl_files]
            if not drugs:
                raise ValueError(f"No .hkl files found in {drug_path}")
        else:
            log.info(f"Detecting Drug IDs from CSV: {drug_path}")
            drug_df = pd.read_csv(drug_path)
            if "Drug_ID" in drug_df.columns:
                drugs = drug_df["Drug_ID"].astype(str).unique().tolist()
            elif "DRUG_ID" in drug_df.columns:
                drugs = drug_df["DRUG_ID"].astype(str).unique().tolist()
            else:
                drugs = drug_df.iloc[:, 0].astype(str).unique().tolist()

        # Cells (for Stratified CV)
        meta_df = pd.read_csv(meta_path)
        cells = meta_df["COSMIC_ID"].astype(str).unique()
        cell_map = dict(zip(meta_df["COSMIC_ID"].astype(str), meta_df["TCGA_DESC"].fillna("unknown"), strict=True))
        cell_labels = [cell_map[c] for c in cells]

        from collections import Counter

        label_counts = Counter(cell_labels)
        log.info(f"Cancer type distribution ({len(label_counts)} types, {len(cells)} cells):")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1]):
            log.info(f"  {label}: {count}")

        # For cancer_stratified splits, drop ambiguous/rare cancer types
        if k_fold > 1 and split_mode == "cancer_stratified":
            exclude_labels = {"UNCLASSIFIED", "OTHER", "unknown"}
            rare_labels = {lab for lab, c in label_counts.items() if c < k_fold}
            drop_labels = exclude_labels | rare_labels
            actual_drops = {lab for lab in drop_labels if lab in label_counts}
            if actual_drops:
                log.warning(f"Dropping {len(actual_drops)} cancer types (rare or ambiguous): {actual_drops}")
                mask = [lab not in drop_labels for lab in cell_labels]
                cells = [c for c, m in zip(cells, mask, strict=True) if m]
                cell_labels = [lab for lab, m in zip(cell_labels, mask, strict=True) if m]
                log.info(f"Remaining: {len(cells)} cells, {len(set(cell_labels))} cancer types")

        splits = []

        if k_fold > 1:
            log.info(f"Generating {k_fold}-Fold Splits for {split_mode}")

            if split_mode == "cold_drug":
                kf = KFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                d_arr = np.array(drugs)
                for tr_idx, te_idx in kf.split(d_arr):
                    t_drugs = d_arr[te_idx].tolist()
                    tr_sub, val_sub = train_test_split(
                        tr_idx, test_size=0.1, random_state=self.cfg.task.get("data_seed", 42)
                    )
                    v_drugs = d_arr[val_sub].tolist()
                    splits.append({"test_drugs": t_drugs, "val_drugs": v_drugs})

            elif split_mode == "cold_cell":
                kf = KFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                c_arr = np.array(cells)
                for tr_idx, te_idx in kf.split(c_arr):
                    t_cells = c_arr[te_idx].tolist()
                    tr_sub, val_sub = train_test_split(
                        tr_idx,
                        test_size=0.1,
                        random_state=self.cfg.task.get("data_seed", 42),
                    )
                    v_cells = c_arr[val_sub].tolist()
                    splits.append({"test_cells": t_cells, "val_cells": v_cells})

            elif split_mode == "cancer_stratified":
                skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                c_arr = np.array(cells)
                l_arr = np.array(cell_labels)
                for tr_idx, te_idx in skf.split(c_arr, l_arr):
                    t_cells = c_arr[te_idx].tolist()
                    tr_sub, val_sub = train_test_split(
                        tr_idx,
                        test_size=0.1,
                        stratify=l_arr[tr_idx],
                        random_state=self.cfg.task.get("data_seed", 42),
                    )
                    v_cells = c_arr[val_sub].tolist()
                    splits.append({"test_cells": t_cells, "val_cells": v_cells})

            elif split_mode == "double_cold":
                kf_d = KFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                skf_c = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                d_arr = np.array(drugs)
                c_arr = np.array(cells)
                l_arr = np.array(cell_labels)

                d_splits = list(kf_d.split(d_arr))
                c_splits = list(skf_c.split(c_arr, l_arr))

                for i in range(k_fold):
                    _, d_te = d_splits[i]
                    _, c_te = c_splits[i]
                    splits.append({"test_drugs": d_arr[d_te].tolist(), "test_cells": c_arr[c_te].tolist()})

        else:
            # Single Split (Legacy/Simple/LOTO)
            splits.append({})

        return splits

    def _run_and_aggregate(
        self,
        cell_path: str,
        splits: list[dict],
        model_label: str,
        model_cls=DualStreamModel,
    ) -> dict[str, Any]:
        """Run all splits for one cell embedding source and aggregate results."""
        results = []
        for i, kwargs in enumerate(splits):
            log.info(f"--- [{model_label}] Running Split {i + 1}/{len(splits)} ---")
            res = self._run_single_split(cell_path, model_label=model_label, model_cls=model_cls, **kwargs)
            results.append(res)

        pd_mae = [r["test_mae"] for r in results]
        pd_per_pccs = [r["test_per_drug_pcc"] for r in results]
        pc_per_pccs = [r["test_per_cell_pcc"] for r in results]
        pd_global_pccs = [r["test_global_pcc"] for r in results]

        log.info(f"=== [{model_label}] Drug Response Prediction Summary ===")
        log.info(f"Mean MAE across splits: {np.mean(pd_mae):.4f}")
        log.info(f"Std Dev MAE across splits: {np.std(pd_mae):.4f}")
        log.info(f"Mean Per-Drug PCC across splits: {np.mean(pd_per_pccs):.4f}")
        log.info(f"Std Dev Per-Drug PCC across splits: {np.std(pd_per_pccs):.4f}")
        log.info(f"Mean Per-Cell PCC across splits: {np.mean(pc_per_pccs):.4f}")
        log.info(f"Std Dev Per-Cell PCC across splits: {np.std(pc_per_pccs):.4f}")
        log.info(f"Mean Global PCC across splits: {np.mean(pd_global_pccs):.4f}")
        log.info(f"Std Dev Global PCC across splits: {np.std(pd_global_pccs):.4f}")

        return {
            "mean_mae": float(np.mean(pd_mae)),
            "std_mae": float(np.std(pd_mae)),
            "mean_per_drug_pcc": float(np.mean(pd_per_pccs)),
            "std_per_drug_pcc": float(np.std(pd_per_pccs)),
            "mean_per_cell_pcc": float(np.mean(pc_per_pccs)),
            "std_per_cell_pcc": float(np.std(pc_per_pccs)),
            "mean_global_pcc": float(np.mean(pd_global_pccs)),
            "std_global_pcc": float(np.std(pd_global_pccs)),
            "all_results": results,
        }

    def _compute_mean_metrics(self, test_df: pd.DataFrame, pred_col: str) -> dict[str, float]:
        """Compute MAE, per-drug PCC, per-cell PCC, global PCC from a test_df with a prediction column."""
        targets = test_df["ic50"].values
        preds = test_df[pred_col].values

        mae = float(np.mean(np.abs(targets - preds)))

        # Per-drug PCC
        drug_correlations = []
        for _, group in test_df.groupby("drug_id"):
            if len(group) > 5:
                if group[pred_col].std() < 1e-9 or group["ic50"].std() < 1e-9:
                    continue
                r, _ = pearsonr(group[pred_col], group["ic50"])
                drug_correlations.append(r)
        mean_per_drug_r = float(np.mean(drug_correlations)) if drug_correlations else 0.0

        # Per-cell PCC
        cell_correlations = []
        for _, group in test_df.groupby("cell_id"):
            if len(group) > 5:
                if group[pred_col].std() < 1e-9 or group["ic50"].std() < 1e-9:
                    continue
                r, _ = pearsonr(group[pred_col], group["ic50"])
                cell_correlations.append(r)
        mean_per_cell_r = float(np.mean(cell_correlations)) if cell_correlations else 0.0

        # Global PCC
        if len(targets) > 1 and np.std(preds) > 1e-9:
            global_r, _ = pearsonr(targets, preds)
            global_r = float(global_r)
        else:
            global_r = 0.0

        return {
            "test_mae": mae,
            "test_per_drug_pcc": mean_per_drug_r,
            "test_per_cell_pcc": mean_per_cell_r,
            "test_global_pcc": global_r,
        }

    def _run_mean_baselines(
        self,
        splits: list[dict],
        cell_path: str,
    ) -> dict[str, dict[str, Any]]:
        """Run mean-based baselines (global, per-drug, per-cell) across splits.

        Returns a dict mapping baseline name -> aggregated results.
        """
        split_mode = self.cfg.task.get("split_mode", "cancer_stratified")

        # Determine which mean baselines to run based on split_mode
        run_per_drug = split_mode in ("random", "cold_cell", "cancer_stratified", "loto")
        run_per_cell = split_mode in ("random", "cold_drug")

        baseline_names = ["global_mean"]
        if run_per_drug:
            baseline_names.append("per_drug_mean")
        if run_per_cell:
            baseline_names.append("per_cell_mean")

        # Accumulate per-split results for each baseline
        baseline_split_results: dict[str, list[dict]] = {name: [] for name in baseline_names}

        drug_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.drug_emb_path)
        matrix_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.dose_response_path)
        metadata_path = (
            hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.metadata)
            if self.cfg.task.data.metadata
            else None
        )

        for i, split_kwargs in enumerate(splits):
            log.info(f"--- [Mean Baselines] Running Split {i + 1}/{len(splits)} ---")

            dm = DataManager(cell_path, drug_path, matrix_path, metadata_path)
            df = dm.get_aligned_indices()

            test_tissue = self.cfg.task.get("test_tissue")
            drug_prop = self.cfg.task.get("drug_proportion")
            data_seed = self.cfg.task.get("data_seed", 42)

            train_df, _val_df, test_df = dm.split_data(
                df,
                mode=split_mode,
                holdout_tissue=test_tissue,
                drug_prop=drug_prop,
                data_seed=data_seed,
                **split_kwargs,
            )

            global_mean = train_df["ic50"].mean()

            # Global mean baseline
            test_df = test_df.copy()
            test_df["pred_global"] = global_mean
            res = self._compute_mean_metrics(test_df, "pred_global")
            baseline_split_results["global_mean"].append(res)
            log.info(f"  global_mean | MAE: {res['test_mae']:.4f} | G-PCC: {res['test_global_pcc']:.4f}")

            # Per-drug mean baseline
            if run_per_drug:
                drug_means = train_df.groupby("drug_id")["ic50"].mean()
                test_df["pred_per_drug"] = test_df["drug_id"].map(drug_means).fillna(global_mean)
                res = self._compute_mean_metrics(test_df, "pred_per_drug")
                baseline_split_results["per_drug_mean"].append(res)
                log.info(f"  per_drug_mean | MAE: {res['test_mae']:.4f} | G-PCC: {res['test_global_pcc']:.4f}")

            # Per-cell mean baseline
            if run_per_cell:
                cell_means = train_df.groupby("cell_id")["ic50"].mean()
                test_df["pred_per_cell"] = test_df["cell_id"].map(cell_means).fillna(global_mean)
                res = self._compute_mean_metrics(test_df, "pred_per_cell")
                baseline_split_results["per_cell_mean"].append(res)
                log.info(f"  per_cell_mean | MAE: {res['test_mae']:.4f} | G-PCC: {res['test_global_pcc']:.4f}")

        # Aggregate across splits (same format as _run_and_aggregate)
        all_baseline_results = {}
        for name, split_results in baseline_split_results.items():
            pd_mae = [r["test_mae"] for r in split_results]
            pd_per_pccs = [r["test_per_drug_pcc"] for r in split_results]
            pc_per_pccs = [r["test_per_cell_pcc"] for r in split_results]
            pd_global_pccs = [r["test_global_pcc"] for r in split_results]

            log.info(f"=== [{name}] Drug Response Prediction Summary ===")
            log.info(f"Mean MAE: {np.mean(pd_mae):.4f} +/- {np.std(pd_mae):.4f}")
            log.info(f"Mean Global PCC: {np.mean(pd_global_pccs):.4f} +/- {np.std(pd_global_pccs):.4f}")

            all_baseline_results[name] = {
                "mean_mae": float(np.mean(pd_mae)),
                "std_mae": float(np.std(pd_mae)),
                "mean_per_drug_pcc": float(np.mean(pd_per_pccs)),
                "std_per_drug_pcc": float(np.std(pd_per_pccs)),
                "mean_per_cell_pcc": float(np.mean(pc_per_pccs)),
                "std_per_cell_pcc": float(np.std(pc_per_pccs)),
                "mean_global_pcc": float(np.mean(pd_global_pccs)),
                "std_global_pcc": float(np.std(pd_global_pccs)),
                "all_results": split_results,
            }

        return all_baseline_results

    def _run_raw_expression_baseline(self, splits: list[dict]) -> dict[str, Any]:
        """Run raw expression baseline: train DualStreamModel on all genes, no preprocessing.

        No HVG selection, no PCA, no scaling — just the raw expression values.
        """
        h5ad_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.ground_truth_h5ad)
        log.info(f"[RawExpr] Loading expression data: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        cell_ids = adata.obs_names.astype(str).tolist()
        log.info(f"[RawExpr] Expression matrix shape: {X.shape}")

        if HydraConfig.initialized():
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw expression as CSV once (no per-split preprocessing needed)
        emb_df = pd.DataFrame(X, index=cell_ids)
        csv_path = str(output_dir / "raw_expr_embeddings.csv")
        emb_df.to_csv(csv_path)
        log.info(f"[RawExpr] Saved raw expression: {X.shape} -> {csv_path}")

        results = []
        for i, split_kwargs in enumerate(splits):
            log.info(f"--- [RawExpr] Running Split {i + 1}/{len(splits)} ---")
            res = self._run_single_split(csv_path, model_label="RawExpr", **split_kwargs)
            results.append(res)

        # Aggregate across splits
        model_label = "RawExpr"
        pd_mae = [r["test_mae"] for r in results]
        pd_per_pccs = [r["test_per_drug_pcc"] for r in results]
        pc_per_pccs = [r["test_per_cell_pcc"] for r in results]
        pd_global_pccs = [r["test_global_pcc"] for r in results]

        log.info(f"=== [{model_label}] Drug Response Prediction Summary ===")
        log.info(f"Mean MAE across splits: {np.mean(pd_mae):.4f}")
        log.info(f"Std Dev MAE across splits: {np.std(pd_mae):.4f}")
        log.info(f"Mean Per-Drug PCC across splits: {np.mean(pd_per_pccs):.4f}")
        log.info(f"Std Dev Per-Drug PCC across splits: {np.std(pd_per_pccs):.4f}")
        log.info(f"Mean Per-Cell PCC across splits: {np.mean(pc_per_pccs):.4f}")
        log.info(f"Std Dev Per-Cell PCC across splits: {np.std(pc_per_pccs):.4f}")
        log.info(f"Mean Global PCC across splits: {np.mean(pd_global_pccs):.4f}")
        log.info(f"Std Dev Global PCC across splits: {np.std(pd_global_pccs):.4f}")

        return {
            "mean_mae": float(np.mean(pd_mae)),
            "std_mae": float(np.std(pd_mae)),
            "mean_per_drug_pcc": float(np.mean(pd_per_pccs)),
            "std_per_drug_pcc": float(np.std(pd_per_pccs)),
            "mean_per_cell_pcc": float(np.mean(pc_per_pccs)),
            "std_per_cell_pcc": float(np.std(pc_per_pccs)),
            "mean_global_pcc": float(np.mean(pd_global_pccs)),
            "std_global_pcc": float(np.std(pd_global_pccs)),
            "all_results": results,
        }

    def _run_hvg_pca_baseline(self, splits: list[dict]) -> dict[str, Any]:
        """Run HVG+PCA baseline: select HVGs, per-split scale+PCA, then train DualStreamModel.

        This baseline tests whether foundation model representations outperform
        simple dimensionality reduction (PCA of highly variable genes). HVG selection
        is global (unsupervised), but scaling and PCA are fit per-split on training
        cells only to prevent data leakage.
        """
        from sklearn.decomposition import PCA

        hvg_cfg = self.cfg.task.get("hvg_pca_baseline", {})
        n_hvgs = int(hvg_cfg.get("n_hvgs", 2000))
        n_pcs = int(hvg_cfg.get("n_pcs", 512))

        # Load expression data
        h5ad_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.ground_truth_h5ad)
        log.info(f"[HVG+PCA] Loading expression data: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

        # Select HVGs (global, unsupervised — no label leakage)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs)
        adata = adata[:, adata.var["highly_variable"]].copy()
        log.info(f"[HVG+PCA] Selected {adata.n_vars} HVGs")

        # Dense expression matrix
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        cell_ids = adata.obs_names.astype(str).tolist()

        # Output directory for per-split CSVs
        if HydraConfig.initialized():
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, split_kwargs in enumerate(splits):
            log.info(f"--- [HVG+PCA] Running Split {i + 1}/{len(splits)} ---")

            # Determine train cells = all cells minus test/val cells
            exclude = set()
            if "test_cells" in split_kwargs:
                exclude.update(str(c) for c in split_kwargs["test_cells"])
            if "val_cells" in split_kwargs:
                exclude.update(str(c) for c in split_kwargs["val_cells"])
            train_mask = np.array([c not in exclude for c in cell_ids])

            if not train_mask.any():
                log.warning("[HVG+PCA] No training cells identified, fitting on all cells")
                train_mask = np.ones(len(cell_ids), dtype=bool)

            n_train = int(train_mask.sum())

            # Per-split scaling: fit on train cells, transform all, clip at ±10
            train_mean = X[train_mask].mean(axis=0)
            train_std = X[train_mask].std(axis=0)
            train_std[train_std < 1e-12] = 1.0
            X_scaled = (X - train_mean) / train_std
            X_scaled = np.clip(X_scaled, -10, 10)

            # Per-split PCA: fit on train cells, transform all
            n_features = X_scaled.shape[1]
            effective_n_pcs = min(n_pcs, n_train - 1, n_features)
            log.info(
                f"[HVG+PCA] PCA: {effective_n_pcs} components "
                f"(requested={n_pcs}, train_cells={n_train}, features={n_features})"
            )

            pca = PCA(n_components=effective_n_pcs, random_state=self.cfg.task.get("data_seed", 42))
            pca.fit(X_scaled[train_mask])
            X_pca = pca.transform(X_scaled)

            # Save as CSV (cell IDs as index, matching FM embedding format)
            emb_df = pd.DataFrame(X_pca, index=cell_ids)
            csv_path = str(output_dir / f"hvg_pca_split{i}_embeddings.csv")
            emb_df.to_csv(csv_path)
            log.info(f"[HVG+PCA] Saved embeddings: {X_pca.shape} -> {csv_path}")

            # Train DualStreamModel with PCA embeddings
            res = self._run_single_split(csv_path, model_label="HVG+PCA", **split_kwargs)
            results.append(res)

        # Aggregate across splits
        model_label = "HVG+PCA"
        pd_mae = [r["test_mae"] for r in results]
        pd_per_pccs = [r["test_per_drug_pcc"] for r in results]
        pc_per_pccs = [r["test_per_cell_pcc"] for r in results]
        pd_global_pccs = [r["test_global_pcc"] for r in results]

        log.info(f"=== [{model_label}] Drug Response Prediction Summary ===")
        log.info(f"Mean MAE across splits: {np.mean(pd_mae):.4f}")
        log.info(f"Std Dev MAE across splits: {np.std(pd_mae):.4f}")
        log.info(f"Mean Per-Drug PCC across splits: {np.mean(pd_per_pccs):.4f}")
        log.info(f"Std Dev Per-Drug PCC across splits: {np.std(pd_per_pccs):.4f}")
        log.info(f"Mean Per-Cell PCC across splits: {np.mean(pc_per_pccs):.4f}")
        log.info(f"Std Dev Per-Cell PCC across splits: {np.std(pc_per_pccs):.4f}")
        log.info(f"Mean Global PCC across splits: {np.mean(pd_global_pccs):.4f}")
        log.info(f"Std Dev Global PCC across splits: {np.std(pd_global_pccs):.4f}")

        return {
            "mean_mae": float(np.mean(pd_mae)),
            "std_mae": float(np.std(pd_mae)),
            "mean_per_drug_pcc": float(np.mean(pd_per_pccs)),
            "std_per_drug_pcc": float(np.std(pd_per_pccs)),
            "mean_per_cell_pcc": float(np.mean(pc_per_pccs)),
            "std_per_cell_pcc": float(np.std(pc_per_pccs)),
            "mean_global_pcc": float(np.mean(pd_global_pccs)),
            "std_global_pcc": float(np.std(pd_global_pccs)),
            "all_results": results,
        }

    def run(self) -> dict[str, Any]:
        # Build eval entries: list of (model_label, cell_path)
        eval_entries: list[tuple[str, str]] = []

        # Add pre-computed submission CSV(s) if configured
        submission = self.cfg.task.data.get("submission")
        if submission is not None:
            submissions = submission if isinstance(submission, (list, ListConfig)) else [submission]
            for sub in submissions:
                abs_submission_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + sub)
                # Derive label from filename (e.g. "submission/scGPT_full_expr.csv" -> "scGPT_full_expr")
                model_label = Path(sub).stem
                eval_entries.append((model_label, abs_submission_path))

        # Compute foundation model embeddings and add them
        fm_list = self.cfg.task.get("foundation_models", [])
        if len(fm_list) > 0:
            fm_embeddings = self._compute_fm_embeddings()
            eval_entries.extend(fm_embeddings)

        # Generate splits once (shared across all models)
        splits = self._generate_splits()

        # Evaluate each model
        all_model_results = {}
        for model_label, cell_path in eval_entries:
            log.info(f"=== Evaluating model: {model_label} ===")
            model_results = self._run_and_aggregate(cell_path, splits, model_label)
            all_model_results[model_label] = model_results

        # Run baselines if enabled
        if self.cfg.task.get("run_baselines", False):
            # Need a cell_path for DataManager alignment and DrugMLP dataloader
            if eval_entries:
                baseline_cell_path = eval_entries[0][1]
            else:
                # Baselines-only run: compute a dummy cell embedding from ground truth
                # We still need a cell CSV for DataManager to align IDs
                h5ad_path = hydra.utils.to_absolute_path(
                    self.cfg.task.data.data_root + self.cfg.task.data.ground_truth_h5ad
                )
                adata = sc.read_h5ad(h5ad_path)
                if HydraConfig.initialized():
                    output_dir = Path(HydraConfig.get().runtime.output_dir)
                else:
                    output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
                output_dir.mkdir(parents=True, exist_ok=True)
                # Use raw expression as a placeholder cell embedding
                emb_df = pd.DataFrame(
                    adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                    index=adata.obs_names,
                )
                baseline_cell_path = str(output_dir / "_baseline_cell_embeddings.csv")
                emb_df.to_csv(baseline_cell_path)
                log.info(f"Created baseline cell embeddings from expression: {baseline_cell_path}")

            # DrugMLP baseline
            log.info("=== Running DrugMLP baseline ===")
            drug_mlp_results = self._run_and_aggregate(baseline_cell_path, splits, "DrugMLP", model_cls=DrugMLP)
            all_model_results["DrugMLP"] = drug_mlp_results

            # Mean baselines
            log.info("=== Running mean baselines ===")
            mean_results = self._run_mean_baselines(splits, baseline_cell_path)
            all_model_results.update(mean_results)

            # HVG+PCA baseline
            log.info("=== Running HVG+PCA baseline ===")
            hvg_pca_results = self._run_hvg_pca_baseline(splits)
            all_model_results["HVG+PCA"] = hvg_pca_results

            # Raw expression baseline
            if self.cfg.task.get("run_raw_expression_baseline", False):
                log.info("=== Running raw expression baseline ===")
                raw_expr_results = self._run_raw_expression_baseline(splits)
                all_model_results["RawExpr"] = raw_expr_results

        # Log comparison summary
        if len(all_model_results) > 1:
            log.info("=== Multi-Model Comparison ===")
            header = f"{'Model':<25} {'MAE':>14} {'PD-PCC':>14} {'PC-PCC':>14} {'G-PCC':>14}"
            log.info(header)
            log.info("-" * len(header))
            for label, res in all_model_results.items():
                log.info(
                    f"{label:<25} {res['mean_mae']:>6.4f}±{res['std_mae']:<6.4f}"
                    f" {res['mean_per_drug_pcc']:>6.4f}±{res['std_per_drug_pcc']:<6.4f}"
                    f" {res['mean_per_cell_pcc']:>6.4f}±{res['std_per_cell_pcc']:<6.4f}"
                    f" {res['mean_global_pcc']:>6.4f}±{res['std_global_pcc']:<6.4f}"
                )

        # Return results (single-model backward compatible)
        if len(all_model_results) == 1:
            return next(iter(all_model_results.values()))
        return all_model_results
