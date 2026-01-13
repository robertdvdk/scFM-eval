import glob
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from loaders import get_drp_dataloaders

from .model import DualStreamModel

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
            ("task", "data", "submission"),
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

    def _create_model(self, cell_dim: int, drug_dim: int, is_graph: bool):
        # Pass is_graph to model
        model = DualStreamModel(cell_dim=cell_dim, drug_dim=drug_dim, is_graph=is_graph).to(self.device)
        # compile might need backend='eager' for PyG depending on PT version
        # model = torch.compile(model)
        log.info(f"Model initialized. Cell Dim: {cell_dim}, Drug Dim: {drug_dim}, Graph Mode: {is_graph}")
        return model

    def _train_epoch(self, train_loader, model, optimizer, criterion):
        model.train()
        running_loss = 0.0
        for cell_vec, drug_vec, target, _ in train_loader:
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

        with torch.no_grad():
            for cell_vec, drug_vec, target, d_idx in data_loader:
                cell_vec = cell_vec.to(self.device)
                drug_vec = drug_vec.to(self.device)
                target = target.to(self.device)

                out = model(cell_vec, drug_vec)
                loss = criterion(out.view(-1), target.view(-1))
                total_loss += loss.item()

                all_preds.extend(out.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                all_drug_ids.extend(d_idx.cpu().numpy().flatten())

        # Calculate Per-Drug Pearson
        df_res = pd.DataFrame({"pred": all_preds, "target": all_targets, "drug_idx": all_drug_ids})
        correlations = []
        for _, group in df_res.groupby("drug_idx"):
            if len(group) > 5:  # TODO remove?
                if group["pred"].std() < 1e-9 or group["target"].std() < 1e-9:  # TODO remove?
                    continue
                r, _ = pearsonr(group["pred"], group["target"])
                correlations.append(r)

        mean_per_drug_r = np.mean(correlations) if correlations else 0.0

        # Calculate Global Pearson
        if len(all_targets) > 1 and np.std(np.array(all_preds)) > 1e-9:
            global_r, _ = pearsonr(all_targets, all_preds)
        else:
            global_r = 0.0

        return total_loss / len(data_loader), mean_per_drug_r, np.array(all_targets), np.array(all_preds), global_r

    def _run_single_split(
        self,
        test_drugs: List[str] | None = None,
        val_drugs: List[str] | None = None,
        test_cells: List[str] | None = None,
        val_cells: List[str] | None = None,
    ) -> Dict[str, Any]:

        # Determine Mode for Logging
        split_mode = self.cfg.task.get("split_mode", "cancer_stratified")
        # if test_drugs:
        #     split_mode = "cold_drug"
        # if test_cells and not test_drugs:
        #     split_mode = "cold_cell"
        # if test_drugs and test_cells:
        #     split_mode = "double_cold"

        split_name = f"split_{split_mode}"

        # Pull Configs
        test_tissue = self.cfg.task.get("test_tissue")
        drug_prop = self.cfg.task.get("drug_proportion")

        log.info(f"Running split: {split_name} (Mode: {split_mode})")

        # Initialize Loader
        train_loader, val_loader, test_loader, dims = get_drp_dataloaders(
            cell_path=hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.submission),
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

        model = self._create_model(dims["cell_dim"], dims["drug_dim"], dims["is_graph"])
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
            val_mae, val_per_drug, _, _, val_global = self._evaluate(val_loader, model, eval_criterion)

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
                    f" Val Per-Drug PCC: {val_per_drug:.4f} | Val Global PCC: {val_global:.4f}"
                )

            if patience_counter >= patience:
                log.info(
                    f"Early stopping at epoch {epoch + 1} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} |"
                    f" Val Per-Drug PCC: {val_per_drug:.4f} | Val Global PCC: {val_global:.4f}"
                )
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        test_mae, test_pd_pcc, test_t, test_p, test_g_pcc = self._evaluate(test_loader, model, eval_criterion)
        log.info(f"Test MAE: {test_mae:.3f} | Test Per-Drug PCC: {test_pd_pcc:.4f} | Test Global PCC: {test_g_pcc:.4f}")

        if self.cfg.task.get("save_predictions", True):
            self._save_predictions(test_p, test_t, test_pd_pcc, split_name, test_g_pcc)

        return {
            "test_mae": float(test_mae),
            "test_per_drug_pcc": float(test_pd_pcc),
            "test_global_pcc": float(test_g_pcc),
        }

    def _save_predictions(self, preds, targets, per_drug_pcc, split_name, global_pcc):
        if HydraConfig.initialized():
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        else:
            output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
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

    def run(self) -> Dict[str, Any]:
        # Config
        split_mode = self.cfg.task.get("split_mode", "cancer_stratified")
        k_fold = self.cfg.task.get("k_fold", 0)

        # Load Indices
        drug_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.drug_emb_path)
        meta_path = hydra.utils.to_absolute_path(self.cfg.task.data.data_root + self.cfg.task.data.metadata)

        # --- FIXED BLOCK START: Detect Directory vs CSV for drug IDs ---
        if os.path.isdir(drug_path):
            log.info(f"Detecting Drug IDs from directory: {drug_path}")
            # Scan directory for .hkl files and extract IDs
            # Expecting filename format: {DRUG_ID}.hkl
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
        # --- FIXED BLOCK END ---

        # Cells (for Stratified CV)
        meta_df = pd.read_csv(meta_path)
        cells = meta_df["COSMIC_ID"].astype(str).unique()
        cell_map = dict(zip(meta_df["COSMIC_ID"].astype(str), meta_df["TCGA_DESC"], strict=True))
        cell_labels = [cell_map.get(c, "Unknown") for c in cells]

        splits = []

        # --- GENERATE SPLITS ---
        if k_fold > 1:
            log.info(f"Generating {k_fold}-Fold Splits for {split_mode}")

            if split_mode == "cold_drug":
                kf = KFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                d_arr = np.array(drugs)
                for tr_idx, te_idx in kf.split(d_arr):
                    t_drugs = d_arr[te_idx].tolist()
                    # Split Train into Train/Val
                    tr_sub, val_sub = train_test_split(
                        tr_idx, test_size=0.1, random_state=self.cfg.task.get("data_seed", 42)
                    )
                    v_drugs = d_arr[val_sub].tolist()
                    splits.append({"test_drugs": t_drugs, "val_drugs": v_drugs})

            elif split_mode in ["cold_cell", "cancer_stratified"]:
                skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.cfg.task.get("data_seed", 42))
                c_arr = np.array(cells)
                l_arr = np.array(cell_labels)
                for tr_idx, te_idx in skf.split(c_arr, l_arr):
                    t_cells = c_arr[te_idx].tolist()
                    tr_sub, val_sub = train_test_split(
                        tr_idx, test_size=0.1, stratify=l_arr[tr_idx], random_state=self.cfg.task.get("data_seed", 42)
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

        # --- EXECUTE ---
        results = []
        for i, kwargs in enumerate(splits):
            log.info(f"--- Running Split {i + 1}/{len(splits)} ---")
            res = self._run_single_split(**kwargs)
            results.append(res)

        # Summarize
        pd_mae = [r["test_mae"] for r in results]
        pd_per_pccs = [r["test_per_drug_pcc"] for r in results]
        pd_global_pccs = [r["test_global_pcc"] for r in results]

        log.info("=== Drug Response Prediction Summary ===")
        log.info(f"Mean MAE across splits: {np.mean(pd_mae):.4f}")
        log.info(f"Std Dev MAE across splits: {np.std(pd_mae):.4f}")
        log.info(f"Mean Per-Drug PCC across splits: {np.mean(pd_per_pccs):.4f}")
        log.info(f"Std Dev Per-Drug PCC across splits: {np.std(pd_per_pccs):.4f}")
        log.info(f"Mean Global PCC across splits: {np.mean(pd_global_pccs):.4f}")
        log.info(f"Std Dev Global PCC across splits: {np.std(pd_global_pccs):.4f}")
        return {
            "mean_mae": float(np.mean(pd_mae)),
            "std_mae": float(np.std(pd_mae)),
            "mean_per_drug_pcc": float(np.mean(pd_per_pccs)),
            "std_per_drug_pcc": float(np.std(pd_per_pccs)),
            "mean_global_pcc": float(np.mean(pd_global_pccs)),
            "std_global_pcc": float(np.std(pd_global_pccs)),
            "all_results": results,
        }
