import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from scipy.stats import pearsonr

from loaders import get_dataloaders

# Import local modules
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
        """Validate required configuration parameters."""
        required_paths = [
            ("task", "paths", "drug_emb_path"),
            ("task", "paths", "dose_response_path"),
            ("task", "paths", "metadata"),
            ("task", "paths", "submission"),
        ]

        for path_parts in required_paths:
            try:
                value = self.cfg
                for part in path_parts:
                    value = value[part]
                if value is None:
                    if path_parts[-1] == "metadata":
                        continue
                    raise ValueError(f"Missing required config: {'.'.join(path_parts)}")
            except (KeyError, AttributeError) as err:
                if path_parts[-1] != "metadata":
                    raise ValueError(f"Missing required config: {'.'.join(path_parts)}") from err

    def _set_device(self):
        gpu_id = self.cfg.task.get("gpu_id", -1)
        if gpu_id == -1 or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            log.info("Using CPU")
        else:
            self.device = torch.device(f"cuda:{gpu_id}")
            log.info(f"Using GPU: {gpu_id}")

    def _seed_everything(self):
        seed = self.cfg.task.get("seed", 42)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        log.info(f"Random seed set to {seed}")

    def _create_model(self, cell_dim: int, drug_dim: int):
        model = DualStreamModel(cell_dim=cell_dim, drug_dim=drug_dim).to(self.device)
        log.info(f"Model initialized with Cell Dim: {cell_dim}, Drug Dim: {drug_dim}")
        return model

    def _train_epoch(self, train_loader, model, optimizer, criterion):
        model.train()
        running_loss = 0.0

        for cell_vec, drug_vec, target, _ in train_loader:
            cell_vec = cell_vec.to(self.device)
            drug_vec = drug_vec.to(self.device)
            target = target.to(self.device)

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

        # Store arrays to group later
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

        # Calculate per-drug Pearson correlation
        df_res = pd.DataFrame({"pred": all_preds, "target": all_targets, "drug_idx": all_drug_ids})

        correlations = []
        # Group by drug and calculate Pearson for that specific drug
        for _, group in df_res.groupby("drug_idx"):
            if len(group) > 5:  # Only calculate if enough samples
                if group["pred"].std() < 1e-9 or group["target"].std() < 1e-9:
                    continue  # Skip constant outputs
                r, _ = pearsonr(group["pred"], group["target"])
                correlations.append(r)

        mean_per_drug_r = np.mean(correlations) if correlations else 0.0

        # Also calc global for comparison
        global_r, _ = pearsonr(all_targets, all_preds)

        return total_loss / len(data_loader), mean_per_drug_r, np.array(all_targets), np.array(all_preds), global_r

    def _run_single_split(self, test_drug: str | None, val_drug: str | None) -> Dict[str, Any]:
        # 1. Determine Logic based on Args vs Config
        # If test_drug is passed (from the loop), it overrides everything for Cold Drug mode
        if test_drug:
            split_mode = "cold_drug"
            split_name = f"drug_{test_drug}_val_{val_drug}"
        else:
            # Fallback to config settings (e.g. LOTO, Random, Stratified)
            split_mode = self.cfg.task.get("split_mode", "cancer_stratified")
            split_name = f"split_{split_mode}"
            test_drug = None
            val_drug = None

        # Pull other config params
        test_tissue = self.cfg.task.get("test_tissue")
        drug_prop = self.cfg.task.get("drug_proportion")

        log.info(f"Running split: {split_name} (Mode: {split_mode})")

        # 2. Initialize Loader
        train_loader, val_loader, test_loader, dims = get_dataloaders(
            cell_path=hydra.utils.to_absolute_path(self.cfg.task.paths.submission),
            drug_path=hydra.utils.to_absolute_path(self.cfg.task.paths.drug_emb_path),
            matrix_path=hydra.utils.to_absolute_path(self.cfg.task.paths.dose_response_path),
            metadata_path=hydra.utils.to_absolute_path(self.cfg.task.paths.metadata)
            if self.cfg.task.paths.metadata
            else None,
            split_mode=split_mode,
            # Pass the specific holdouts
            test_drug=test_drug,
            val_drug=val_drug,
            holdout_tissue=test_tissue,
            drug_prop=drug_prop,
            batch_size=self.cfg.task.get("batch_size", 256),
        )

        # 3. Model & Opt
        model = self._create_model(dims["cell_dim"], dims["drug_dim"])
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.task.get("learning_rate", 0.003))
        criterion = nn.MSELoss()

        # 4. Train Loop
        epochs = self.cfg.task.get("epochs", 500)
        patience = self.cfg.task.get("patience", 10)
        best_val_per_drug_pcc = -float("inf")
        best_val_global_pcc = -float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, model, optimizer, criterion)
            val_loss, val_per_drug_pcc, _, _, val_global_pcc = self._evaluate(val_loader, model, criterion)

            if val_per_drug_pcc > best_val_per_drug_pcc:
                best_val_per_drug_pcc = val_per_drug_pcc
                best_val_global_pcc = val_global_pcc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                log.info(
                    f"Epoch {epoch + 1} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val per drug PCC: {val_per_drug_pcc:.4f} | "
                    f"Val naive PCC: {val_global_pcc:.4f}"
                )

            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

        # 5. Test
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_loss, test_per_drug_pcc, test_targets, test_preds, test_global_pcc = self._evaluate(
            test_loader, model, criterion
        )
        log.info(f"Test PCC for {split_name}: {test_per_drug_pcc:.4f}")

        if self.cfg.task.get("save_predictions", True):
            self._save_predictions(test_preds, test_targets, test_per_drug_pcc, split_name, test_global_pcc)
        return {
            "test_drug": test_drug,
            "test_loss": float(test_loss),
            "test_per_drug_pcc": float(test_per_drug_pcc),
            "best_val_per_drug_pcc": float(best_val_per_drug_pcc),
            "test_global_pcc": float(test_global_pcc),
            "best_val_global_pcc": float(best_val_global_pcc),
        }

    def _save_predictions(self, preds, targets, per_drug_pcc, split_name, global_pcc):
        output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"prediction": preds.flatten(), "ground_truth": targets.flatten()})
        df.to_csv(output_dir / f"pred_{split_name}.csv", index=False)

        with open(output_dir / "metrics.csv", "a") as f:
            f.write(f"{split_name},{per_drug_pcc},{global_pcc}\n")

    def run(self) -> Dict[str, Any]:
        drug_prop = self.cfg.task.get("drug_proportion")
        test_drug = self.cfg.task.get("test_drug")
        val_drug = self.cfg.task.get("val_drug")

        # Load drug list for iteration logic
        drug_path = hydra.utils.to_absolute_path(self.cfg.task.paths.drug_emb_path)
        drug_df = pd.read_csv(drug_path)

        if "Drug_ID" in drug_df.columns:
            available_drugs = drug_df["Drug_ID"].astype(str).unique().tolist()
        elif "DRUG_ID" in drug_df.columns:
            available_drugs = drug_df["DRUG_ID"].astype(str).unique().tolist()
        else:
            available_drugs = drug_df.iloc[:, 0].astype(str).unique().tolist()

        splits = []

        # Logic to generate list of (drug, val_drug) tuples to iterate over
        if drug_prop is not None:
            num_test = max(1, int(len(available_drugs) * drug_prop))
            test_drugs_list = random.sample(available_drugs, num_test)
            splits = [(d, None) for d in test_drugs_list]
            log.info(f"Proportion Mode: Running {len(splits)} cold-drug splits.")

        elif test_drug is not None:
            splits = [(str(test_drug), str(val_drug))]
            log.info(f"Specific Mode: Testing on drug {test_drug}, validating on drug {val_drug}.")

        else:
            # If no drug logic is set, we run once.
            # _run_single_split will fallback to config (e.g. LOTO or Stratified Cell)
            splits = [(None, None)]
            log.info("Cell/LOTO Mode: Running single split configuration.")

        results = []
        for t_drug, v_drug in splits:
            res = self._run_single_split(test_drug=t_drug, val_drug=v_drug)
            results.append(res)

        pccs = [r["test_per_drug_pcc"] for r in results]
        return {"mean_pcc": float(np.mean(pccs)), "std_pcc": float(np.std(pccs)), "results": results}
