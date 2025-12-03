import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from omegaconf import DictConfig
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from .model import MLP, UGCNN, CombinedMLP
from .utils import (
    DataSplit,
    DrugSplit,
    FeatureExtract,
    MetadataGenerate,
)

log = logging.getLogger(__name__)


class DrugResponsePredictionRunner:
    """Runner for drug response prediction evaluation task."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the runner with Hydra configuration.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self._validate_config()
        self._set_device()
        # self._seed_everything()

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_paths = [
            ("task", "paths", "drug_info"),
            ("task", "paths", "cell_line_info"),
            ("task", "paths", "drug_features"),
            ("task", "paths", "ground_truth"),
            ("task", "paths", "submission"),
        ]

        for path_parts in required_paths:
            try:
                value = self.cfg
                for part in path_parts:
                    value = value[part]
                if value is None:
                    raise ValueError(f"Missing required config: {'.'.join(path_parts)}")
            except (KeyError, AttributeError) as err:
                raise ValueError(f"Missing required config: {'.'.join(path_parts)}") from err

        # Validate test and val drugs
        test_drug = self.cfg.task.get("test_drug")
        val_drug = self.cfg.task.get("val_drug")
        if (test_drug is None and val_drug is not None) or (test_drug is not None and val_drug is None):
            raise ValueError("test_drug and val_drug must both be None or both be specified")
        if test_drug is not None and test_drug == val_drug:
            raise ValueError("test_drug and val_drug must be different")

    def _set_device(self):
        """Set the compute device based on configuration."""
        gpu_id = self.cfg.task.get("gpu_id", -1)
        if gpu_id == -1:
            self.device = torch.device("cpu")
            log.info("Using CPU")
        elif gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU with ID {gpu_id} is not available")
        else:
            self.device = torch.device(f"cuda:{gpu_id}")
            log.info(f"Using GPU: {gpu_id}")

    def _seed_everything(self):
        """Set random seeds for reproducibility."""
        seed = self.cfg.task.get("seed", 42)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        log.info(f"Random seed set to {seed}")

    def _load_data(self):
        """Load and prepare data for training and evaluation."""
        # Resolve absolute paths
        drug_info_file = hydra.utils.to_absolute_path(self.cfg.task.paths.drug_info)
        cell_line_info_file = hydra.utils.to_absolute_path(self.cfg.task.paths.cell_line_info)
        drug_feature_file = hydra.utils.to_absolute_path(self.cfg.task.paths.drug_features)
        ground_truth_file = hydra.utils.to_absolute_path(self.cfg.task.paths.ground_truth)
        submission_file = hydra.utils.to_absolute_path(self.cfg.task.paths.submission)

        log.info(f"Loading drug info from {drug_info_file}")
        log.info(f"Loading cell line info from {cell_line_info_file}")
        log.info(f"Loading drug features from {drug_feature_file}")
        log.info(f"Loading ground truth from {ground_truth_file}")
        log.info(f"Loading submission (gene expression) from {submission_file}")

        # Generate metadata and data indices
        drug_feature, gexpr_feature, data_idx = MetadataGenerate(
            drug_info_file,
            cell_line_info_file,
            drug_feature_file,
            submission_file,
            ground_truth_file,
        )

        return drug_feature, gexpr_feature, data_idx

    def _split_data(self, data_idx):
        """Split data into train, validation, and test sets."""
        test_drug = self.cfg.task.get("test_drug")
        val_drug = self.cfg.task.get("val_drug")
        tcga_labels = [
            "ALL",
            "BLCA",
            "BRCA",
            "CESC",
            "DLBC",
            "LIHC",
            "LUAD",
            "ESCA",
            "GBM",
            "HNSC",
            "KIRC",
            "LAML",
            "LCML",
            "LGG",
            "LUSC",
            "MESO",
            "MM",
            "NB",
            "OV",
            "PAAD",
            "SCLC",
            "SKCM",
            "STAD",
            "THCA",
            "COAD/READ",
        ]

        if test_drug is None and val_drug is None:
            # Cell line split
            log.info("Performing cell line split (95/5)")
            data_train_idx, data_test_idx = DataSplit(data_idx, tcga_labels, ratio=0.95)
            data_train_idx, data_val_idx = DataSplit(data_train_idx, tcga_labels, ratio=1 - 0.05 / 0.95)
        else:
            # Drug split
            log.info(f"Performing drug split (test={test_drug}, val={val_drug})")
            data_train_idx, data_test_idx = DrugSplit(data_idx, test_drug)
            if len(data_test_idx) == 0:
                raise ValueError(f"Test drug {test_drug} not found in data")
            data_train_idx, data_val_idx = DrugSplit(data_train_idx, val_drug)
            if len(data_val_idx) == 0:
                raise ValueError(f"Validation drug {val_drug} not found in data")

        log.info(f"Data split: train={len(data_train_idx)}, val={len(data_val_idx)}, test={len(data_test_idx)}")
        return data_train_idx, data_val_idx, data_test_idx

    def _create_model(self, gexpr_dim: int):
        """Create and initialize the model."""
        drug_gcnn = UGCNN(input_dim=75, hidden_dims=[256, 256, 256], out_channels=100).to(self.device)
        gexpr_mlp = MLP(out_dim=100, input_dim=gexpr_dim).to(self.device)
        comb_mlp = CombinedMLP(input_dim=200).to(self.device)

        log.info("Model initialized")
        return drug_gcnn, gexpr_mlp, comb_mlp

    def _train_epoch(self, train_loader, drug_gcnn, gexpr_mlp, comb_mlp, optimizer, criterion):
        """Train for one epoch."""
        drug_gcnn.train()
        gexpr_mlp.train()
        comb_mlp.train()

        running_loss = 0.0
        for drug, gexpr, target in train_loader:
            target = target.to(self.device)
            drug = drug.to(self.device)
            gexpr = gexpr.to(self.device)

            optimizer.zero_grad()
            drug_emb = drug_gcnn(drug)
            gexpr_emb = gexpr_mlp(gexpr)
            out = comb_mlp(torch.cat((drug_emb, gexpr_emb), dim=1))

            loss = criterion(out.view(-1), target.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        return running_loss / len(train_loader)

    def _evaluate(self, data_loader, drug_gcnn, gexpr_mlp, comb_mlp, criterion):
        """Evaluate model on a dataset."""
        drug_gcnn.eval()
        gexpr_mlp.eval()
        comb_mlp.eval()

        total_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for drug, gexpr, target in data_loader:
                target = target.to(self.device)
                drug = drug.to(self.device)
                gexpr = gexpr.to(self.device)

                drug_emb = drug_gcnn(drug)
                gexpr_emb = gexpr_mlp(gexpr)
                out = comb_mlp(torch.cat((drug_emb, gexpr_emb), dim=1))

                loss = criterion(out.view(-1), target.view(-1))
                total_loss += loss.item()

                all_targets.append(target.cpu().numpy())
                all_outputs.append(out.cpu().numpy())

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        pcc = pearsonr(all_targets.flatten(), all_outputs.flatten())[0]

        return total_loss / len(data_loader), pcc, all_targets, all_outputs

    def run(self) -> Dict[str, Any]:
        """
        Execute the drug response prediction evaluation.

        Returns:
            Dictionary containing evaluation metrics
        """
        log.info("Starting drug response prediction evaluation")

        # Load data
        drug_feature, gexpr_feature, data_idx = self._load_data()
        gexpr_dim = gexpr_feature.shape[-1]
        log.info(f"Gene expression dimension: {gexpr_dim}")

        # Split data
        data_train_idx, data_val_idx, data_test_idx = self._split_data(data_idx)

        # Extract features
        log.info("Extracting features for training set")
        X_drug_train, X_gexpr_train, Y_train, _ = FeatureExtract(data_train_idx, drug_feature, gexpr_feature)

        log.info("Extracting features for validation set")
        X_drug_val, X_gexpr_val, Y_val, _ = FeatureExtract(data_val_idx, drug_feature, gexpr_feature)

        log.info("Extracting features for test set")
        X_drug_test, X_gexpr_test, Y_test, test_metadata = FeatureExtract(data_test_idx, drug_feature, gexpr_feature)

        # Create data loaders
        batch_size = self.cfg.task.get("batch_size", 256)
        train_loader = DataLoader(
            list(zip(X_drug_train, X_gexpr_train, Y_train, strict=True)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cfg.task.get("num_workers", 4),
        )
        val_loader = DataLoader(
            list(zip(X_drug_val, X_gexpr_val, Y_val, strict=True)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.task.get("num_workers", 4),
        )
        test_loader = DataLoader(
            list(zip(X_drug_test, X_gexpr_test, Y_test, strict=True)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.task.get("num_workers", 4),
        )

        # Create model
        drug_gcnn, gexpr_mlp, comb_mlp = self._create_model(gexpr_dim)

        # Setup training
        parameters = list(drug_gcnn.parameters()) + list(gexpr_mlp.parameters()) + list(comb_mlp.parameters())
        optimizer = optim.Adam(
            parameters,
            lr=self.cfg.task.get("learning_rate", 0.003),
        )
        criterion = torch.nn.MSELoss()

        # Training loop
        epochs = self.cfg.task.get("epochs", 500)
        patience = self.cfg.task.get("patience", 10)
        best_val_pcc = -float("inf")
        best_models = None
        patience_counter = 0

        log.info(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, drug_gcnn, gexpr_mlp, comb_mlp, optimizer, criterion)
            val_loss, val_pcc, _, _ = self._evaluate(val_loader, drug_gcnn, gexpr_mlp, comb_mlp, criterion)

            log.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val PCC: {val_pcc:.4f}"
            )

            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_models = (
                    drug_gcnn.state_dict(),
                    gexpr_mlp.state_dict(),
                    comb_mlp.state_dict(),
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best models
        drug_gcnn.load_state_dict(best_models[0])
        gexpr_mlp.load_state_dict(best_models[1])
        comb_mlp.load_state_dict(best_models[2])

        # Evaluate on test set
        log.info("Evaluating on test set")
        test_loss, test_pcc, test_targets, test_predictions = self._evaluate(
            test_loader, drug_gcnn, gexpr_mlp, comb_mlp, criterion
        )

        log.info(f"Test PCC: {test_pcc:.4f}")

        # Prepare results
        results = {
            "pearson_correlation": float(test_pcc),
            "test_loss": float(test_loss),
            "best_val_pcc": float(best_val_pcc),
        }

        # Save predictions if output path is configured
        if self.cfg.task.get("save_predictions", True):
            self._save_predictions(test_metadata, test_predictions, test_targets, test_pcc)

        return results

    def _save_predictions(self, metadata, predictions, targets, pcc):
        """Save predictions to CSV file."""
        output_dir = Path(hydra.utils.to_absolute_path(self.cfg.task.get("output_dir", "./outputs")))
        output_dir.mkdir(parents=True, exist_ok=True)

        test_drug = self.cfg.task.get("test_drug", "cell_line")
        output_file = output_dir / f"predictions_{test_drug}.csv"

        df = pd.DataFrame(
            {
                "cancer_type": [m[0] for m in metadata],
                "cell_line": [m[1] for m in metadata],
                "pubchem_cid": [m[2] for m in metadata],
                "prediction": predictions.flatten(),
                "ground_truth": targets.flatten(),
            }
        )
        df.to_csv(output_file, index=False)
        log.info(f"Predictions saved to {output_file}")

        # Save summary metrics
        summary_file = output_dir / f"metrics_{test_drug}.csv"
        summary_df = pd.DataFrame(
            {
                "test_drug": [test_drug],
                "val_drug": [self.cfg.task.get("val_drug", "N/A")],
                "pearson_correlation": [pcc],
            }
        )
        summary_df.to_csv(summary_file, index=False)
        log.info(f"Metrics saved to {summary_file}")
