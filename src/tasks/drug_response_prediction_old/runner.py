import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.stats import pearsonr

from loaders import create_dataloaders, generate_drug_splits

from .model import MLP, UGCNN, CombinedMLP

log = logging.getLogger(__name__)


class DrugResponsePredictionRunnerOld:
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
        self._seed_everything()

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

        # Check split configuration
        drug_proportion = self.cfg.task.get("drug_proportion")
        test_drug = self.cfg.task.get("test_drug")
        val_drug = self.cfg.task.get("val_drug")

        if drug_proportion is not None:
            if not (0 < drug_proportion <= 1):
                raise ValueError("drug_proportion must be between 0 and 1")
            # In proportion mode, test_drug and val_drug are ignored
            log.info(f"Using proportion mode: {drug_proportion:.1%} of drugs will be tested")
        else:
            # Specific mode validation
            if (test_drug is None) != (val_drug is None):
                raise ValueError("Both test_drug and val_drug must be set, or both must be null")
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
        log.info(f"Random seed set to {seed}")

        if self.cfg.task.get("fully_deterministic", False):
            log.info("Using fully deterministic algorithms")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

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

    def _run_single_split(self, test_drug: str | None, val_drug: str | None) -> Dict[str, Any]:
        """
        Run evaluation for a single test/val drug split.

        Args:
            test_drug: Test drug ID or None for cell line split
            val_drug: Validation drug ID or None for cell line split

        Returns:
            Dictionary containing evaluation metrics
        """
        split_name = test_drug if test_drug else "cell_line_split"
        log.info(f"Running split: test={test_drug}, val={val_drug}")

        train_loader, val_loader, test_loader, test_metadata, gexpr_dim = create_dataloaders(
            drug_info_file=Path(hydra.utils.to_absolute_path(self.cfg.task.paths.drug_info)),
            cell_line_info_file=Path(hydra.utils.to_absolute_path(self.cfg.task.paths.cell_line_info)),
            drug_feature_dir=Path(hydra.utils.to_absolute_path(self.cfg.task.paths.drug_features)),
            ground_truth_file=Path(hydra.utils.to_absolute_path(self.cfg.task.paths.ground_truth)),
            submission_file=Path(hydra.utils.to_absolute_path(self.cfg.task.paths.submission)),
            val_drug=val_drug,
            test_drug=test_drug,
            batch_size=self.cfg.task.get("batch_size", 256),
            num_workers=self.cfg.task.get("num_workers", 4),
        )

        # Create model
        drug_gcnn, gexpr_mlp, comb_mlp = self._create_model(gexpr_dim)

        # Setup training
        parameters = list(drug_gcnn.parameters()) + list(gexpr_mlp.parameters()) + list(comb_mlp.parameters())
        optimizer = torch.optim.Adam(
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

            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_models = (
                    drug_gcnn.state_dict().copy(),
                    gexpr_mlp.state_dict().copy(),
                    comb_mlp.state_dict().copy(),
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0:
                log.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val PCC: {val_pcc:.4f}")

            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best models
        drug_gcnn.load_state_dict(best_models[0])
        gexpr_mlp.load_state_dict(best_models[1])
        comb_mlp.load_state_dict(best_models[2])

        # Evaluate on test set
        test_loss, test_pcc, test_targets, test_predictions = self._evaluate(
            test_loader, drug_gcnn, gexpr_mlp, comb_mlp, criterion
        )

        log.info(f"Test PCC for {split_name}: {test_pcc:.4f}")

        result = {
            "test_drug": test_drug,
            "val_drug": val_drug,
            "pearson_correlation": float(test_pcc),
            "test_loss": float(test_loss),
            "best_val_pcc": float(best_val_pcc),
        }

        # Save predictions if configured
        if self.cfg.task.get("save_predictions", True):
            self._save_predictions(test_metadata, test_predictions, test_targets, test_pcc, test_drug, val_drug)

        return result

    def run(self) -> Dict[str, Any]:
        """
        Execute the drug response prediction evaluation.

        Returns:
            Dictionary containing evaluation metrics
        """
        log.info("Starting drug response prediction evaluation")

        # Generate all splits to run
        splits = generate_drug_splits(
            drug_info_file=Path(hydra.utils.to_absolute_path(self.cfg.task.paths.drug_info)),
            drug_proportion=self.cfg.task.get("drug_proportion"),
            test_drug=self.cfg.task.get("test_drug"),
            val_drug=self.cfg.task.get("val_drug"),
            seed=self.cfg.task.get("seed", 42),
        )

        # Run each split
        all_results = []
        for test_drug, val_drug in splits:
            result = self._run_single_split(test_drug, val_drug)
            all_results.append(result)

        # Aggregate results
        if len(all_results) == 1:
            return all_results[0]

        # Multiple splits: compute summary statistics
        pccs = [r["pearson_correlation"] for r in all_results]
        summary = {
            "mean_pearson_correlation": float(np.mean(pccs)),
            "std_pearson_correlation": float(np.std(pccs)),
            "min_pearson_correlation": float(np.min(pccs)),
            "max_pearson_correlation": float(np.max(pccs)),
            "n_splits": len(all_results),
            "individual_results": all_results,
        }

        log.info(f"Completed {len(all_results)} splits")
        log.info(f"Mean PCC: {summary['mean_pearson_correlation']:.4f} ± {summary['std_pearson_correlation']:.4f}")

        return summary

    def _save_predictions(
        self,
        metadata: List[Tuple[str, str, str]],
        predictions: np.ndarray,
        targets: np.ndarray,
        pcc: float,
        test_drug: str | None,
        val_drug: str | None,
    ) -> None:
        """Save predictions to CSV file."""
        output_dir = Path(self.cfg.task.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        split_name = test_drug if test_drug else "cell_line_split"
        output_file = output_dir / f"predictions_{split_name}.csv"

        df = pd.DataFrame({
            "cancer_type": [m[0] for m in metadata],
            "cell_line": [m[1] for m in metadata],
            "pubchem_cid": [m[2] for m in metadata],
            "prediction": predictions.flatten(),
            "ground_truth": targets.flatten(),
        })
        df.to_csv(output_file, index=False)
        log.info(f"Predictions saved to {output_file}")

        # Save summary metrics
        summary_file = output_dir / f"metrics_{split_name}.csv"
        summary_df = pd.DataFrame({
            "test_drug": [test_drug],
            "val_drug": [val_drug],
            "pearson_correlation": [pcc],
        })
        summary_df.to_csv(summary_file, index=False)
        log.info(f"Metrics saved to {summary_file}")
