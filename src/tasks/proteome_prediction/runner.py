import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from loaders import get_prot_dataloaders

from .model import ProteomePredictionModel

log = logging.getLogger(__name__)


class ProteomePredictionRunner:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._seed_everything()

    def _seed_everything(self):
        seed = self.cfg.task.model_seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        log.info(f"Random seed set to {seed}")

    def train_model(self, submission, train_loader, val_loader, test_loader, cell_dim, prot_dim, test_df):
        model = ProteomePredictionModel(cell_dim=cell_dim, protein_dim=prot_dim).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.MSELoss()

        for epoch in range(5):
            model.train()
            running_train_loss, n_train_samples = 0.0, 0
            for X, y in train_loader:
                X, y = X.cuda(), y.cuda()
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(y, pred)
                running_train_loss += loss.item() * X.shape[0]
                n_train_samples += X.shape[0]
                loss.backward()
                optimizer.step()

            if epoch % 1 == 0:
                model.eval()
                running_val_loss, n_val_samples = 0.0, 0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.cuda(), y.cuda()
                        pred = model(X)
                        loss = loss_fn(y, pred)
                        running_val_loss += loss.item() * X.shape[0]
                        n_val_samples += X.shape[0]

                log.info(
                    f"Epoch {epoch}. Training loss: {running_train_loss / n_train_samples:.4f}. "
                    f"Validation loss: {running_val_loss / n_val_samples:.4f}"
                )

        self.all_results_dfs[submission] = pd.DataFrame(columns=test_df.columns[cell_dim:], index=test_df.index)
        model.eval()
        running_loss, n_samples = 0.0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                pred = model(X)
                loss = loss_fn(y, pred)
                running_loss += loss.item() * X.shape[0]
                n_samples += X.shape[0]
                self.all_results_dfs[submission][i * self.cfg.task.batch_size : (i + 1) * self.cfg.task.batch_size] = (
                    pred.cpu().numpy()
                )
        log.info(f"Test loss: {running_loss / n_samples:.4f}")

    def run(self) -> str:
        log.info(f"Running task: {self.cfg.task.name}")

        self.all_results_dfs = dict()
        self.all_metrics_dfs = dict()
        for submission in self.cfg.task.data.submission:
            log.info(f"Running submission: {submission.split('/')[1]}")
            train_loader, val_loader, test_loader, cell_dim, prot_dim, train_df, val_df, test_df = get_prot_dataloaders(
                cell_path=self.cfg.task.data.data_root + submission + ".csv",
                prot_path=self.cfg.task.data.data_root + self.cfg.task.data.prot_path + ".csv",
                batch_size=self.cfg.task.batch_size,
                num_workers=self.cfg.task.num_workers,
                data_seed=self.cfg.task.data_seed,
            )
            log.info(f"Input dimension: {cell_dim}, output dimension: {prot_dim}")

            if submission == "submission/mean":
                y_train = train_df.iloc[:, cell_dim:].to_numpy(dtype=np.float32)
                y_test = test_df.iloc[:, cell_dim:].to_numpy(dtype=np.float32)

                mean_vector = np.mean(y_train, axis=0)

                y_pred = np.tile(mean_vector, (y_test.shape[0], 1))

                # Convert to tensors for metric calculation
                res = torch.from_numpy(y_pred)
                gt = torch.from_numpy(y_test)

                # Pearson Correlation will be 0 or NaN because variance of res is 0.
                pcc = np.zeros(prot_dim)
                cos_sim = torch.cosine_similarity(res, gt, dim=0)  # Per-cell similarity

                # Mean Squared Error (Total Average)
                mse = np.mean((y_pred - y_test) ** 2)
                log.info(f"Mean Predictor Global MSE: {mse:.4f}")

                self.all_metrics_dfs[submission] = pd.DataFrame({
                    "Cosine Similarity": cos_sim.cpu().numpy(),
                    "Pearson Correlation": pcc,
                })

                self.all_results_dfs[submission] = pd.DataFrame(
                    y_pred, columns=test_df.columns[cell_dim:], index=test_df.index
                )
            elif submission == "submission/LinReg":
                X_train = train_df.iloc[:, :cell_dim].to_numpy(dtype=np.float32)
                y_train = train_df.iloc[:, cell_dim:].to_numpy(dtype=np.float32)
                X_test = test_df.iloc[:, :cell_dim].to_numpy(dtype=np.float32)
                y_test = test_df.iloc[:, cell_dim:].to_numpy(dtype=np.float32)

                scaler_x = StandardScaler()
                scaler_y = StandardScaler()

                X_train_s = scaler_x.fit_transform(X_train)
                y_train_s = scaler_y.fit_transform(y_train)
                X_test_s = scaler_x.transform(X_test)

                reg = LinearRegression().fit(X_train_s, y_train_s)

                pred_s = reg.predict(X_test_s)
                pred_raw = scaler_y.inverse_transform(pred_s)

                res = torch.from_numpy(pred_raw)
                gt = torch.from_numpy(y_test)

                mse = np.mean((pred_raw - y_test) ** 2)
                log.info(f"Linear Regression Global MSE: {mse:.4f}")

                # Pearson R per protein (axis=0)
                pcc = stats.pearsonr(res, gt, axis=0)
                # Cosine Similarity per protein
                cos_sim = torch.cosine_similarity(res.T, gt.T)

                self.all_metrics_dfs[submission] = pd.DataFrame({
                    "Cosine Similarity": cos_sim.cpu().numpy(),
                    "Pearson Correlation": pcc.statistic,
                })

                # Store results for plotting/comparison
                self.all_results_dfs[submission] = pd.DataFrame(
                    pred_raw, columns=test_df.columns[cell_dim:], index=test_df.index
                )
            else:
                self.train_model(submission, train_loader, val_loader, test_loader, cell_dim, prot_dim, test_df)

                res = torch.from_numpy(self.all_results_dfs[submission].to_numpy(dtype=np.float32))
                gt = torch.from_numpy(test_df.iloc[:, cell_dim:].to_numpy(dtype=np.float32))

                pcc = stats.pearsonr(res, gt, axis=0)
                cos_sim = torch.cosine_similarity(res.T, gt.T)
                self.all_metrics_dfs[submission] = pd.DataFrame({
                    "Cosine Similarity": cos_sim.cpu().numpy(),
                    "Pearson Correlation": pcc.statistic,
                })

        # 1. Prepare long-form DataFrame
        plot_list = []
        for sub_name, df in self.all_metrics_dfs.items():
            display_name = sub_name.split("/")[-1]
            temp_df = df.copy()
            temp_df["Method"] = display_name
            plot_list.append(temp_df)

        full_plot_df = pd.concat(plot_list, ignore_index=True)

        # 2. Setup Subplots: 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        metrics = ["Pearson Correlation", "Cosine Similarity"]
        colors = sns.color_palette("Set2", len(self.all_metrics_dfs))

        for i, metric in enumerate(metrics):
            sns.boxplot(
                data=full_plot_df,
                x="Method",
                y=metric,
                hue="Method",
                ax=axes[i],  # Direct the plot to the specific subplot axis
                palette=colors,
                showfliers=False,  # Removes extreme outliers for visual clarity
                legend=(i == 1),  # Only show legend on the second plot to avoid redundancy
            )

            axes[i].set_title(metric)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            axes[i].grid(axis="y", linestyle="--", alpha=0.5)

        # 3. Refine Layout and Save
        if axes[1].get_legend():
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Submissions")

        plt.tight_layout()
        plt.savefig("model_comparison_metrics.png", dpi=300)
        log.info("Combined subplot saved to model_comparison_metrics.png")
        plt.close()
        return
