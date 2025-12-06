import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. Load and Align Data (Pandas Style)
# ==========================================
print("1. Loading Data...")

# A. Convert H5AD to DataFrame (In-memory CSV)
adata = sc.read_h5ad("./gdsc/processed_data/CancerGPT_epoch_15_expr.h5ad")
df_cell = adata.to_df()  # Converts X to a dataframe with cell_ids as index
print(f"   Cells loaded: {df_cell.shape}")

# B. Load Drugs
df_drug = pd.read_csv("./gdsc/processed_data/drug_embeddings.csv", index_col=0)
print(f"   Drugs loaded: {df_drug.shape}")

# C. Load, Melt, and Clean Response Matrix
# ---------------------------------------------------------
print("   Loading Response Matrix...")
resp_matrix = pd.read_csv("./gdsc/processed_data/dose_response_matrix.csv", index_col=0)

# 1. Move the row index (Drugs) into the dataframe as a column
df_reset = resp_matrix.reset_index()
id_col_name = df_reset.columns[0]  # Detects "Drug", "ID", etc.

# 2. Melt to Long Format
df_response = df_reset.melt(id_vars=id_col_name, var_name="cell_id", value_name="ic50")
df_response.rename(columns={id_col_name: "drug_id"}, inplace=True)

# 3. [CRITICAL FIX] Drop NaN values
# This removes rows where the drug was not tested on the cell line.
# Without this, your Loss function will become NaN.
initial_len = len(df_response)
df_response = df_response.dropna(subset=["ic50"])
dropped_len = initial_len - len(df_response)

print(f"   Cleaned NaNs: Dropped {dropped_len} empty experiments. Remaining: {len(df_response)}")

# 4. [OPTIONAL SAFETY] Force numeric types
# Sometimes 'NaN' strings or artifacts exist. This forces them to real NaNs and drops them.
df_response["ic50"] = pd.to_numeric(df_response["ic50"], errors="coerce")
df_response = df_response.dropna(subset=["ic50"])
# ---------------------------------------------------------

# D. The "Master Merge"
# Inner join to keep only valid triplets
print("2. Merging tables...")
data = df_response.merge(df_cell, left_on="cell_id", right_index=True)
data = data.merge(df_drug, left_on="drug_id", right_index=True)

# Separate Features (X) and Target (y)
# Drop metadata columns to leave only the features
metadata_cols = ["cell_id", "drug_id", "ic50"]
feature_cols = [c for c in data.columns if c not in metadata_cols]

X = data[feature_cols].values
y = data["ic50"].values

print(f"   Final Training Data Shape: {X.shape}")

# ==========================================
# 2. Train Simple Model
# ==========================================

# Standardize Targets
scaler = StandardScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split
# ==========================================
# REPLACEMENT: Cold Split Logic
# ==========================================
print("Performing Cold Drug Split (Strict Generalization Test)...")

# 1. Extract unique drugs
unique_drugs = data["drug_id"].unique()

# 2. Split the DRUGS, not the rows
train_drugs, test_drugs = train_test_split(unique_drugs, test_size=0.2, random_state=42)

# 3. Create boolean masks
# Train on rows where drug is in train_drugs
# Test on rows where drug is in test_drugs
train_mask = data["drug_id"].isin(train_drugs)
test_mask = data["drug_id"].isin(test_drugs)

# 4. Apply split
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"   Train Drugs: {len(train_drugs)} | Test Drugs: {len(test_drugs)}")
print(f"   Train Rows: {len(X_train)} | Test Rows: {len(X_test)}")

# PyTorch Tensors
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

# Simple MLP
model = nn.Sequential(
    nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("\n3. Training with Pearson Metric...")

# Lists to store history for plotting later if needed
history = {"train_loss": [], "val_loss": [], "val_pearson": []}

for epoch in range(10):  # Increased to 10 epochs to see convergence
    # --- TRAIN ---
    model.train()
    batch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X).squeeze()
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds = model(batch_X).squeeze()

            # Accumulate loss
            val_loss += criterion(preds, batch_y).item()

            # Store predictions and targets for Pearson calculation
            # We must move to CPU and convert to numpy
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Calculate Metrics
    epoch_train_loss = np.mean(batch_losses)
    epoch_val_loss = val_loss / len(test_loader)

    # Pearson R calculation
    # Note: Pearson is invariant to scaling, so we don't need to inverse-transform the data first
    r_value, p_value = pearsonr(all_targets, all_preds)

    history["val_pearson"].append(r_value)

    print(
        f"Epoch {epoch + 1:02d} | Train MSE: {epoch_train_loss:.4f} "
        f"| Val MSE: {epoch_val_loss:.4f} | Val Pearson r: {r_value:.4f}"
    )

print(f"\nFinal Result - Correlation: {history['val_pearson'][-1]:.4f}")
