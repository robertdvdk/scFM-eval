import logging
from typing import Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class RNAtoProteomeDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, cell_dim: int, prot_dim: int):
        # Store references to shared memory tensors
        self.data_tensor = data_tensor
        self.cell_dim = cell_dim

        assert cell_dim + prot_dim == data_tensor.shape[1]

    def __len__(self):
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        cell_vec = self.data_tensor[idx, : self.cell_dim]
        protein_vec = self.data_tensor[idx, self.cell_dim :]

        return cell_vec, protein_vec


# ==========================================
# 2. The Data Manager
# ==========================================
class DataManager:
    def __init__(
        self,
        cell_embedding_path: str,
        protein_values_path: str,
    ):
        log.info("Initializing DataManager...")

        # A. Load Cell Embeddings
        log.info(f"Loading Cell Embeddings: {cell_embedding_path}")
        self.cell_df = pd.read_csv(cell_embedding_path, index_col=0)

        # B. Load Protein Values
        log.info(f"Loading Protein Values: {protein_values_path}")
        self.prot_df = pd.read_csv(protein_values_path, index_col=0)

    def get_aligned_indices(self) -> Tuple[pd.DataFrame, int, int]:
        full_df = self.cell_df.merge(self.prot_df, left_index=True, right_index=True)
        cell_dim = self.cell_df.shape[1]
        prot_dim = self.prot_df.shape[1]

        return full_df, cell_dim, prot_dim

    def split_data(
        self,
        df: pd.DataFrame,
        val_split: float = 0.1,
        test_split: float = 0.1,
        data_seed: int = 42,
    ):
        train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=data_seed)
        train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=data_seed)
        return train_df, val_df, test_df


# ==========================================
# 3. Helper to Build Loaders
# ==========================================
def get_prot_dataloaders(
    cell_path: str,
    prot_path: str,
    batch_size: int,
    num_workers: int,
    data_seed: int,
):
    manager = DataManager(cell_path, prot_path)
    df, cell_dim, prot_dim = manager.get_aligned_indices()

    train_df, val_df, test_df = manager.split_data(
        df,
        data_seed=data_seed,
    )

    train_tens = torch.Tensor(train_df.values)
    val_tens = torch.Tensor(val_df.values)
    test_tens = torch.Tensor(test_df.values)

    log.info(f"Final Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_ds = RNAtoProteomeDataset(train_tens, cell_dim=cell_dim, prot_dim=prot_dim)
    val_ds = RNAtoProteomeDataset(val_tens, cell_dim=cell_dim, prot_dim=prot_dim)
    test_ds = RNAtoProteomeDataset(test_tens, cell_dim=cell_dim, prot_dim=prot_dim)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, cell_dim, prot_dim, train_df, val_df, test_df
