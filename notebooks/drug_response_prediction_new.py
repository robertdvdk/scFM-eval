import logging
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ==========================================
# 1. The Lazy Dataset (High Performance)
# ==========================================
class PharmacogenomicsDataset(Dataset):
    def __init__(self, indices: List[Tuple[int, int, float]], cell_tensor: torch.Tensor, drug_tensor: torch.Tensor):
        """
        Args:
            indices: List of (cell_idx, drug_idx, label)
            cell_tensor:  Dense tensor of shape (N_cells, Cell_Dim)
            drug_tensor:  Dense tensor of shape (N_drugs, Drug_Dim)
        """
        self.indices = indices
        # We store references to the large tensors, not copies.
        self.cell_tensor = cell_tensor
        self.drug_tensor = drug_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1. Retrieve integer indices for this specific experiment
        c_idx, d_idx, label = self.indices[idx]

        # 2. Lazy Lookup (O(1) pointer arithmetic)
        # This is where the speed comes from - no dataframe lookups in the loop
        cell_vec = self.cell_tensor[c_idx]
        drug_vec = self.drug_tensor[d_idx]

        return cell_vec, drug_vec, torch.tensor(label, dtype=torch.float32)


# ==========================================
# 2. The Data Manager (Logic Core)
# ==========================================
class DataManager:
    def __init__(
        self,
        cell_embedding_path: str,
        drug_embedding_path: str,
        response_matrix_path: str,
        metadata_path: Optional[str] = None,
    ):
        """
        Args:
            cell_embedding_path: Path to CancerGPT.csv (Cells)
            drug_embedding_path: Path to drug_embeddings.csv (Drugs)
            response_matrix_path: Path to dose_response_matrix.csv (Wide format)
            metadata_path: Path to dose_response_subset.csv (For TCGA labels)
        """
        log.info("Initializing DataManager...")

        # -------------------------------------------------------
        # A. Load Cell Embeddings (CancerGPT.csv)
        # -------------------------------------------------------
        log.info(f"Loading Cell Embeddings: {cell_embedding_path}")
        # index_col=0 assumes the first column is the Cell ID (e.g., 683667)
        self.cell_df = pd.read_csv(cell_embedding_path, index_col=0)
        self.cell_tensor = torch.tensor(self.cell_df.values, dtype=torch.float32)

        # Map Cell Name (str) -> Row Index (int)
        # We cast index to str to ensure matching works even if CSV has ints
        self.cell_id_to_idx = {str(cid): i for i, cid in enumerate(self.cell_df.index)}

        # -------------------------------------------------------
        # B. Load Drug Embeddings
        # -------------------------------------------------------
        log.info(f"Loading Drug Embeddings: {drug_embedding_path}")
        # Assumes 'Drug_ID' is the identifier
        self.drug_df = pd.read_csv(drug_embedding_path)
        if "Drug_ID" in self.drug_df.columns:
            self.drug_df.set_index("Drug_ID", inplace=True)
        elif "DRUG_ID" in self.drug_df.columns:
            self.drug_df.set_index("DRUG_ID", inplace=True)

        self.drug_tensor = torch.tensor(self.drug_df.values, dtype=torch.float32)
        self.drug_id_to_idx = {str(did): i for i, did in enumerate(self.drug_df.index)}

        # -------------------------------------------------------
        # C. Load & Melt Response Matrix
        # -------------------------------------------------------
        log.info(f"Processing Response Matrix: {response_matrix_path}")
        # This file is wide: Rows=Drugs, Cols=Cells
        matrix_df = pd.read_csv(response_matrix_path)

        # Identify the Drug ID column (usually the first one)
        id_col = matrix_df.columns[0]

        # MELT: Convert Matrix -> List of (Drug, Cell, IC50)
        self.response_df = matrix_df.melt(id_vars=id_col, var_name="cell_id", value_name="ic50")
        self.response_df.rename(columns={id_col: "drug_id"}, inplace=True)

        # Clean: Drop NaNs (Untested pairs)
        initial_len = len(self.response_df)
        self.response_df.dropna(subset=["ic50"], inplace=True)
        log.info(
            f"   Matrix Melted. Valid Experiments: {len(self.response_df)}"
            f"   (Dropped {initial_len - len(self.response_df)} NaNs)"
        )

        # -------------------------------------------------------
        # D. Load Metadata (For Stratification)
        # -------------------------------------------------------
        self.cell_to_cancer = {}
        if metadata_path:
            log.info(f"Loading Metadata for Stratification: {metadata_path}")
            meta_df = pd.read_csv(metadata_path)
            # We expect columns like 'COSMIC_ID' (Cell ID) and 'TCGA_DESC' (Cancer Type)
            # Adjust these column names if your file is slightly different
            cell_col = "COSMIC_ID" if "COSMIC_ID" in meta_df.columns else "cell_line_name"
            type_col = "TCGA_DESC" if "TCGA_DESC" in meta_df.columns else "cancer_type"

            if cell_col in meta_df.columns and type_col in meta_df.columns:
                # specific logic to drop duplicates so we have unique cell->type map
                meta_unique = meta_df.drop_duplicates(subset=[cell_col])
                self.cell_to_cancer = dict(zip(meta_unique[cell_col].astype(str), meta_unique[type_col], strict=True))
            else:
                log.warning("   Could not find 'COSMIC_ID'/'TCGA_DESC' in metadata. Stratification will be random.")

    def get_aligned_indices(self) -> pd.DataFrame:
        """
        Creates the master index table.
        Filters out interactions where we lack embeddings for either the drug or the cell.
        """
        # Convert IDs to string for consistent matching
        self.response_df["drug_id"] = self.response_df["drug_id"].astype(str)
        self.response_df["cell_id"] = self.response_df["cell_id"].astype(str)

        # 1. Intersection Filter
        # Only keep rows where Drug ID exists in Drug CSV AND Cell ID exists in Cell CSV
        valid_drugs = set(self.drug_id_to_idx.keys())
        valid_cells = set(self.cell_id_to_idx.keys())

        mask = (self.response_df["drug_id"].isin(valid_drugs)) & (self.response_df["cell_id"].isin(valid_cells))

        df_clean = self.response_df[mask].copy()

        # 2. Map String IDs to Integer Indices (The "Pointers")
        df_clean["c_idx"] = df_clean["cell_id"].map(self.cell_id_to_idx)
        df_clean["d_idx"] = df_clean["drug_id"].map(self.drug_id_to_idx)

        # 3. Map Cancer Type (for splitting)
        # Default to 'Unknown' if not in metadata or no metadata provided
        df_clean["cancer_type"] = df_clean["cell_id"].map(self.cell_to_cancer).fillna("Unknown")

        return df_clean

    def split_data(
        self,
        df: pd.DataFrame,
        mode: str = "cancer_stratified",
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        The Sophisticated Splitting Logic.
        """
        log.info(f"Splitting data... Mode: {mode}")

        # --- Mode A: Cold Drug Split ---
        # "Can we predict response for a completely new chemical?"
        if mode == "cold_drug":
            all_drugs = df["drug_id"].unique()
            train_drugs, test_drugs = train_test_split(all_drugs, test_size=test_split, random_state=seed)

            # Further split train drugs to get validation drugs
            # We want validation to ALSO be unseen drugs to simulate the test case
            train_drugs_final, val_drugs = train_test_split(train_drugs, test_size=val_split, random_state=seed)

            train_df = df[df["drug_id"].isin(train_drugs_final)]
            val_df = df[df["drug_id"].isin(val_drugs)]
            test_df = df[df["drug_id"].isin(test_drugs)]

        # --- Mode B: Stratified Cell Split ---
        # "Can we predict response for new cells, balancing across tissue types?"
        elif mode == "cancer_stratified":
            # We split based on 'cancer_type' column
            try:
                # 1. Split Train+Val / Test
                train_val_df, test_df = train_test_split(
                    df, test_size=test_split, stratify=df["cancer_type"], random_state=seed
                )
                # 2. Split Train / Val
                train_df, val_df = train_test_split(
                    train_val_df, test_size=val_split, stratify=train_val_df["cancer_type"], random_state=seed
                )
            except ValueError as e:
                # This happens if a cancer type has too few samples (e.g., < 2)
                log.warning(f"Stratification failed (likely rare cancer types). Fallback to Random. Error: {e}")
                return self.split_data(df, mode="random")

        # --- Mode C: Random Split ---
        else:
            train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)
            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

        return train_df, val_df, test_df


# ==========================================
# 3. Helper to Build Loaders
# ==========================================
def get_dataloaders(
    cell_path: str,
    drug_path: str,
    matrix_path: str,
    metadata_path: Optional[str],
    split_mode: str = "cancer_stratified",
    batch_size: int = 32,
):
    # 1. Setup Manager
    manager = DataManager(cell_path, drug_path, matrix_path, metadata_path)

    # 2. Get Index Table
    df = manager.get_aligned_indices()

    # 3. Split
    train_df, val_df, test_df = manager.split_data(df, mode=split_mode)

    # 4. Helper to convert DataFrame to List of Tuples
    def to_list(d):
        return list(zip(d["c_idx"], d["d_idx"], d["ic50"], strict=True))

    # 5. Build Datasets
    # Note: We pass the tensor references from the manager
    train_ds = PharmacogenomicsDataset(to_list(train_df), manager.cell_tensor, manager.drug_tensor)
    val_ds = PharmacogenomicsDataset(to_list(val_df), manager.cell_tensor, manager.drug_tensor)
    test_ds = PharmacogenomicsDataset(to_list(test_df), manager.cell_tensor, manager.drug_tensor)

    # 6. Build Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Return loaders + input dimensions for model creation
    dims = {"cell_dim": manager.cell_tensor.shape[1], "drug_dim": manager.drug_tensor.shape[1]}

    return train_loader, val_loader, test_loader, dims


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Example paths based on your file names
    train_loader, val_loader, test_loader, dims = get_dataloaders(
        cell_path="./gdsc/processed_data/CancerGPT.csv",
        drug_path="./gdsc/processed_data/drug_embeddings.csv",
        matrix_path="./gdsc/processed_data/dose_response_matrix.csv",
        metadata_path="./gdsc/processed_data/dose_response_subset.csv",  # Used for looking up 'MB' etc.
        split_mode="cancer_stratified",  # or 'cold_drug'
        batch_size=64,
    )

    print(f"Loaders ready. Cell Dim: {dims['cell_dim']}, Drug Dim: {dims['drug_dim']}")

    # Test a batch
    for batch_cell, batch_drug, batch_y in enumerate(train_loader):
        print(f"Batch Shapes: Cell {batch_cell.shape}, Drug {batch_drug.shape}, Label {batch_y.shape}")
