import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ==========================================
# Tissue Lineages for Strict Leave-One-Tissue-Out (LOTO)
# ==========================================
TISSUE_GROUPS = {
    "LUNG": ["LUAD", "LUSC", "SCLC", "MESO", "NSCLC"],
    "GI": ["COREAD", "STAD", "PAAD", "ESCA", "LIHC", "BLCA", "READ", "COAD"],
    "BLOOD": ["LAML", "LCML", "DLBC", "ALL", "CLL", "MM", "NB", "Burkitt"],
    "SKIN": ["SKCM"],
    "CNS": ["GBM", "LGG", "MB", "NB"],
    "GYN": ["BRCA", "OV", "UCEC", "CESC"],
    "KIDNEY": ["KIRC", "KIRP", "KICH"],
    "BONE": ["EWS", "OS"],
    # Any label not here is treated as an independent group
}


class PharmacogenomicsDataset(Dataset):
    def __init__(self, indices: List[Tuple[int, int, float]], cell_tensor: torch.Tensor, drug_tensor: torch.Tensor):
        self.indices = indices
        # Store references to shared memory tensors
        self.cell_tensor = cell_tensor
        self.drug_tensor = drug_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        c_idx, d_idx, label = self.indices[idx]

        cell_vec = self.cell_tensor[c_idx]
        drug_vec = self.drug_tensor[d_idx]

        # Return d_idx to group by drug during evaluation
        return cell_vec, drug_vec, torch.tensor(label, dtype=torch.float32), d_idx


# ==========================================
# 2. The Data Manager
# ==========================================
class DataManager:
    def __init__(
        self,
        cell_embedding_path: str,
        drug_embedding_path: str,
        response_matrix_path: str,
        metadata_path: Optional[str] = None,
    ):
        log.info("Initializing DataManager...")

        # A. Load Cell Embeddings
        log.info(f"Loading Cell Embeddings: {cell_embedding_path}")
        self.cell_df = pd.read_csv(cell_embedding_path, index_col=0)
        self.cell_tensor = torch.tensor(self.cell_df.values, dtype=torch.float32)
        self.cell_id_to_idx = {str(cid): i for i, cid in enumerate(self.cell_df.index)}

        # B. Load Drug Embeddings
        log.info(f"Loading Drug Embeddings: {drug_embedding_path}")
        self.drug_df = pd.read_csv(drug_embedding_path)

        # Robust ID handling
        if "Drug_ID" in self.drug_df.columns:
            self.drug_df.set_index("Drug_ID", inplace=True)
        elif "DRUG_ID" in self.drug_df.columns:
            self.drug_df.set_index("DRUG_ID", inplace=True)
        else:
            self.drug_df.set_index(self.drug_df.columns[0], inplace=True)

        self.drug_tensor = torch.tensor(self.drug_df.values, dtype=torch.float32)
        self.drug_id_to_idx = {str(did): i for i, did in enumerate(self.drug_df.index)}

        # C. Load & Melt Response Matrix
        log.info(f"Processing Response Matrix: {response_matrix_path}")
        matrix_df = pd.read_csv(response_matrix_path)
        id_col = matrix_df.columns[0]

        self.response_df = matrix_df.melt(id_vars=id_col, var_name="cell_id", value_name="ic50")
        self.response_df.rename(columns={id_col: "drug_id"}, inplace=True)

        # Drop NaNs
        initial_len = len(self.response_df)
        self.response_df.dropna(subset=["ic50"], inplace=True)
        log.info(f"   Valid Interactions: {len(self.response_df)} (Dropped {initial_len - len(self.response_df)} NaNs)")

        # D. Load Metadata
        self.cell_to_cancer = {}
        if metadata_path:
            log.info(f"Loading Metadata: {metadata_path}")
            meta_df = pd.read_csv(metadata_path)
            cell_col = next((c for c in ["COSMIC_ID", "cell_line_name", "cell_id"] if c in meta_df.columns), None)
            type_col = next((c for c in ["TCGA_DESC", "cancer_type", "tissue"] if c in meta_df.columns), None)

            if cell_col and type_col:
                meta_unique = meta_df.drop_duplicates(subset=[cell_col])
                self.cell_to_cancer = dict(zip(meta_unique[cell_col].astype(str), meta_unique[type_col], strict=True))
                log.info(f"   Mapped {len(self.cell_to_cancer)} cell lines to cancer types.")
            else:
                log.warning("   Metadata columns not found. Stratification/LOTO will fail.")

    def get_aligned_indices(self) -> pd.DataFrame:
        self.response_df["drug_id"] = self.response_df["drug_id"].astype(str)
        self.response_df["cell_id"] = self.response_df["cell_id"].astype(str)

        valid_drugs = set(self.drug_id_to_idx.keys())
        valid_cells = set(self.cell_id_to_idx.keys())

        mask = (self.response_df["drug_id"].isin(valid_drugs)) & (self.response_df["cell_id"].isin(valid_cells))
        df_clean = self.response_df[mask].copy()

        df_clean["c_idx"] = df_clean["cell_id"].map(self.cell_id_to_idx)
        df_clean["d_idx"] = df_clean["drug_id"].map(self.drug_id_to_idx)
        df_clean["cancer_type"] = df_clean["cell_id"].map(self.cell_to_cancer)

        # Hygiene: Handle missing metadata
        df_clean["cancer_type"] = df_clean["cancer_type"].fillna("Unknown")
        df_clean["cancer_type"] = df_clean["cancer_type"].replace({"nan": "Unknown", "NA": "Unknown"})

        return df_clean

    def split_data(
        self,
        df: pd.DataFrame,
        mode: str = "random",
        test_drugs: Optional[List[str]] = None,
        val_drugs: Optional[List[str]] = None,
        test_cells: Optional[List[str]] = None,
        val_cells: Optional[List[str]] = None,
        holdout_tissue: Optional[str] = None,
        drug_prop: Optional[float] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        data_seed: int = 42,
    ):
        log.info(f"Splitting data... Mode: {mode}")

        # ---------------------------------------------------------
        # Method 1: Random Split (Transductive)
        # ---------------------------------------------------------
        if mode == "random":
            train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=data_seed)
            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=data_seed)

        # ---------------------------------------------------------
        # Method 2: Cold Drug Split (Inductive Chemistry)
        # ---------------------------------------------------------
        elif mode == "cold_drug":
            all_drugs = df["drug_id"].unique()

            # CASE A: Explicit Lists (K-Fold / Specific)
            if test_drugs is not None:
                # Convert string to list if necessary
                if isinstance(test_drugs, str):
                    test_drugs = [test_drugs]
                if val_drugs is not None and isinstance(val_drugs, str):
                    val_drugs = [val_drugs]

                # Auto-select val if not provided (random sample from remaining)
                if val_drugs is None:
                    remaining = [d for d in all_drugs if d not in test_drugs]
                    _, val_drugs = train_test_split(remaining, test_size=val_split, random_state=data_seed)

                log.info(f"   Cold Drug (Explicit): {len(test_drugs)} Test, {len(val_drugs)} Val.")

                test_df = df[df["drug_id"].isin(test_drugs)]
                val_df = df[df["drug_id"].isin(val_drugs)]

                # Train = Anything NOT in Test AND NOT in Val
                # Using Set operations for speed
                excluded = set(test_drugs) | set(val_drugs)
                train_df = df[~df["drug_id"].isin(excluded)]

            # CASE B: Random Proportion (Legacy)
            else:
                prop = drug_prop if drug_prop else test_split
                log.info(f"   Cold Drug (Random): Holding out {prop:.1%} of drugs.")

                # 1. Test Split
                remaining_drugs, t_drugs = train_test_split(all_drugs, test_size=prop, random_state=data_seed)
                # 2. Val Split (from remaining) - Ensures Val is also cold
                train_drugs, v_drugs = train_test_split(remaining_drugs, test_size=val_split, random_state=data_seed)

                train_df = df[df["drug_id"].isin(train_drugs)]
                val_df = df[df["drug_id"].isin(v_drugs)]
                test_df = df[df["drug_id"].isin(t_drugs)]

        # ---------------------------------------------------------
        # Method 3: Cold Cell Split (Inductive Biology)
        # ---------------------------------------------------------
        elif mode == "cold_cell" or mode == "cancer_stratified":
            # CASE A: Explicit Lists (K-Fold)
            if test_cells is not None:
                if val_cells is None:
                    val_cells = []
                log.info(f"   Cold Cell (Explicit): {len(test_cells)} Test, {len(val_cells)} Val.")

                test_df = df[df["cell_id"].isin(test_cells)]
                val_df = df[df["cell_id"].isin(val_cells)]

                excluded = set(test_cells) | set(val_cells)
                train_df = df[~df["cell_id"].isin(excluded)]

            # CASE B: Random Stratified
            else:
                try:
                    train_val_df, test_df = train_test_split(
                        df, test_size=test_split, stratify=df["cancer_type"], random_state=data_seed
                    )
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=val_split, stratify=train_val_df["cancer_type"], random_state=data_seed
                    )
                except ValueError:
                    log.warning("Stratification failed. Fallback to random.")
                    return self.split_data(df, mode="random", data_seed=data_seed)

        # ---------------------------------------------------------
        # Method 4: Double Cold Split (Zero-Shot)
        # ---------------------------------------------------------
        elif mode == "double_cold":
            # CASE A: Explicit Lists (K-Fold)
            if test_drugs is not None and test_cells is not None:
                # Train = NOT test_drug AND NOT test_cell
                train_mask = (~df["drug_id"].isin(test_drugs)) & (~df["cell_id"].isin(test_cells))
                train_val_df = df[train_mask]

                # Test = IS test_drug AND IS test_cell
                test_mask = df["drug_id"].isin(test_drugs) & df["cell_id"].isin(test_cells)
                test_df = df[test_mask]

                if len(test_df) == 0:
                    log.warning("Double Cold split resulted in empty test set (sparsity issue).")

                train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=data_seed)

            # CASE B: Random (Legacy)
            else:
                rng = np.random.default_rng(seed=data_seed)
                d_prop = drug_prop if drug_prop else 0.1
                all_drugs = df["drug_id"].unique()
                t_drugs = rng.choice(all_drugs, int(len(all_drugs) * d_prop), replace=False)

                all_cells = df["cell_id"].unique()
                t_cells = rng.choice(all_cells, int(len(all_cells) * 0.1), replace=False)  # 10% cells

                log.info(f"   Double Cold (Random): {len(t_drugs)} drugs x {len(t_cells)} cells block.")

                train_mask = (~df["drug_id"].isin(t_drugs)) & (~df["cell_id"].isin(t_cells))
                train_val_df = df[train_mask]

                test_mask = df["drug_id"].isin(t_drugs) & df["cell_id"].isin(t_cells)
                test_df = df[test_mask]

                train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=data_seed)

        # ---------------------------------------------------------
        # Method 5: Strict LOTO (OOD)
        # ---------------------------------------------------------
        elif mode == "loto":
            if not holdout_tissue:
                raise ValueError("Config Error: 'loto' mode requires 'test_tissue' to be set.")

            # Identify Lineage Group
            target_group = None
            exclude_labels = [holdout_tissue]
            for group_name, labels in TISSUE_GROUPS.items():
                if holdout_tissue in labels:
                    target_group = group_name
                    exclude_labels = labels
                    break

            unsafe_labels = ["UNCLASSIFIED", "Unknown", "NA", "nan"]

            log.info(f"   LOTO: Testing on '{holdout_tissue}'.")
            if target_group:
                log.info(f"   Strict LOTO: Excluding lineage {exclude_labels}")

            test_df = df[df["cancer_type"] == holdout_tissue]

            # Train on everything NOT in the excluded lineage AND NOT unsafe
            train_mask = (~df["cancer_type"].isin(exclude_labels)) & (~df["cancer_type"].isin(unsafe_labels))
            train_val_df = df[train_mask]

            if len(test_df) == 0:
                available = df["cancer_type"].unique()
                raise ValueError(f"Tissue '{holdout_tissue}' not found. Available: {available[:5]}...")

            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=data_seed)

        else:
            raise ValueError(f"Unknown split mode: {mode}")

        return train_df, val_df, test_df


# ==========================================
# 3. Helper to Build Loaders
# ==========================================
def get_drp_dataloaders(
    cell_path: str,
    drug_path: str,
    matrix_path: str,
    metadata_path: Optional[str],
    split_mode: str = "cancer_stratified",
    test_drugs: Optional[List[str]] = None,
    val_drugs: Optional[List[str]] = None,
    test_cells: Optional[List[str]] = None,
    val_cells: Optional[List[str]] = None,
    holdout_tissue: Optional[str] = None,
    drug_prop: Optional[float] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    data_seed: int = 42,
):
    manager = DataManager(cell_path, drug_path, matrix_path, metadata_path)
    df = manager.get_aligned_indices()

    train_df, val_df, test_df = manager.split_data(
        df,
        mode=split_mode,
        test_drugs=test_drugs,
        val_drugs=val_drugs,
        test_cells=test_cells,
        val_cells=val_cells,
        holdout_tissue=holdout_tissue,
        drug_prop=drug_prop,
        data_seed=data_seed,
    )

    log.info(f"Final Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def to_list(d):
        return list(zip(d["c_idx"], d["d_idx"], d["ic50"], strict=True))

    train_ds = PharmacogenomicsDataset(to_list(train_df), manager.cell_tensor, manager.drug_tensor)
    val_ds = PharmacogenomicsDataset(to_list(val_df), manager.cell_tensor, manager.drug_tensor)
    test_ds = PharmacogenomicsDataset(to_list(test_df), manager.cell_tensor, manager.drug_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dims = {"cell_dim": manager.cell_tensor.shape[1], "drug_dim": manager.drug_tensor.shape[1]}

    return train_loader, val_loader, test_loader, dims
