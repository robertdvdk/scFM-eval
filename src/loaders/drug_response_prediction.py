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
# CONSTANTS: Tissue Lineages for Strict LOTO
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
}


# ==========================================
# 1. The Lazy Dataset
# ==========================================
class PharmacogenomicsDataset(Dataset):
    def __init__(self, indices: List[Tuple[int, int, float]], cell_tensor: torch.Tensor, drug_tensor: torch.Tensor):
        self.indices = indices
        self.cell_tensor = cell_tensor
        self.drug_tensor = drug_tensor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        c_idx, d_idx, label = self.indices[idx]
        cell_vec = self.cell_tensor[c_idx]
        drug_vec = self.drug_tensor[d_idx]
        # Return d_idx so we know which drug this is (useful for per-drug evaluation)
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

        # Hygiene
        df_clean["cancer_type"] = df_clean["cancer_type"].fillna("Unknown")
        df_clean["cancer_type"] = df_clean["cancer_type"].replace({"nan": "Unknown", "NA": "Unknown"})

        return df_clean

    def split_data(
        self,
        df: pd.DataFrame,
        mode: str = "random",
        test_drug: Optional[str] = None,
        val_drug: Optional[str] = None,
        holdout_tissue: Optional[str] = None,
        drug_prop: Optional[float] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        log.info(f"Splitting data... Mode: {mode}")

        # ---------------------------------------------------------
        # Method 1: Random Split
        # ---------------------------------------------------------
        if mode == "random":
            train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)
            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

        # ---------------------------------------------------------
        # Method 2: Cold Drug Split (Corrected for Unseen Validation)
        # ---------------------------------------------------------
        elif mode == "cold_drug":
            all_drugs = df["drug_id"].unique()

            if test_drug:
                # A. Specific Drug Case
                # We expect val_drug to be passed, or we must pick one from the remaining?
                # If val_drug is None, we need to pick a random one from remaining to ensure 'cold' val

                log.info(f"   Specific Cold Drug. Test: {test_drug}")

                if not val_drug:
                    # Auto-select a validation drug if not provided to ensure cold validation
                    remaining_candidates = [d for d in all_drugs if d != str(test_drug)]
                    np.random.seed(seed)
                    val_drug = np.random.choice(remaining_candidates)
                    log.info(f"   No val_drug provided. Auto-selected {val_drug} as holdout validation drug.")

                log.info(f"   Validation Drug: {val_drug}")

                test_df = df[df["drug_id"] == str(test_drug)]
                val_df = df[df["drug_id"] == str(val_drug)]

                # Train on everything else
                train_df = df[(df["drug_id"] != str(test_drug)) & (df["drug_id"] != str(val_drug))]

            else:
                # B. Random Proportion Case (FIXED)
                prop = drug_prop if drug_prop else test_split

                # 1. Split off Test Drugs
                remaining_drugs, test_drugs = train_test_split(all_drugs, test_size=prop, random_state=seed)

                # 2. Split off Validation Drugs from the REMAINING list
                # This ensures Val drugs are disjoint from Train AND Test
                train_drugs, val_drugs = train_test_split(remaining_drugs, test_size=val_split, random_state=seed)

                log.info(
                    f"   Cold Drug Proportion: {len(test_drugs)} Test Drugs, "
                    f"{len(val_drugs)} Val Drugs, {len(train_drugs)} Train Drugs."
                )

                # 3. Create sets
                train_df = df[df["drug_id"].isin(train_drugs)]
                val_df = df[df["drug_id"].isin(val_drugs)]
                test_df = df[df["drug_id"].isin(test_drugs)]

        # ---------------------------------------------------------
        # Method 3: Cold Cell Split
        # ---------------------------------------------------------
        elif mode == "cold_cell" or mode == "cancer_stratified":
            try:
                train_val_df, test_df = train_test_split(
                    df, test_size=test_split, stratify=df["cancer_type"], random_state=seed
                )
                train_df, val_df = train_test_split(
                    train_val_df, test_size=val_split, stratify=train_val_df["cancer_type"], random_state=seed
                )
            except ValueError:
                log.warning("Stratification failed. Fallback to random.")
                return self.split_data(df, mode="random", seed=seed)

        # ---------------------------------------------------------
        # Method 4: Double Cold Split
        # ---------------------------------------------------------
        elif mode == "double_cold":
            d_prop = drug_prop if drug_prop else 0.1
            all_drugs = df["drug_id"].unique()
            train_drugs, test_drugs = train_test_split(all_drugs, test_size=d_prop, random_state=seed)

            all_cells = df["cell_id"].unique()
            train_cells, test_cells = train_test_split(all_cells, test_size=0.1, random_state=seed)

            log.info(f"   Double Cold: {len(test_drugs)} drugs x {len(test_cells)} cells in Test Block.")

            train_mask = df["drug_id"].isin(train_drugs) & df["cell_id"].isin(train_cells)
            train_val_df = df[train_mask]

            test_mask = df["drug_id"].isin(test_drugs) & df["cell_id"].isin(test_cells)
            test_df = df[test_mask]

            if len(test_df) == 0:
                log.warning("Double Cold split result is empty! Increase drug_prop.")

            # For Double Cold, standard validation on Seen Data is acceptable to tune the head,
            # unless you want to burn another block of Cold Drugs/Cells for validation (expensive).
            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

        # ---------------------------------------------------------
        # Method 5: Strict LOTO
        # ---------------------------------------------------------
        elif mode == "loto":
            if not holdout_tissue:
                raise ValueError("Config Error: 'loto' mode requires 'test_tissue' to be set.")

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

            # Train on everything NOT in the excluded lineage
            train_mask = (~df["cancer_type"].isin(exclude_labels)) & (~df["cancer_type"].isin(unsafe_labels))
            train_val_df = df[train_mask]

            if len(test_df) == 0:
                available = df["cancer_type"].unique()
                raise ValueError(f"Tissue '{holdout_tissue}' not found. Available: {available[:5]}...")

            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)
        else:
            raise ValueError(f"Unknown split mode: {mode}")

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
    test_drug: Optional[str] = None,
    val_drug: Optional[str] = None,
    holdout_tissue: Optional[str] = None,
    drug_prop: Optional[float] = None,
    batch_size: int = 32,
):
    manager = DataManager(cell_path, drug_path, matrix_path, metadata_path)
    df = manager.get_aligned_indices()

    train_df, val_df, test_df = manager.split_data(
        df, mode=split_mode, test_drug=test_drug, val_drug=val_drug, holdout_tissue=holdout_tissue, drug_prop=drug_prop
    )

    log.info(f"Final Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def to_list(d):
        return list(zip(d["c_idx"], d["d_idx"], d["ic50"], strict=True))

    train_ds = PharmacogenomicsDataset(to_list(train_df), manager.cell_tensor, manager.drug_tensor)
    val_ds = PharmacogenomicsDataset(to_list(val_df), manager.cell_tensor, manager.drug_tensor)
    test_ds = PharmacogenomicsDataset(to_list(test_df), manager.cell_tensor, manager.drug_tensor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    dims = {"cell_dim": manager.cell_tensor.shape[1], "drug_dim": manager.drug_tensor.shape[1]}

    return train_loader, val_loader, test_loader, dims
