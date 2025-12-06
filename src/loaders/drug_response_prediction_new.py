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
    # Any label not here is treated as an independent group
}


# ==========================================
# 1. The Lazy Dataset
# ==========================================
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
        # Return d_idx so we know which drug this is
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

        # D. Load Metadata (Stratification/LOTO)
        self.cell_to_cancer = {}
        if metadata_path:
            log.info(f"Loading Metadata: {metadata_path}")
            meta_df = pd.read_csv(metadata_path)
            # Adjust column names dynamically
            cell_col = next((c for c in ["COSMIC_ID", "cell_line_name", "cell_id"] if c in meta_df.columns), None)
            type_col = next((c for c in ["TCGA_DESC", "cancer_type", "tissue"] if c in meta_df.columns), None)

            if cell_col and type_col:
                # Ensure unique mapping
                meta_unique = meta_df.drop_duplicates(subset=[cell_col])
                self.cell_to_cancer = dict(zip(meta_unique[cell_col].astype(str), meta_unique[type_col], strict=True))
                log.info(f"   Mapped {len(self.cell_to_cancer)} cell lines to cancer types.")
            else:
                log.warning("   Metadata columns not found. Stratification/LOTO will fail.")

    def get_aligned_indices(self) -> pd.DataFrame:
        """Create master index table with integer pointers."""
        # Ensure strings for matching
        self.response_df["drug_id"] = self.response_df["drug_id"].astype(str)
        self.response_df["cell_id"] = self.response_df["cell_id"].astype(str)

        valid_drugs = set(self.drug_id_to_idx.keys())
        valid_cells = set(self.cell_id_to_idx.keys())

        mask = (self.response_df["drug_id"].isin(valid_drugs)) & (self.response_df["cell_id"].isin(valid_cells))

        df_clean = self.response_df[mask].copy()

        # Map to pointers
        df_clean["c_idx"] = df_clean["cell_id"].map(self.cell_id_to_idx)
        df_clean["d_idx"] = df_clean["drug_id"].map(self.drug_id_to_idx)

        # Map Cancer Types & Handle Missing/NA
        df_clean["cancer_type"] = df_clean["cell_id"].map(self.cell_to_cancer)

        # HYGIENE: Treat 'nan', 'NA' as 'Unknown' so we don't lose data,
        # but can exclude them from LOTO training.
        df_clean["cancer_type"] = df_clean["cancer_type"].fillna("Unknown")
        df_clean["cancer_type"] = df_clean["cancer_type"].replace({"nan": "Unknown", "NA": "Unknown"})

        return df_clean

    def split_data(
        self,
        df: pd.DataFrame,
        mode: str = "random",
        holdout_drug: Optional[str] = None,
        holdout_tissue: Optional[str] = None,
        drug_prop: Optional[float] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        log.info(f"Splitting data... Mode: {mode}")

        # ---------------------------------------------------------
        # Method 1: Random Split (Transductive)
        # ---------------------------------------------------------
        if mode == "random":
            train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)
            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

        # ---------------------------------------------------------
        # Method 2: Cold Drug Split (Inductive Chemistry)
        # ---------------------------------------------------------
        elif mode == "cold_drug":
            if holdout_drug:
                # Specific Drug Case
                log.info(f"   Holding out specific drug: {holdout_drug}")
                test_df = df[df["drug_id"] == str(holdout_drug)]
                train_val_df = df[df["drug_id"] != str(holdout_drug)]
            else:
                # Random Proportion Case
                prop = drug_prop if drug_prop else test_split
                log.info(f"   Holding out {prop:.1%} of drugs randomly.")
                all_drugs = df["drug_id"].unique()
                train_drugs, test_drugs = train_test_split(all_drugs, test_size=prop, random_state=seed)

                train_val_df = df[df["drug_id"].isin(train_drugs)]
                test_df = df[df["drug_id"].isin(test_drugs)]

            # Validation on SEEN drugs (standard practice for tuning)
            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

        # ---------------------------------------------------------
        # Method 3: Cold Cell Split (Inductive Biology)
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
                log.warning("Stratification failed (rare classes?). Fallback to random.")
                return self.split_data(df, mode="random", seed=seed)

        # ---------------------------------------------------------
        # Method 4: Double Cold Split (Zero-Shot)
        # ---------------------------------------------------------
        elif mode == "double_cold":
            # 1. Define Cold Drugs
            d_prop = drug_prop if drug_prop else 0.1
            all_drugs = df["drug_id"].unique()
            train_drugs, test_drugs = train_test_split(all_drugs, test_size=d_prop, random_state=seed)

            # 2. Define Cold Cells
            all_cells = df["cell_id"].unique()
            train_cells, test_cells = train_test_split(all_cells, test_size=0.1, random_state=seed)

            log.info(f"   Double Cold: {len(test_drugs)} drugs x {len(test_cells)} cells in Test Block.")

            # 3. Create Quadrants
            # Train = Seen Drugs AND Seen Cells
            train_mask = df["drug_id"].isin(train_drugs) & df["cell_id"].isin(train_cells)
            train_val_df = df[train_mask]

            # Test = Unseen Drugs AND Unseen Cells
            test_mask = df["drug_id"].isin(test_drugs) & df["cell_id"].isin(test_cells)
            test_df = df[test_mask]

            if len(test_df) == 0:
                log.warning("Double Cold split result is empty! Increase drug_prop.")

            train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

        # ---------------------------------------------------------
        # Method 5: Strict LOTO (OOD Biology)
        # ---------------------------------------------------------
        elif mode == "loto":
            if not holdout_tissue:
                raise ValueError("Config Error: 'loto' mode requires 'test_tissue' to be set.")

            # 1. Identify Lineage Group to prevent leakage
            target_group = None
            exclude_labels = [holdout_tissue]
            for group_name, labels in TISSUE_GROUPS.items():
                if holdout_tissue in labels:
                    target_group = group_name
                    exclude_labels = labels
                    break

            # 2. Define "Ambiguous" labels that are unsafe for LOTO training
            # We assume 'Other' is safe (rare specific cancers), but 'UNCLASSIFIED' is risky.
            unsafe_labels = ["UNCLASSIFIED", "Unknown", "NA", "nan"]

            log.info(f"   LOTO: Testing on '{holdout_tissue}'.")
            if target_group:
                log.info(f"   Strict LOTO: Excluding entire '{target_group}' lineage: {exclude_labels}")
            log.info(f"   Hygiene: Excluding unsafe labels from train: {unsafe_labels}")

            # 3. Test Set (Strictly the target)
            test_df = df[df["cancer_type"] == holdout_tissue]

            # 4. Train Set (Everything else MINUS Lineage MINUS Unsafe)
            train_mask = (~df["cancer_type"].isin(exclude_labels)) & (~df["cancer_type"].isin(unsafe_labels))
            train_val_df = df[train_mask]

            if len(test_df) == 0:
                available = df["cancer_type"].unique()
                raise ValueError(f"Tissue '{holdout_tissue}' not found. Available: {available[:5]}...")

            # Validation from Train distribution (Seen Tissues)
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
    holdout_drug: Optional[str] = None,
    holdout_tissue: Optional[str] = None,
    drug_prop: Optional[float] = None,
    batch_size: int = 32,
):
    # 1. Init Manager
    manager = DataManager(cell_path, drug_path, matrix_path, metadata_path)

    # 2. Get Indices
    df = manager.get_aligned_indices()

    # 3. Perform Split
    train_df, val_df, test_df = manager.split_data(
        df, mode=split_mode, holdout_drug=holdout_drug, holdout_tissue=holdout_tissue, drug_prop=drug_prop
    )

    log.info(f"Final Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 4. Convert to list
    def to_list(d):
        return list(zip(d["c_idx"], d["d_idx"], d["ic50"], strict=True))

    # 5. Build Datasets
    train_ds = PharmacogenomicsDataset(to_list(train_df), manager.cell_tensor, manager.drug_tensor)
    val_ds = PharmacogenomicsDataset(to_list(val_df), manager.cell_tensor, manager.drug_tensor)
    test_ds = PharmacogenomicsDataset(to_list(test_df), manager.cell_tensor, manager.drug_tensor)

    # 6. Build Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    dims = {"cell_dim": manager.cell_tensor.shape[1], "drug_dim": manager.drug_tensor.shape[1]}

    return train_loader, val_loader, test_loader, dims
