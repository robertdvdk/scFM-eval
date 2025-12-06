import argparse
import os
import subprocess as sp
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration Definitions
# ==========================================

CONFIGS = {
    "norman19": {
        "url": "https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1",
        "dir": "./data/gene_perturbation/norman19",
        "filename": "norman19_downloaded.h5ad",
        "output_filename": "norman19_processed.h5ad",
        "max_cells_pert": 256,
        "max_cells_control": 8192,
        "col_map": {
            "nCount_RNA": "ncounts",
            "nFeature_RNA": "ngenes",
            "percent.mt": "percent_mito",
            "cell_line": "cell_type",
        },
    },
    "replogle22": {
        "url": "https://plus.figshare.com/ndownloader/files/35775606",
        "dir": "./data/gene_perturbation/replogle22",
        "filename": "replogle22_downloaded.h5ad",
        "output_filename": "replogle22_processed.h5ad",
        "max_cells_pert": 64,
        "max_cells_control": 8192,
        "col_map": {
            "UMI_count": "ncounts",
            "mitopercent": "percent_mito",
            # Note: Replogle doesn't have a 'cell_type' col in raw data, handled in logic
        },
    },
}

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================


def load_and_standardize(dataset_name):
    """
    Downloads data if needed, loads it, and standardizes metadata columns
    based on the specific dataset configuration.
    """
    cfg = CONFIGS[dataset_name]

    # Ensure directory exists
    if not os.path.exists(cfg["dir"]):
        os.makedirs(cfg["dir"])

    filepath = os.path.join(cfg["dir"], cfg["filename"])

    # Download
    if not os.path.exists(filepath):
        print(f"[{dataset_name}] Downloading data from {cfg['url']}...")
        sp.call(f"wget -q {cfg['url']} -O {filepath}", shell=True)

    print(f"[{dataset_name}] Loading data...")
    adata = sc.read_h5ad(filepath)

    # 1. Rename Columns
    print(f"[{dataset_name}] Mapping metadata columns...")
    adata.obs.rename(columns=cfg["col_map"], inplace=True)

    # 2. Dataset-Specific Logic for Condition/Perturbation
    if dataset_name == "norman19":
        # Norman specific: Clean underscores to plus signs
        adata.obs["perturbation"] = adata.obs["perturbation"].str.replace("_", "+")
        adata.obs["condition"] = adata.obs["perturbation"].astype("category")

    elif dataset_name == "replogle22":
        # Replogle specific: cell_type is RPE1, condition comes from 'gene'
        adata.obs["cell_type"] = "RPE1"
        # Map 'non-targeting' to 'control'
        adata.obs["condition"] = adata.obs["gene"].astype(str)
        adata.obs.loc[adata.obs["condition"] == "non-targeting", "condition"] = "control"
        adata.obs["condition"] = adata.obs["condition"].astype("category")

    # Ensure sparse matrix
    adata.X = csr_matrix(adata.X)

    return adata


def process_data(adata, dataset_name):
    """
    Performs filtering, normalization, QC stats, and downsampling.
    """
    cfg = CONFIGS[dataset_name]

    print("Filtering cells (min_genes=200) and genes (min_cells=3)...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Stash raw counts
    adata.layers["counts"] = adata.X.copy()

    # Calculate Mean and Variance per perturbation (Technical Artifact check)
    print("Calculating mean and dispersion per perturbation...")
    unique_conditions = adata.obs["condition"].unique()

    # Pre-allocate dictionaries
    mean_dict = {}
    disp_dict = {}

    # Note: Vectorized approach is faster, but keeping loop for memory safety on large arrays if needed.
    # Using the loop as per original request logic
    for pert in tqdm(unique_conditions, desc="QC Stats"):
        pert_cells = adata.obs[adata.obs["condition"] == pert].index
        pert_counts = adata[pert_cells].X.toarray()  # dense for numpy math
        mean_dict[pert] = np.mean(pert_counts, axis=0)
        disp_dict[pert] = np.var(pert_counts, axis=0)

    # Save to uns
    adata.uns["mean_dict"] = pd.DataFrame(mean_dict, index=adata.var_names).to_dict(orient="list")
    adata.uns["disp_dict"] = pd.DataFrame(disp_dict, index=adata.var_names).to_dict(orient="list")
    adata.uns["mean_disp_dict_genes"] = adata.var_names.tolist()

    # Normalize
    print("Normalizing and Log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Downsampling
    print(f"Downsampling: Max {cfg['max_cells_pert']} per pert, {cfg['max_cells_control']} for control...")
    np.random.seed(42)  # Fixed seed for reproducibility

    pert_counts = adata.obs["condition"].value_counts()

    cells_to_keep = []

    for pert in adata.obs["condition"].unique():
        pert_cells = adata.obs[adata.obs["condition"] == pert].index.tolist()
        n_cells = len(pert_cells)

        limit = cfg["max_cells_control"] if pert == "control" else cfg["max_cells_pert"]

        if n_cells > limit:
            selected = np.random.choice(pert_cells, size=limit, replace=False)
            cells_to_keep.extend(selected)
        else:
            cells_to_keep.extend(pert_cells)

    # Subset
    adata = adata[cells_to_keep].copy()

    # HVGs
    print("Selecting 8192 Highly Variable Genes...")
    sc.pp.highly_variable_genes(adata, n_top_genes=8192, subset=True)

    return adata


# ==========================================
# 3. Differential Expression & Weighting
# ==========================================


def compute_degs(adata, mode="vsrest", pval_threshold=0.05):
    """
    Computes DEGs using t-test.
    mode: 'vsrest' (One-vs-Rest) or 'vscontrol' (One-vs-Control)
    """
    if mode == "vsrest":
        adata_subset = adata[adata.obs["condition"] != "control"].copy()
        reference = "rest"
    elif mode == "vscontrol":
        adata_subset = adata.copy()
        reference = "control"
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")

    print(f"Computing DEGs ({mode})...")
    sc.tl.rank_genes_groups(adata_subset, "condition", method="t-test_overestim_var", reference=reference)

    # Store results in the main adata.uns temporarily for extraction
    key_name = f"rank_genes_groups_{mode}"
    adata.uns[key_name] = adata_subset.uns["rank_genes_groups"].copy()

    return adata.uns[key_name]


def calculate_weights(adata, mode="vsrest"):
    """
    Calculates weighting matrix for WMSE (Weighted Mean Squared Error).
    Methodology:
    1. Abs(Score) -> 2. MinMax Scale -> 3. Square -> 4. Normalize sum to 1.
    """
    # Ensure DEGs are computed
    if f"rank_genes_groups_{mode}" not in adata.uns:
        compute_degs(adata, mode=mode)

    key = f"rank_genes_groups_{mode}"

    # Extract structured arrays
    scores_df = pd.DataFrame(adata.uns[key]["scores"])
    names_df = pd.DataFrame(adata.uns[key]["names"])

    print(f"Calculating weights for {mode}...")

    # 1. Absolute values
    abs_scores = scores_df.abs()

    # 2. Min-Max Normalization (per perturbation column)
    # Avoid divide by zero if max == min
    denom = abs_scores.max() - abs_scores.min()
    denom[denom == 0] = 1.0
    norm_scores = (abs_scores - abs_scores.min()) / denom

    # 3. Square weights (accentuate strong signals)
    norm_scores = norm_scores**2

    # 4. Normalize to sum to 1 (probability distribution)
    norm_scores = norm_scores / norm_scores.sum(axis=0)

    # 5. Realign to Gene Order (adata.var_names)
    # rank_genes_groups outputs genes sorted by rank, not by original index
    aligned_weights = pd.DataFrame(index=adata.var_names, columns=norm_scores.columns)

    # Iterate through perturbations to map ranks back to gene names
    # Note: Using a dictionary map is often faster than reindexing inside a loop,
    # but maintaining user logic structure for clarity.
    for pert in tqdm(norm_scores.columns, desc=f"Aligning {mode} weights"):
        # Map: GeneName -> Weight
        mapper = dict(zip(names_df[pert].values, norm_scores[pert].values, strict=True))
        # Map to adata.var_names order
        aligned_weights[pert] = adata.var_names.map(mapper).fillna(0)

    return aligned_weights


def save_dual_weights(adata, dataset_name):
    """
    Calculates both vs-rest and vs-control weights and saves them to adata.varm.
    """
    output_path = os.path.join(CONFIGS[dataset_name]["dir"], CONFIGS[dataset_name]["output_filename"])

    # 1. vsRest Weights
    df_vsrest = calculate_weights(adata, mode="vsrest")
    adata.varm["weights_vsrest"] = df_vsrest.values
    adata.uns["weights_vsrest_columns"] = df_vsrest.columns.tolist()

    # 2. vsControl Weights
    df_vscontrol = calculate_weights(adata, mode="vscontrol")
    adata.varm["weights_vscontrol"] = df_vscontrol.values
    adata.uns["weights_vscontrol_columns"] = df_vscontrol.columns.tolist()

    # Cleanup: Remove heavy uns dictionaries before saving if desired
    # (Optional: adata.uns.pop('rank_genes_groups_vsrest', None))

    # [FIX] Remove heavy unstructured data that causes HDF5 header overflow
    # The weights have likely already been extracted to adata.layers or logical masks
    keys_to_remove = ["rank_genes_groups_vsrest", "rank_genes_groups_vscontrol"]
    for key in keys_to_remove:
        if key in adata.uns:
            print(f"Removing {key} from uns to prevent HDF5 header overflow...")
            del adata.uns[key]

    print(f"Saving final AnnData to {output_path}...")
    adata.write_h5ad(output_path)

    # Save gene list separately as CSV (useful for foundation model tokenization checks)
    gene_path = os.path.join(CONFIGS[dataset_name]["dir"], f"{dataset_name}_genes.csv")
    pd.Series(adata.var_names).to_csv(gene_path, index=False, header=["gene_name"])

    print("Processing Complete.")


# ==========================================
# 4. Main Execution
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="Process Single Cell Perturbation Data")
    parser.add_argument("dataset", choices=["norman19", "replogle22"], help="The dataset to process.")
    args = parser.parse_args()

    # 1. Load
    adata = load_and_standardize(args.dataset)

    # 2. Process (Filter -> QC -> Norm -> Downsample -> HVG)
    adata = process_data(adata, args.dataset)

    # 3. Calculate Weights and Save
    save_dual_weights(adata, args.dataset)


if __name__ == "__main__":
    main()
