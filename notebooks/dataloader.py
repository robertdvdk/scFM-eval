"""
Weighted MSE and Weighted Delta R² Implementation
==================================================

This script demonstrates the weighted MSE and weighted delta R² metrics
used to prioritize perturbation-specific DEGs as described in the paper
"Diversity by Design: Addressing Mode Collapse Improves scRNA-seq
Perturbation Modeling on Well-Calibrated Metrics"

The script:
1. Loads (or downloads) the norman19 dataset
2. Determines perturbation-specific DEGs for every perturbation
3. Calculates weights based on DEG scores
4. Uses these weights to calculate weighted MSE and weighted delta R²
"""

import os
import pickle
import subprocess as sp

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# SECTION 1: DATA LOADING / DOWNLOAD
# ============================================================================


def load_or_download_norman19(data_cache_dir="./data/norman19"):
    """
    Load or download the Norman19 dataset.

    Args:
        data_cache_dir: Directory to store/load the dataset

    Returns:
        adata: Processed AnnData object with norman19 data
    """
    data_url = "https://zenodo.org/records/7041849/files/NormanWeissman2019_filtered.h5ad?download=1"

    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    tmp_data_dir = f"{data_cache_dir}/norman19_downloaded.h5ad"
    output_data_path = f"{data_cache_dir}/norman19_processed.h5ad"

    # Check if processed data already exists
    # if os.path.exists(output_data_path):
    #     print(f"Loading existing processed data from {output_data_path}")
    #     adata = sc.read_h5ad(output_data_path)
    #     return adata

    # Download if raw data doesn't exist
    if not os.path.exists(tmp_data_dir):
        print(f"Downloading data from {data_url}")
        sp.call(f"wget -q {data_url} -O {tmp_data_dir}", shell=True)

    print(f"Loading data from {tmp_data_dir}")
    adata = sc.read_h5ad(tmp_data_dir)

    # Preprocess the data
    print("Preprocessing data...")

    # Rename columns
    adata.obs.rename(
        columns={
            "nCount_RNA": "ncounts",
            "nFeature_RNA": "ngenes",
            "percent.mt": "percent_mito",
            "cell_line": "cell_type",
        },
        inplace=True,
    )
    adata.obs["perturbation"] = adata.obs["perturbation"].str.replace("_", "+")
    adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")
    adata.obs["condition"] = adata.obs.perturbation.copy()
    adata.X = csr_matrix(adata.X)

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Stash raw counts
    adata.layers["counts"] = adata.X.copy()

    # Do library size normalization and log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Downsample each perturbation to have no more than N cells
    MAX_CELLS = 256
    MAX_CELLS_CONTROL = 8192

    pert_counts = adata.obs["condition"].value_counts()
    pert_counts = pert_counts[pert_counts > MAX_CELLS]
    cells_to_keep = []
    for pert in pert_counts.index:
        pert_cells = adata.obs[adata.obs["condition"] == pert].index.tolist()
        if pert == "control":
            pert_cells = np.random.choice(pert_cells, size=MAX_CELLS_CONTROL, replace=False)
        else:
            pert_cells = np.random.choice(pert_cells, size=MAX_CELLS, replace=False)
        cells_to_keep.extend(pert_cells)

    # Subset the adata object
    adata = adata[cells_to_keep]

    # Get 8192 HVGs -- subset the adata object to only include the HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=8192, subset=True)

    # Save the processed data
    print(f"Saving processed data to {output_data_path}")
    adata.write_h5ad(output_data_path)

    return adata


# ============================================================================
# SECTION 2: PERTURBATION-SPECIFIC DEG CALCULATION
# ============================================================================


def compute_degs(adata, mode="vsrest", pval_threshold=0.05):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.

    This function identifies genes that are significantly differentially expressed
    for each perturbation compared to either:
    - 'vsrest': all other perturbations (excluding control)
    - 'vscontrol': the control condition only

    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)

    Returns:
        dict: rank_genes_groups results dictionary containing:
            - 'names': Gene names ranked by significance
            - 'scores': Statistical test scores for each gene
            - 'pvals_adj': Adjusted p-values
            - 'logfoldchanges': Log fold changes

    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == "vsrest":
        # Remove control cells for vsrest analysis
        # This compares each perturbation against all other perturbations
        adata_subset = adata[adata.obs["condition"] != "control"].copy()
        reference = "rest"
    elif mode == "vscontrol":
        # Use full dataset for vscontrol analysis
        # This compares each perturbation against the control
        adata_subset = adata.copy()
        reference = "control"
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")

    # Compute DEGs using t-test with overestimated variance
    print(f"Computing DEGs ({mode})...")
    sc.tl.rank_genes_groups(adata_subset, "condition", method="t-test_overestim_var", reference=reference)

    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])

    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs["condition"].unique(), desc=f"Computing DEGs {mode}"):
        if mode == "vscontrol" and pert == "control":
            continue  # Skip control when comparing vs control

        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]

        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]

        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()

        deg_dict[pert] = {"up": pert_degs_sig_up, "down": pert_degs_sig_down}

    # Save results to adata.uns
    adata.uns[f"deg_dict_{mode}"] = deg_dict
    adata.uns[f"rank_genes_groups_{mode}"] = adata_subset.uns["rank_genes_groups"].copy()

    return adata_subset.uns["rank_genes_groups"]


# ============================================================================
# SECTION 3: WEIGHT CALCULATION FOR WEIGHTED METRICS
# ============================================================================


def calculate_weights_for_perturbation(pert_scores, gene_names, adata_var_names):
    """
    Calculate normalized weights for a single perturbation based on DEG scores.

    The weights are calculated as follows:
    1. Take absolute value of scores
    2. Min-max normalize to [0, 1]
    3. Square the normalized values to emphasize strong DEGs
    4. Reindex to match the order of genes in the dataset

    Args:
        pert_scores: Array of statistical scores for genes (from rank_genes_groups)
        gene_names: Array of gene names corresponding to the scores
        adata_var_names: Index object with all gene names in the dataset

    Returns:
        pd.Series: Weights for each gene, indexed by gene name
    """
    abs_scores = np.abs(pert_scores)  # Take absolute value
    min_val = np.min(abs_scores)
    max_val = np.max(abs_scores)

    # Handle edge cases
    if max_val == min_val:
        if max_val == 0:  # All scores are 0
            normalized_weights = np.zeros_like(abs_scores)
        else:  # All scores are the same non-zero value
            normalized_weights = np.ones_like(abs_scores)
    else:
        # Min-max normalization to [0, 1]
        normalized_weights = (abs_scores - min_val) / (max_val - min_val)

    # Ensure no NaNs in weights, replace with 0 if any
    normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)

    # Square the normalized weights to make weighting stronger
    # This emphasizes genes with higher scores (stronger DEGs)
    stronger_normalized_weights = np.square(normalized_weights)

    stronger_normalized_weights /= stronger_normalized_weights.sum()  # Normalize to sum to 1

    # Create a Series with gene names as index
    weights = pd.Series(stronger_normalized_weights, index=gene_names)

    # Reindex to match the order of genes in the dataset
    weights = weights.reindex(adata_var_names)

    return weights


def calculate_all_weights(adata, mode="vsrest", score_type="scores"):
    """
    Calculate weights for all perturbations based on DEG scores.

    Args:
        adata: AnnData object with DEG results in .uns
        mode: 'vsrest' or 'vscontrol' (must match the mode used in compute_degs)
        score_type: Type of score to use ('scores' or 'logfoldchanges')

    Returns:
        dict: Dictionary mapping perturbation names to weight Series
    """
    # Extract DEG results
    deg_results = adata.uns[f"rank_genes_groups_{mode}"]
    names_df = pd.DataFrame(deg_results["names"])
    scores_df = pd.DataFrame(deg_results[score_type])

    # Calculate weights for each perturbation
    pert_weights = {}
    for pert in tqdm(scores_df.columns, desc="Calculating weights"):
        if pert == "control":  # Skip control
            continue

        pert_scores = scores_df[pert].values
        pert_gene_names = names_df[pert].values

        weights = calculate_weights_for_perturbation(pert_scores, pert_gene_names, adata.var_names)

        pert_weights[pert] = weights

    return pert_weights


def main():
    print("=" * 80)
    print("Calculating DEG weights")
    print("=" * 80)
    print()

    # Step 1: Load or download the norman19 dataset
    print("Step 1: Loading norman19 dataset...")
    print("-" * 80)
    adata = load_or_download_norman19()
    print(f"Dataset loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"Perturbations: {adata.obs['condition'].nunique()}")
    print()

    # Step 2: Compute perturbation-specific DEGs
    print("Step 2: Computing perturbation-specific DEGs...")
    print("-" * 80)
    _ = compute_degs(adata, mode="vsrest", pval_threshold=0.05)
    _ = compute_degs(adata, mode="vscontrol", pval_threshold=0.05)

    # Step 3: Calculate weights
    print("Step 3: Calculating weights for weighted metrics...")
    print("-" * 80)
    vsrest_weights = calculate_all_weights(adata, mode="vsrest", score_type="scores")
    vscontrol_weights = calculate_all_weights(adata, mode="vscontrol", score_type="scores")

    output_dir = "./data/norman19"
    os.makedirs(output_dir, exist_ok=True)

    vsrest_weights_path = f"{output_dir}/norman19_weights_vsrest.pkl"
    vscontrol_weights_path = f"{output_dir}/norman19_weights_vscontrol.pkl"

    with open(vsrest_weights_path, "wb") as f:
        pickle.dump(vsrest_weights, f)
    print(f"Saved vsrest weights to {vsrest_weights_path}")

    with open(vscontrol_weights_path, "wb") as f:
        pickle.dump(vscontrol_weights, f)
    print(f"Saved vscontrol weights to {vscontrol_weights_path}")

    print()
    print("=" * 80)
    print("Weight calculation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
