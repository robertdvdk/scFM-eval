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
import subprocess as sp

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
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
    np.random.seed(0)
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


def calculate_weights(adata, mode="vsrest"):
    # 1. Compute DEGs
    compute_degs(adata, mode=mode, pval_threshold=0.05)

    # 2. Retrieve the results dynamically based on mode
    key = f"rank_genes_groups_{mode}"

    # scanpy returns 'names' and 'scores' as structured arrays
    # columns are groups (perturbations), rows are ranks (0 to n_genes)
    scores_df = pd.DataFrame(adata.uns[key]["scores"])
    names_df = pd.DataFrame(adata.uns[key]["names"])

    # 3. Apply transformations (Abs -> MinMax -> Square)
    abs_scores = scores_df.abs()

    # Min-max normalization per perturbation (axis=0 is default for min/max)
    # This scales everything to [0, 1]
    denom = abs_scores.max() - abs_scores.min()
    # Handle cases where max == min (e.g., all zeros) to avoid divide-by-zero
    denom[denom == 0] = 1.0

    norm_scores = (abs_scores - abs_scores.min()) / denom

    # Square the weights to accentuate differences [cite: 163]
    norm_scores = norm_scores**2

    # 4. Normalize to sum to 1 per perturbation [cite: 163]
    # This turns them into a probability distribution
    norm_scores = norm_scores / norm_scores.sum(axis=0)

    # 5. REINDEXING (Crucial Step)
    # We need to map these scores back to the actual gene order in adata.var_names
    # We iterate over columns because each column has a different gene order in 'names_df'
    aligned_weights = pd.DataFrame(index=adata.var_names, columns=norm_scores.columns)

    print("Aligning weights to gene list...")
    for pert in tqdm(norm_scores.columns):
        # Create a series: index=GeneName, value=Weight
        pert_weights = pd.Series(norm_scores[pert].values, index=names_df[pert].values)
        # Reindex to match the exact order of adata.var_names
        aligned_weights[pert] = pert_weights.reindex(adata.var_names).fillna(0)

    # 6. Return the aligned weights dataframe
    # Rows = Genes (aligned with adata.X), Columns = Perturbations
    return aligned_weights


# ============================================================================
# SECTION 4: WEIGHTED MSE AND WEIGHTED DELTA R² METRICS
# ============================================================================


def mse(x1, x2):
    """
    Calculate Mean Squared Error (MSE).

    Args:
        x1: First array (e.g., prediction)
        x2: Second array (e.g., ground truth)

    Returns:
        float: MSE value
    """
    return np.mean((x1 - x2) ** 2)


def wmse(x1, x2, weights):
    """
    Calculate Weighted Mean Squared Error (WMSE).

    The weights are normalized to sum to 1, so the WMSE represents
    a weighted average of squared errors across genes.

    Args:
        x1: First array (e.g., prediction)
        x2: Second array (e.g., ground truth)
        weights: Array of weights for each element

    Returns:
        float: WMSE value
    """
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)

    # Normalize weights to sum to 1
    normalized_weights = weights_arr / np.sum(weights_arr)

    # Calculate weighted MSE
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))


def r2_score_on_deltas(delta_true, delta_pred, weights=None):
    """
    Calculate R² score on delta values (change from control/baseline).

    If weights are provided, this calculates the weighted R² score.

    Args:
        delta_true: True delta values (ground truth change from baseline)
        delta_pred: Predicted delta values (predicted change from baseline)
        weights: Optional weights for each gene (for weighted R²)

    Returns:
        float: R² score (or weighted R² if weights provided)
    """
    if len(delta_true) < 2 or len(delta_pred) < 2 or delta_true.shape != delta_pred.shape:
        return np.nan

    if weights is not None:
        # Weighted R² score
        return r2_score(delta_true, delta_pred, sample_weight=weights)
    else:
        # Standard R² score
        return r2_score(delta_true, delta_pred)


# ============================================================================
# DEMO USAGE
# ============================================================================


def demo():
    """
    Demonstrate the calculation of weighted MSE and weighted delta R².
    """
    print("=" * 80)
    print("Weighted MSE and Weighted Delta R² Demo")
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

    # Print summary of DEGs
    print("\nDEG Summary:")
    for pert in list(adata.uns["deg_dict_vsrest"].keys())[:5]:  # Show first 5
        deg_info = adata.uns["deg_dict_vsrest"][pert]
        n_up = len(deg_info["up"])
        n_down = len(deg_info["down"])
        print(f"  {pert}: {n_up} up-regulated, {n_down} down-regulated")
    print(f"  ... and {len(adata.uns['deg_dict_vsrest']) - 5} more perturbations")
    print()

    # Step 3: Calculate weights
    print("Step 3: Calculating weights for weighted metrics...")
    print("-" * 80)
    pert_weights = calculate_weights(adata, mode="vsrest")

    print(f"Calculated weights for {len(pert_weights)} perturbations")

    # Show weight statistics for a few perturbations
    print("\nWeight Statistics (first 3 perturbations):")
    for pert, weights in enumerate(list(pert_weights.items())[:3]):
        print(f"  {pert}:")
        print(f"    Min: {weights.min():.6f}")
        print(f"    Max: {weights.max():.6f}")
        print(f"    Mean: {weights.mean():.6f}")
        print(f"    # Non-zero weights: {(weights > 0).sum()}")
    print()

    # Step 4: Calculate example metrics
    print("Step 4: Calculating weighted MSE and weighted delta R² (example)...")
    print("-" * 80)

    # Get a random perturbation for demonstration
    example_pert = list(pert_weights.keys())[0]

    # Get perturbation mean expression
    pert_cells = adata.obs["condition"] == example_pert
    pert_mean = adata[pert_cells].X.mean(axis=0).A1

    # Get control mean expression
    control_cells = adata.obs["condition"] == "control"
    control_mean = adata[control_cells].X.mean(axis=0).A1

    # Get overall mean (all perturbations)
    all_pert_means = []
    for p in adata.obs["condition"].unique():
        if p != "control":
            p_cells = adata.obs["condition"] == p
            p_mean = adata[p_cells].X.mean(axis=0).A1
            all_pert_means.append(p_mean)
    total_mean = np.mean(all_pert_means, axis=0)

    # Calculate metrics comparing perturbation to control
    print(f"\nExample perturbation: {example_pert}")
    print(f"  Number of cells: {pert_cells.sum()}")

    # Standard MSE
    mse_value = mse(pert_mean, control_mean)
    print(f"  Standard MSE (vs control): {mse_value:.6f}")

    # Weighted MSE
    weights = pert_weights[example_pert]
    wmse_value = wmse(pert_mean, control_mean, weights)
    print(f"  Weighted MSE (vs control): {wmse_value:.6f}")

    # Delta calculations (change from total mean)
    delta_pert = pert_mean - total_mean
    delta_control = control_mean - total_mean

    # Standard R² on deltas
    r2_value = r2_score_on_deltas(delta_pert, delta_control)
    print(f"  Standard R² (on deltas): {r2_value:.6f}")

    # Weighted R² on deltas
    wr2_value = r2_score_on_deltas(delta_pert, delta_control, weights)
    print(f"  Weighted R² (on deltas): {wr2_value:.6f}")

    # TEST: Compare against Mean Baseline (Mode Collapse Simulation)
    print("\n  --- Mode Collapse Test (Prediction = Dataset Mean) ---")

    # Standard metrics often incorrectly reward this
    mse_mean_base = mse(pert_mean, total_mean)
    print(f"  Standard MSE (vs mean baseline): {mse_mean_base:.6f}")

    # Weighted metrics should punish this
    wmse_mean_base = wmse(pert_mean, total_mean, weights)
    print(f"  Weighted MSE (vs mean baseline): {wmse_mean_base:.6f}")

    # Calculate Deltas for R2
    # If prediction is total_mean, then predicted_delta is 0 (total_mean - total_mean)
    delta_pred_mean_base = np.zeros_like(delta_pert)

    # Paper Appendix B proves this is always negative for Weighted R2
    wr2_mean_base = r2_score_on_deltas(delta_pert, delta_pred_mean_base, weights)
    print(f"  Weighted R² (vs mean baseline):  {wr2_mean_base:.6f}")

    print()
    print("=" * 80)
    print("Demo completed!")
    print("=" * 80)

    return {
        "adata": adata,
        "pert_weights": pert_weights,
        "example_pert": example_pert,
        "metrics": {
            "mse": mse_value,
            "wmse": wmse_value,
            "r2": r2_value,
            "wr2": wr2_value,
        },
    }


if __name__ == "__main__":
    results = demo()
