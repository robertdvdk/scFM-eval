import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import hickle as hkl
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


def get_available_drugs(drug_info_file: Path) -> List[str]:
    """
    Get list of all available drug IDs.

    Args:
        drug_info_file: Path to drug metadata CSV

    Returns:
        List of drug ID strings
    """
    drug_info = pd.read_csv(drug_info_file)
    drug_info = drug_info[drug_info["PubCHEM"].apply(lambda x: str(x).isdigit())]
    return drug_info["PubCHEM"].astype(str).tolist()


def generate_drug_splits(
    drug_info_file: Path,
    drug_proportion: float | None,
    test_drug: str | None,
    val_drug: str | None,
    seed: int = 42,
) -> List[Tuple[str | None, str | None]]:
    """
    Generate list of (test_drug, val_drug) pairs for evaluation.

    Args:
        drug_info_file: Path to drug metadata CSV
        drug_proportion: Proportion of drugs to sample (0-1), or None for specific mode
        test_drug: Specific test drug ID (used if drug_proportion is None)
        val_drug: Specific val drug ID (used if drug_proportion is None)
        seed: Random seed for reproducibility

    Returns:
        List of (test_drug, val_drug) tuples
    """
    if drug_proportion is None:
        # Specific mode: return single pair
        if test_drug is None:
            return [(None, None)]
        return [(test_drug, val_drug)]

    # Proportion mode: sample drugs
    all_drugs = get_available_drugs(drug_info_file)

    rng = random.Random(seed)
    n_test = max(1, int(len(all_drugs) * drug_proportion))
    test_drugs = rng.sample(all_drugs, n_test)

    splits = []
    for t_drug in test_drugs:
        available_val_drugs = [d for d in all_drugs if d != t_drug]
        v_drug = rng.choice(available_val_drugs)
        splits.append((t_drug, v_drug))

    log.info(f"Generated {len(splits)} drug splits with proportion {drug_proportion}")
    log.info(f"Val, test drug pairs: {splits}")

    return splits


def load_drug_mapping(drug_info_file: Path) -> Dict[str, str]:
    """Map drug IDs to PubChem IDs."""
    with open(drug_info_file) as f:
        reader = csv.reader(f)
        return {row[0]: row[5] for row in reader if row[5].isdigit()}


def load_cellline_mapping(cell_line_info_file: Path) -> Dict[str, str]:
    """Map cell line IDs to cancer types."""
    mapping = {}
    with open(cell_line_info_file) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            mapping[parts[1]] = parts[-1]  # cellline_id -> TCGA_label
    return mapping


def load_drug_features(drug_feature_dir: Path) -> Dict[str, Tuple[np.ndarray, list, list]]:
    """Load drug graph features from hickle files."""
    features = {}
    pubchem_ids = []

    for file_path in drug_feature_dir.iterdir():
        if file_path.suffix != ".hkl":
            continue
        pubchem_id = file_path.stem
        pubchem_ids.append(pubchem_id)
        features[pubchem_id] = hkl.load(file_path)

    log.info(f"Loaded {len(features)} drug features")
    assert len(pubchem_ids) == len(features), "Mismatch in drug feature counts"
    return features


def build_data_index(
    experiment_df: pd.DataFrame,
    gexpr_df: pd.DataFrame,
    drugid2pubchemid: Dict[str, str],
    cellline2cancertype: Dict[str, str],
    drug_pubchem_ids: set,
) -> List[Tuple[str, str, float, str]]:
    """Build index of valid (cell_line, drug, IC50, cancer_type) tuples."""
    data_idx = []

    for drug_key in experiment_df.index:
        drug_id = drug_key.split(":")[-1]
        if drug_id not in drugid2pubchemid:
            continue

        pubchem_id = drugid2pubchemid[drug_id]
        if pubchem_id not in drug_pubchem_ids:
            continue

        for cell_line in experiment_df.columns:
            if cell_line not in gexpr_df.index or cell_line not in cellline2cancertype:
                continue

            ic50 = experiment_df.loc[drug_key, cell_line]
            if not np.isnan(ic50):
                data_idx.append(
                    (
                        cell_line,
                        pubchem_id,
                        float(ic50),
                        cellline2cancertype[cell_line],
                    )
                )

    nb_celllines = len({item[0] for item in data_idx})
    nb_drugs = len({item[1] for item in data_idx})
    log.info(f"Generated {len(data_idx)} instances across {nb_celllines} cell lines and {nb_drugs} drugs")

    return data_idx


def create_graph_data(feat_mat: np.ndarray, adj_list: list) -> Data:
    """Convert adjacency list to PyTorch Geometric Data object."""
    edge_index = []
    for node, neighbors in enumerate(adj_list):
        edge_index.extend([[node, neighbor] for neighbor in neighbors])

    edge_index = (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index
        else torch.empty((2, 0), dtype=torch.long)
    )

    return Data(x=torch.tensor(feat_mat, dtype=torch.float32), edge_index=edge_index)


def extract_features(
    data_idx: List[Tuple[str, str, float, str]],
    drug_features: Dict[str, Tuple],
    gexpr_df: pd.DataFrame,
) -> Tuple[List[Data], torch.Tensor, torch.Tensor, List[Tuple[str, str, str]]]:
    """Extract drug graphs, gene expression, targets, and metadata."""
    n_samples = len(data_idx)
    n_genes = gexpr_df.shape[1]

    drug_graphs = []
    gexpr_data = torch.zeros((n_samples, n_genes), dtype=torch.float32)
    targets = torch.zeros(n_samples, dtype=torch.float32)
    metadata = []

    for idx, (cell_line, pubchem_id, ic50, cancer_type) in enumerate(tqdm(data_idx, desc="Extracting features")):
        feat_mat, adj_list, _ = drug_features[pubchem_id]
        drug_graphs.append(create_graph_data(feat_mat, adj_list))
        gexpr_data[idx] = torch.tensor(gexpr_df.loc[cell_line].values, dtype=torch.float32)
        targets[idx] = ic50
        metadata.append((cancer_type, cell_line, pubchem_id))

    return drug_graphs, gexpr_data, targets, metadata


def split_by_cancer_type(
    data_idx: List[Tuple],
    cancer_types: List[str],
    train_ratio: float = 0.95,
    seed: int | None = None,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Split data maintaining cancer type distribution."""
    if seed is not None:
        random.seed(seed)

    train_idx, test_idx = [], []
    for cancer_type in cancer_types:
        subtype_data = [item for item in data_idx if item[3] == cancer_type]
        n_train = int(train_ratio * len(subtype_data))
        train_sample = random.sample(subtype_data, n_train)
        test_sample = [item for item in subtype_data if item not in train_sample]
        train_idx.extend(train_sample)
        test_idx.extend(test_sample)

    return train_idx, test_idx


def split_by_drug(
    data_idx: List[Tuple],
    held_out_drug: str,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Split data by holding out a specific drug."""
    train_idx = [item for item in data_idx if item[1] != held_out_drug]
    test_idx = [item for item in data_idx if item[1] == held_out_drug]

    if not test_idx:
        raise ValueError(f"Held-out drug {held_out_drug} not found in dataset")

    return train_idx, test_idx


# TCGA cancer type labels
TCGA_LABELS = [
    "MB",
    "UNCLASSIFIED",
    "SKCM",
    "BLCA",
    "CESC",
    "GBM",
    "LUAD",
    "LUSC",
    "SCLC",
    "MESO",
    "NB",
    "MM",
    "PAAD",
    "ESCA",
    "BRCA",
    "HNSC",
    "KIRC",
    "LAML",
    "OV",
    "PRAD",
    "COREAD",
    "LCML",
    "ALL",
    "LGG",
    "THCA",
    "STAD",
    "DLBC",
    "UCEC",
    "LIHC",
    "CLL",
    "ACC",
    "OTHER",
]


def load_data(
    drug_info_file: Path,
    cell_line_info_file: Path,
    drug_feature_dir: Path,
    ground_truth_file: Path,
    submission_file: Path,
) -> Tuple[Dict, pd.DataFrame, List[Tuple]]:
    """Load all data and build index of valid instances."""
    log.info("Loading metadata and features")

    drugid2pubchemid = load_drug_mapping(drug_info_file)
    cellline2cancertype = load_cellline_mapping(cell_line_info_file)
    drug_features = load_drug_features(drug_feature_dir)
    gexpr_df = pd.read_csv(submission_file, index_col=0)
    experiment_df = pd.read_csv(ground_truth_file, index_col=0)

    # Filter to drugs with available PubChem mappings
    valid_drugs = experiment_df.index[experiment_df.index.str.split(":").str[-1].isin(drugid2pubchemid.keys())]
    experiment_df = experiment_df.loc[valid_drugs]

    data_idx = build_data_index(
        experiment_df,
        gexpr_df,
        drugid2pubchemid,
        cellline2cancertype,
        set(drug_features.keys()),
    )

    return drug_features, gexpr_df, data_idx


def split_data(
    data_idx: List[Tuple],
    val_drug: str | None = None,
    test_drug: str | None = None,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """Split data into train/val/test sets."""
    if test_drug is None and val_drug is None:
        # Cell line split
        log.info("Performing cell line split (90/5/5)")
        train_idx, test_idx = split_by_cancer_type(data_idx, TCGA_LABELS, train_ratio=0.95)
        train_idx, val_idx = split_by_cancer_type(train_idx, TCGA_LABELS, train_ratio=1 - 0.05 / 0.95)
    else:
        # Drug split
        log.info(f"Performing drug split (test={test_drug}, val={val_drug})")
        train_idx, test_idx = split_by_drug(data_idx, test_drug)
        train_idx, val_idx = split_by_drug(train_idx, val_drug)

    log.info(f"Split sizes - train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


def create_dataloaders(
    drug_info_file: Path,
    cell_line_info_file: Path,
    drug_feature_dir: Path,
    ground_truth_file: Path,
    submission_file: Path,
    val_drug: str | None = None,
    test_drug: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[Tuple], int]:
    """Create train/val/test dataloaders."""
    # Load and split data
    drug_features, gexpr_df, data_idx = load_data(
        drug_info_file, cell_line_info_file, drug_feature_dir, ground_truth_file, submission_file
    )
    train_idx, val_idx, test_idx = split_data(data_idx, val_drug, test_drug)

    # Extract features for each split
    log.info("Extracting training features")
    X_drug_train, X_gexpr_train, y_train, _ = extract_features(train_idx, drug_features, gexpr_df)

    log.info("Extracting validation features")
    X_drug_val, X_gexpr_val, y_val, _ = extract_features(val_idx, drug_features, gexpr_df)

    log.info("Extracting test features")
    X_drug_test, X_gexpr_test, y_test, test_metadata = extract_features(test_idx, drug_features, gexpr_df)

    # Create dataloaders
    train_loader = DataLoader(
        list(zip(X_drug_train, X_gexpr_train, y_train, strict=True)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        list(zip(X_drug_val, X_gexpr_val, y_val, strict=True)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        list(zip(X_drug_test, X_gexpr_test, y_test, strict=True)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, test_metadata, gexpr_df.shape[-1]
