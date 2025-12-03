import csv
import os
import random

import hickle as hkl
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def MetadataGenerate(
    Drug_info_file,
    Cell_line_info_file,
    Drug_feature_file,
    Gene_expression_file,
    Cancer_response_exp_file,
):
    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, "r"))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    # map cellline --> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split("\t")[1]
        TCGA_label = line.strip().split("\t")[-1]
        cellline2cancertype[cellline_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split(".")[0])
        feat_mat, adj_list, degree_list = hkl.load("%s/%s" % (Drug_feature_file, each))
        drug_feature[each.split(".")[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    # load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=",", header=0, index_col=[0])

    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=",", header=0, index_col=[0])
    # filter experiment data
    drug_match_list = [item for item in experiment_data.index if item.split(":")[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(":")[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature.index:
                if (
                    not np.isnan(experiment_data_filtered.loc[each_drug, each_cellline])
                    and each_cellline in cellline2cancertype.keys()
                ):
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append(
                        (
                            each_cellline,
                            pubchem_id,
                            ln_IC50,
                            cellline2cancertype[each_cellline],
                        )
                    )
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print("%d instances across %d cell lines and %d drugs were generated." % (len(data_idx), nb_celllines, nb_drugs))
    return drug_feature, gexpr_feature, data_idx


def DataSplit(data_idx, TCGA_label_set, ratio=0.95):
    data_train_idx, data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        train_list = random.sample(data_subtype_idx, int(ratio * len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx, data_test_idx


def DrugSplit(data_idx, drugtype):
    data_train_idx, data_test_idx = [], []
    data_test_idx = [item for item in data_idx if item[1] == drugtype]
    data_train_idx = [item for item in data_idx if item[1] != drugtype]
    return data_train_idx, data_test_idx


def CalculateGraphFeat(feat_mat, adj_list):
    edge_index = []

    # Convert adjacency list to edge_index format
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edge_index.append([node, neighbor])

    # --- FIX STARTS HERE ---
    if len(edge_index) == 0:
        # Explicitly create an empty 2D tensor [2, 0] if there are no edges
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
    else:
        # Normal case: transpose to [2, N]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create a data object
    data = Data(x=torch.tensor(feat_mat).float(), edge_index=edge_index)

    return data


def FeatureExtract(data_idx, drug_feature, gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    gexpr_data = torch.zeros((nb_instance, nb_gexpr_features)).float()
    target = torch.zeros(nb_instance).float()
    for idx in tqdm(range(nb_instance)):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]

        feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]

        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list)

        gexpr_data[idx, :] = torch.tensor(gexpr_feature.loc[cell_line_id].values)
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
    return drug_data, gexpr_data, target, cancer_type_list
