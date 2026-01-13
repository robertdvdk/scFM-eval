from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool


class UGCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(UGCNN, self).__init__()
        dims = [input_dim] + hidden_dims
        self.conv_list = nn.ModuleList([])

        for i in range(len(dims) - 1):
            self.conv_list.append(GCNConv(dims[i], dims[i + 1]))

        self.norm_list = nn.ModuleList([])
        for i in range(len(dims) - 1):
            self.norm_list.append(nn.Sequential(BatchNorm(dims[i + 1]), nn.ReLU(), nn.Dropout(p=0.1)))

        # Project final GCN embedding to match latent_dim of the fusion head
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.conv_list, self.norm_list, strict=True):
            x = conv(x, edge_index)
            x = norm(x)

        # Global pooling: (Num_Nodes, Feat) -> (Batch_Size, Feat)
        x = global_mean_pool(x, batch)

        return self.output_proj(x)


class DualStreamModel(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int, latent_dim: int = 256, is_graph: bool = False):
        super().__init__()

        # Stream 1: Cell Embeddings (MLP)
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        # Stream 2: Drug Embeddings (MLP vs GCN)
        self.is_graph = is_graph
        if self.is_graph:
            # GCN: input=drug_dim (feat size), hidden=[256, 256], out=latent_dim
            self.drug_encoder = UGCNN(input_dim=drug_dim, hidden_dims=[256, 256], output_dim=latent_dim)
        else:
            # MLP
            self.drug_encoder = nn.Sequential(
                nn.Linear(drug_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, latent_dim),
                nn.ReLU(),
            )

        # Fusion Head
        self.head = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, cell, drug_input):
        """
        cell: Tensor (Batch, Cell_Dim)
        drug_input: Tensor (Batch, Drug_Dim) OR PyG Batch Object
        """
        c_emb = self.cell_encoder(cell)
        d_emb = self.drug_encoder(drug_input)

        # Late Fusion
        combined = torch.cat([c_emb, d_emb], dim=1)
        return self.head(combined)


# Used as baseline: what happens when we don't use the cell at all, and predict
# IC50 based on only the drug
class DrugMLP(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int, latent_dim: int = 256):
        super().__init__()
        # Stream 2: Process drug embeddings
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        # Fusion Head
        self.head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, cell, drug):
        d_emb = self.drug_encoder(drug)

        # Late Fusion
        return self.head(d_emb)


# Used as baseline: what happens when we don't use the drugs at all, and predict
# IC50 based on only the cell
class CellMLP(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int, latent_dim: int = 256):
        super().__init__()

        # Stream 1: Process cell embeddings
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        # Fusion Head
        self.head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, cell, drug):
        c_emb = self.cell_encoder(cell)

        # Late Fusion
        return self.head(c_emb)
