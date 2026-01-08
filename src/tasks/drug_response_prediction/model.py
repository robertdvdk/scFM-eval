from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, global_mean_pool


class UGCNN(nn.Module):
    def __init__(self, hidden_dims: List[int], input_dim: int, out_channels: int):
        super(UGCNN, self).__init__()
        dims = [input_dim] + hidden_dims + [out_channels]
        self.conv_list = nn.ModuleList([])
        for i in range(len(dims) - 1):
            self.conv_list.extend([GCNConv(dims[i], dims[i + 1])])
        self.norm_list = nn.ModuleList([])
        for i in range(len(dims) - 1):
            self.norm_list.append(nn.Sequential(BatchNorm(dims[i + 1]), nn.ReLU(dims[i + 1]), nn.Dropout(p=0.1)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.conv_list, self.norm_list, strict=True):
            x = norm(conv(x, edge_index))

        # Use global mean pooling for the entire graph
        x = global_mean_pool(x, batch)

        return x


class DualStreamModel(nn.Module):
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
        self.head = nn.Sequential(nn.Linear(latent_dim * 2, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, cell, drug):
        c_emb = self.cell_encoder(cell)
        d_emb = self.drug_encoder(drug)

        # Late Fusion
        combined = torch.cat([c_emb, d_emb], dim=1)
        return self.head(combined)


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
