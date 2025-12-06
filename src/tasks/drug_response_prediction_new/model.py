import torch
import torch.nn as nn


class DualStreamModel(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int, latent_dim: int = 256):
        super().__init__()

        # Stream 1: Cell (CancerGPT)
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        # Stream 2: Drug (Embeddings)
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
