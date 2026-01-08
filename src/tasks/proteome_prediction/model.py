import torch
import torch.nn as nn


class ProteomePredictionModel(nn.Module):
    def __init__(self, cell_dim: int, protein_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, protein_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
