##codigo dos modelos
from torch import Tensor
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: int, n_layers: int, num_classes: int):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims, dtype=torch.float32),
            nn.ReLU(),
        )
        for _ in range(n_layers):
            self.model.append(nn.LazyLinear(hidden_dims, dtype=torch.float32))
            self.model.append(nn.SELU())

        self.model.append(nn.Linear(hidden_dims, num_classes, dtype=torch.float32))
        # print(self.model)
    

    def forward(self, x) -> Tensor:
        return self.model(x)
