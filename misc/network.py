import torch.nn.functional as F
from torch import nn

from gridencoder import GridEncoder


class Network(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_dim: int = 16):
        super().__init__()

        self.__num_layers: int = num_layers
        self.__hidden_dim: int = hidden_dim

        self.encoder: GridEncoder = GridEncoder(input_dim=3, num_levels=8, level_dim=2, base_resolution=8,
                                                log2_hashmap_size=18, desired_resolution=32, gridtype='hash')
        self._in_dim = self.encoder.output_dim

        self.backbone = nn.ModuleList([
            nn.Linear(
                self._in_dim if layer_idx == 0 else self.__hidden_dim,
                1 if layer_idx == num_layers - 1 else self.__hidden_dim,
                bias=False
            )
            for layer_idx in range(num_layers)
        ])

    def forward(self, x):
        h = self.encoder(x)
        for layer_idx, layer in enumerate(self.backbone):
            h = layer(h)
            if layer_idx != self.__num_layers - 1:
                h = F.relu(h, inplace=True)

        return h
