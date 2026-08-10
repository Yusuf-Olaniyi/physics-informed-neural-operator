"""Fourier Neural Operator for the beam forward/inverse problems.

Default `input_channels=2` reflects the input layout [signal(x), x]  (signal is q(x) for the
forward operator, or w(x) for the inverse operator).
"""
import torch.nn as nn
import torch.nn.functional as F
from .fno_block import FNOBlock1d


class BeamFNO(nn.Module):
    def __init__(self, input_channels: int = 2, output_channels: int = 1,
                 width: int = 64, modes: int = 16, num_layers: int = 4):
        super().__init__()

        self.input_layer = nn.Linear(input_channels, width)
        self.layers = nn.ModuleList(
            [FNOBlock1d(width, modes) for _ in range(num_layers)]
        )

        self.output_layer1 = nn.Linear(width, 128)
        self.output_layer2 = nn.Linear(128, output_channels)

    def forward(self, x):
        """x shape: (batch, num_nodes, input_channels)"""
        x = self.input_layer(x)

        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)

        x = self.output_layer1(x)
        x = F.gelu(x)
        x = self.output_layer2(x)
        return x
