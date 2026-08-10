"""A single Fourier Neural Operator block: spectral conv + pointwise linear + activation."""
import torch.nn as nn
import torch.nn.functional as F
from .spectral_conv import SpectralConv1d


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        """x shape: (batch, width, num_points)"""
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        return F.gelu(x1 + x2)
