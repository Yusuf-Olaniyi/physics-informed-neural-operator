"""1D Fourier (spectral) convolution layer, the core building block of the FNO."""
import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D Fourier convolution layer.

    Applies a learned complex-valued linear transform to the lowest
    `modes` Fourier coefficients of the input signal, then transforms back
    to physical space.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, in_channels, num_points)"""
        batch_size = x.shape[0]
        num_points = x.shape[-1]

        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device,
        )

        modes = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )

        x_out = torch.fft.irfft(out_ft, n=num_points, dim=-1)
        return x_out
