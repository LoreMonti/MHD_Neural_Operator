"""Fourier Neural Operator for one-step MHD evolution.

The model learns the map  field(t) -> field(t + dt), where a "field" is the 2-channel
stack (omega, a), conditioned on the physical parameters (M_A, Re).

Why Fourier layers? A convolution is local (a pixel and its neighbours), but plasma
structures interact across the whole box. In Fourier space each mode already spans the
domain, so a single spectral layer mixes information globally. Keeping only the low
modes makes the operator compact and *resolution-independent*: the learned weights act
on wavenumbers, not pixels, so a model trained at 128^2 can be evaluated at 256^2.

Architecture (standard FNO, Li et al. 2020):
    lift (1x1 conv)  ->  [SpectralConv2d + pointwise conv + activation] x n_layers
                     ->  project (1x1 convs)  ->  residual add to the input fields.
The model predicts the *increment* (added to the input), which helps autoregressive
rollout stay stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Global spectral convolution: multiply the low Fourier modes by learned weights."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1   # kept modes along the first spatial axis (both ends)
        self.modes2 = modes2   # kept modes along the second axis (rfft: only low end)

        # Learnable complex weights for the two retained corners of the spectrum.
        scale = 1.0 / (in_channels * out_channels)
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _mul(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # (B, Cin, X, Y) x (Cin, Cout, X, Y) -> (B, Cout, X, Y), complex.
        return torch.einsum("bixy,ioxy->boxy", inp, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)   # (B, Cin, H, W//2+1)

        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.weight1)
        out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.weight2)

        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    """FNO mapping (omega, a) + (M_A, Re) at time t to (omega, a) at t + dt."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        n_params: int = 2,
        modes: int = 32,
        width: int = 64,
        n_layers: int = 4,
        residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_params = n_params
        self.residual = residual

        # Lift: fields + parameter channels -> width feature channels.
        self.lift = nn.Conv2d(in_channels + n_params, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        # Project features back to the output fields.
        self.project1 = nn.Conv2d(width, 128, kernel_size=1)
        self.project2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, fields: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """fields: (B, in_channels, H, W); params: (B, n_params)."""
        b, _, h, w = fields.shape
        # Broadcast the (already-normalized) parameters into constant channels.
        p = params.view(b, self.n_params, 1, 1).expand(b, self.n_params, h, w)
        x = torch.cat([fields, p], dim=1)

        x = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = F.gelu(spec(x) + pw(x))

        x = F.gelu(self.project1(x))
        out = self.project2(x)
        if self.residual:
            out = out + fields[:, : self.out_channels]
        return out
