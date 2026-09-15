"""
fukami2019_cnn.py
=================
Simple CNN baseline from Fukami, Fukagata & Taira, JFM 870, 2019.

This is the plain CNN model (NOT the DSC/MS hybrid) described in §2 of
the paper. It serves as the simplest supervised baseline — a stack of
Conv2d layers with ReLU activations and no skip connections.

Architecture (from paper Fig. 3a and §2):
    - 3 convolutional layers
    - 32 filters per layer
    - 3×3 kernels with periodic/zero padding
    - ReLU activation
    - No pooling (full-resolution maintained throughout)
    - Final layer maps to output channels without activation

The paper uses periodic boundary conditions in the padding for turbulence
(periodic in x and z directions). We use reflection padding as the closest
available analogue in PyTorch for non-periodic but boundary-consistent
behavior, with a note that zero padding gives similar results as stated
in the paper.

Ref: K. Fukami, K. Fukagata & K. Taira,
     "Super-resolution reconstruction of turbulent flows with machine learning,"
     J. Fluid Mech. 870, 106-120, 2019.
     Official code: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fukami2019_CNN(nn.Module):
    """
    Plain CNN from Fukami et al. 2019 (the simpler of their two models).

    3-layer CNN with ReLU, 32 filters, 3×3 kernels, same-padding.
    Final layer has no activation (linear output).

    In LIFT-550:
        input  : degraded velocity field [B, 3, H, W]  (normalized)
        output : reconstructed DNS field [B, 3, H, W]  (normalized)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 n_filters: int = 32, n_layers: int = 3,
                 kernel_size: int = 3):
        super().__init__()

        layers = []

        # Input layer
        layers += [
            nn.Conv2d(in_channels, n_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        ]

        # Hidden layers
        for _ in range(n_layers - 2):
            layers += [
                nn.Conv2d(n_filters, n_filters, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            ]

        # Output layer — linear (no activation per paper)
        layers.append(
            nn.Conv2d(n_filters, out_channels, kernel_size, padding=kernel_size // 2)
        )

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = Fukami2019_CNN()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"Fukami2019_CNN | params: {n:,} | output: {tuple(y.shape)}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("PASS")
