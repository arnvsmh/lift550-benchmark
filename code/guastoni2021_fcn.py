"""
guastoni2021_fcn.py
===================
Fully-Convolutional Network (FCN) from Guastoni et al., JFM 928 A27, 2021.

The paper trains two CNN variants to predict 2D velocity-fluctuation fields
at different wall-normal locations from wall quantities (wall shear stress,
wall pressure). In LIFT-550 we adapt the FCN as a pure SR baseline:
    input  : degraded velocity field (LR proxy)
    output : reconstructed DNS velocity field

Architecture (from paper Table 1 and §2.2):
    - Fully convolutional: no dense layers, no pooling
    - 5 convolutional layers (we follow the FCN described, not FCN-POD)
    - 32 filters per layer (from paper)
    - 3×3 kernels
    - LeakyReLU activations (negative_slope=0.2, from paper §2.2)
    - Final layer: linear output (no activation) for velocity regression
    - Same-padding throughout to preserve spatial dimensions

The paper also presents an FCN-POD variant (using POD basis functions)
but the plain FCN is the more direct baseline and is more commonly cited.

Ref: L. Guastoni et al.,
     "Convolutional-network models to predict wall-bounded turbulence
      from wall quantities,"
     J. Fluid Mech. 928, A27, 2021.
     Dataset: KTH-FlowAI (same as LIFT-550 benchmark).
"""

import torch
import torch.nn as nn


class Guastoni2021_FCN(nn.Module):
    """
    FCN from Guastoni et al. 2021.

    Uses the same KTH DNS dataset as LIFT-550 — most directly comparable
    baseline. In LIFT-550 adapted as spatial SR: LR field → HR DNS field.

    In LIFT-550:
        input  : degraded velocity field [B, 3, H, W]  (normalized)
        output : reconstructed DNS field [B, 3, H, W]  (normalized)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 n_filters: int = 32, n_layers: int = 5,
                 kernel_size: int = 3, negative_slope: float = 0.2):
        """
        Args:
            in_channels    : input velocity components (3)
            out_channels   : output velocity components (3)
            n_filters      : convolutional filters per layer (32 per paper)
            n_layers       : total conv layers including input/output (5 per paper)
            kernel_size    : conv kernel size (3 per paper)
            negative_slope : LeakyReLU slope (0.2 per paper §2.2)
        """
        super().__init__()

        assert n_layers >= 2, "Need at least input and output layers"
        pad = kernel_size // 2

        layers = []

        # Input conv + activation
        layers.append(nn.Conv2d(in_channels, n_filters, kernel_size, padding=pad))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size, padding=pad))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

        # Output conv — linear (no activation, velocity regression)
        layers.append(nn.Conv2d(n_filters, out_channels, kernel_size, padding=pad))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize with He initialization appropriate for LeakyReLU.
        The paper does not specify initialization; we use kaiming_uniform
        with a=negative_slope which is the PyTorch default for LeakyReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2,
                                         nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = Guastoni2021_FCN()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"Guastoni2021_FCN | params: {n:,} | output: {tuple(y.shape)}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("PASS")
