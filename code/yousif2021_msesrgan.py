"""
yousif2021_msesrgan.py
======================
MS-ESRGAN Generator from Yousif, Yu & Lim, arXiv:2110.05047, 2021.

This paper applies Multi-Scale Enhanced Super-Resolution GAN (MS-ESRGAN)
to reconstruct high-resolution turbulent velocity fields from low-resolution
data at various Reynolds numbers — the most relevant baseline for the
cross-Re finding in LIFT-550.

Architecture (from paper Fig. 1 and Table I):
    Generator G:
        1. Input conv: 64 filters, 3×3
        2. N_RRDB Residual-in-Residual Dense Blocks (RRDB)
           - Each RRDB = 3 Dense Blocks (DB)
           - Each DB = 5 dense conv layers, growth_rate=32, LeakyReLU(0.2)
           - Residual scaling β=0.2 within DB and RRDB
        3. Trunk conv: 64 filters, 3×3
        4. Multi-Scale Part (MSP) — 3 parallel branches (Table I):
           Branch k: Conv(k,k) → UpSamp(2) → Conv(k,k) → LeakyReLU →
                     UpSamp(2) → Conv(k,k) → LeakyReLU → UpSamp(2) → Conv(k,k)
           Kernel sizes: 3, 5, 7
           Add(branch1, branch2, branch3)
        5. Final conv: 3×3 → out_channels

LIFT-550 adaptation:
    - Generator ONLY, trained with MSE loss (no discriminator)
    - No pixel-shuffle upsampling: input and output are SAME resolution
      (degradation reduces spectral content, not grid size after preprocessing)
    - MSP: upsampling branches removed (they exist for pixel-shuffle SR
      which we don't need). Kept 3 parallel multi-scale conv branches.
    - n_rrdb=6 (paper uses 6 RRDB blocks, from §II.A)

Running generator-only with MSE loss for fair comparison with supervised
baselines is standard practice in turbulence SR literature
(e.g., Yousif themselves compare with MSE-only variants).

Ref: M.Z. Yousif, L. Yu & H.-C. Lim,
     "Super-resolution reconstruction of turbulent flows at various
      Reynolds numbers based on generative adversarial networks,"
     arXiv:2110.05047, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    """Single conv layer in a Dense Block, with dense connectivity."""
    def __init__(self, in_channels: int, growth_rate: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _DenseBlock(nn.Module):
    """
    Dense Block with 5 dense layers (as in ESRGAN RRDB).
    Each layer takes concatenation of all previous feature maps as input.
    Residual scaling β=0.2 applied to the block output.
    """
    def __init__(self, channels: int = 64, growth_rate: int = 32,
                 beta: float = 0.2):
        super().__init__()
        self.beta = beta
        self.d1 = _DenseLayer(channels,                 growth_rate)
        self.d2 = _DenseLayer(channels +   growth_rate, growth_rate)
        self.d3 = _DenseLayer(channels + 2*growth_rate, growth_rate)
        self.d4 = _DenseLayer(channels + 3*growth_rate, growth_rate)
        # Last layer maps back to channels (not growth_rate)
        self.d5 = nn.Conv2d(channels + 4*growth_rate, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.d1(x)
        x2 = self.d2(torch.cat([x,  x1],          dim=1))
        x3 = self.d3(torch.cat([x,  x1, x2],      dim=1))
        x4 = self.d4(torch.cat([x,  x1, x2, x3],  dim=1))
        x5 = self.d5(torch.cat([x,  x1, x2, x3, x4], dim=1))
        return x5 * self.beta + x   # residual scaling


class _RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).
    3 Dense Blocks with residual scaling β=0.2 on the whole RRDB output.
    """
    def __init__(self, channels: int = 64, growth_rate: int = 32,
                 beta: float = 0.2):
        super().__init__()
        self.beta = beta
        self.db1 = _DenseBlock(channels, growth_rate, beta)
        self.db2 = _DenseBlock(channels, growth_rate, beta)
        self.db3 = _DenseBlock(channels, growth_rate, beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db3(self.db2(self.db1(x)))
        return out * self.beta + x   # residual scaling


class Yousif2021_MSESRGANGenerator(nn.Module):
    """
    MS-ESRGAN Generator adapted for LIFT-550 (no pixel-shuffle, MSE loss).

    Architecture:
        head → trunk (N RRDB) → trunk_conv → [global residual] →
        MSP (3 parallel multi-scale branches) → msp_out

    In LIFT-550:
        input  : degraded velocity field [B, 3, H, W]  (normalized)
        output : reconstructed DNS field [B, 3, H, W]  (normalized)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 n_feat: int = 64, n_rrdb: int = 6, growth_rate: int = 32,
                 beta: float = 0.2, msp_mid: int = 16):
        """
        Args:
            in_channels  : input velocity components (3)
            out_channels : output velocity components (3)
            n_feat       : feature channels in RRDB trunk (64 per paper)
            n_rrdb       : number of RRDB blocks (6 per paper §II.A)
            growth_rate  : dense growth rate (32 per ESRGAN standard)
            beta         : residual scaling factor (0.2 per paper)
            msp_mid      : channels in each MSP branch (16, from Table I)
        """
        super().__init__()

        # Input feature extraction
        self.head = nn.Conv2d(in_channels, n_feat, 3, padding=1)

        # RRDB trunk
        self.trunk = nn.Sequential(
            *[_RRDB(n_feat, growth_rate, beta) for _ in range(n_rrdb)]
        )
        self.trunk_conv = nn.Conv2d(n_feat, n_feat, 3, padding=1)

        # Multi-Scale Part (MSP) — 3 parallel branches (Table I)
        # Each branch: Conv(k) → LeakyReLU → Conv(k) → LeakyReLU → Conv(k)
        # Kernel sizes: 3, 5, 7

        def _msp_branch(k):
            pad = k // 2
            return nn.Sequential(
                nn.Conv2d(n_feat, msp_mid, k, padding=pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(msp_mid, msp_mid, k, padding=pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(msp_mid, msp_mid, k, padding=pad),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.msp_branch1 = _msp_branch(3)
        self.msp_branch2 = _msp_branch(5)
        self.msp_branch3 = _msp_branch(7)

        # MSP final conv: concat 3 branches → output
        self.msp_out = nn.Conv2d(3 * msp_mid, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2,
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Last conv of trunk and msp_out: small init to stabilize training
        nn.init.normal_(self.trunk_conv.weight, 0, 0.02)
        nn.init.normal_(self.msp_out.weight, 0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.head(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat  = feat + trunk            # global residual connection

        m1 = self.msp_branch1(feat)
        m2 = self.msp_branch2(feat)
        m3 = self.msp_branch3(feat)

        msp = torch.cat([m1, m2, m3], dim=1)
        return self.msp_out(msp)


if __name__ == "__main__":
    model = Yousif2021_MSESRGANGenerator()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"Yousif2021_MSESRGANGenerator | params: {n:,} | output: {tuple(y.shape)}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("PASS")
