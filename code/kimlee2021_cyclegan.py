"""
kimlee2021_cyclegan.py
======================
CycleGAN generator from Kim, Kim, Won & Lee, JFM 910 A29, 2021.

The paper applies CycleGAN (Zhu et al. 2017) to unsupervised super-resolution
of turbulent flows, using unpaired LES and DNS data. It is the only unsupervised
method in the LIFT-550 baseline comparison.

Architecture:
    The paper uses the standard CycleGAN generator architecture from
    Zhu et al. 2017 (the original CycleGAN paper), which for 256×256
    images uses 9 ResNet blocks. This is confirmed by Kim & Lee referencing
    the original CycleGAN implementation.

    Generator G (LR → HR):
        Initial block:  ReflectionPad(3) → Conv(64, 7×7) → InstanceNorm → ReLU
        Downsample d128: Conv(128, 3×3, stride=2) → InstanceNorm → ReLU
        Downsample d256: Conv(256, 3×3, stride=2) → InstanceNorm → ReLU
        ResNet blocks × 9: each = ReflectPad→Conv→IN→ReLU→ReflectPad→Conv→IN
        Upsample u128: ConvTranspose(128, 3×3, stride=2) → InstanceNorm → ReLU
        Upsample u64:  ConvTranspose(64,  3×3, stride=2) → InstanceNorm → ReLU
        Output block:  ReflectionPad(3) → Conv(out, 7×7) → Tanh

    Note on Tanh output:
        CycleGAN uses Tanh to bound outputs in [-1, 1]. Since LIFT-550
        data is normalized to approximately [-3, 3], we remove the Tanh
        for regression accuracy (the output layer remains a linear conv).
        This is a deliberate LIFT-550 adaptation — for the supervised
        MSE regression task, unbounded output is strictly better.

LIFT-550 adaptation:
    - Generator G only, trained with MSE loss (supervised mode)
    - Paired data available → no cycle consistency loss needed
    - Tanh removed from output (better for regression, bounded data not required)
    - InstanceNorm kept (prevents batch-size-1 issues, standard for GANs)

Ref: J. Kim, J. Kim, J. Won & C. Lee,
     "Unsupervised deep learning for super-resolution reconstruction
      of turbulence,"
     J. Fluid Mech. 910, A29, 2021.
     Original CycleGAN: J.-Y. Zhu et al., ICCV 2017.
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """
    ResNet block as used in CycleGAN (Zhu et al. 2017).
    Uses ReflectionPad2d + InstanceNorm (not BatchNorm).
    No dropout (Kim & Lee do not mention dropout).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=True),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=True),
            nn.InstanceNorm2d(channels, affine=False),
            # No ReLU after second conv in ResNet block (He et al. 2016)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)    # residual connection


class KimLee2021_CycleGANGenerator(nn.Module):
    """
    CycleGAN generator G, supervised mode for LIFT-550.

    Architecture follows Zhu et al. 2017 (9-block version for 256×256).
    Adapted for LIFT-550: Tanh removed, MSE loss, supervised training.

    In LIFT-550:
        input  : degraded velocity field [B, 3, H, W]  (normalized)
        output : reconstructed DNS field [B, 3, H, W]  (normalized, linear)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 n_filters: int = 64, n_resblocks: int = 9):
        """
        Args:
            in_channels  : input channels (3 for u/v/w)
            out_channels : output channels (3 for u/v/w)
            n_filters    : base filter count (64 per CycleGAN standard)
            n_resblocks  : number of ResNet blocks (9 for 256×256 per paper)
        """
        super().__init__()

        # ── Initial block (c7s1-64) ──────────────────────────────────────────
        self.init_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, n_filters, 7, bias=True),
            nn.InstanceNorm2d(n_filters, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Downsampling (d128, d256) ────────────────────────────────────────
        self.down1 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(n_filters * 2, affine=False),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(n_filters * 4, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── ResNet blocks (R256 × n_resblocks) ──────────────────────────────
        self.res_blocks = nn.Sequential(
            *[_ResBlock(n_filters * 4) for _ in range(n_resblocks)]
        )

        # ── Upsampling (u128, u64) ───────────────────────────────────────────
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(n_filters * 2, affine=False),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 2, n_filters, 3,
                               stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(n_filters, affine=False),
            nn.ReLU(inplace=True),
        )

        # ── Output block (c7s1-out, linear for regression) ──────────────────
        # Tanh REMOVED for LIFT-550 regression task
        self.out_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, out_channels, 7, bias=True),
            # No Tanh — linear output for normalized velocity regression
        )

        self._init_weights()

    def _init_weights(self):
        """
        CycleGAN paper uses normal distribution N(0, 0.02) for weight init.
        InstanceNorm params follow N(1, 0.02) for weight, 0 for bias.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_block(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.out_block(x)
        return x


if __name__ == "__main__":
    model = KimLee2021_CycleGANGenerator()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"KimLee2021_CycleGANGenerator | params: {n:,} | output: {tuple(y.shape)}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("PASS")
