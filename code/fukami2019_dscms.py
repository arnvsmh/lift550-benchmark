"""
fukami2019_dscms.py
===================
Hybrid Downsampled Skip-Connection / Multi-Scale (DSC/MS) model.

Faithfully ported to PyTorch from the official Keras implementation:
    http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py

Architecture (directly from official code):

DSC branch (blue in Fig. 3c):
    Level 1: MaxPool(8×8) → Conv(32, 3×3) → Conv(32, 3×3) → Upsample(2×)
    Level 2: [up1 + MaxPool(4×4)] → Conv(32, 3×3) → Conv(32, 3×3) → Upsample(2×)
    Level 3: [up2 + MaxPool(2×2)] → Conv(32, 3×3) → Conv(32, 3×3) → Upsample(2×)
    Level 4: [up3 + input]         → Conv(32, 3×3) → Conv(32, 3×3)

MS branch (yellow in Fig. 3c, from Du et al. 2018):
    Path 1 (k=5):  Conv(16, 5×5) → Conv(8, 5×5) → Conv(8, 5×5)
    Path 2 (k=9):  Conv(16, 9×9) → Conv(8, 9×9) → Conv(8, 9×9)
    Path 3 (k=13): Conv(16,13×13)→ Conv(8,13×13)→ Conv(8,13×13)
    Merge: [path1 + path2 + path3 + input] → Conv(8, 7×7) → Conv(3, 5×5)

Final: [DSC_out + MS_out] → Conv(out_channels, 3×3)

Notes:
    - Official code uses MaxPooling (not AvgPool) for DSC downsampling
    - Merging is via concatenation ('concat' in Keras = torch.cat)
    - All activations are ReLU
    - Final conv has NO activation (linear output)
    - Filter counts: 32 (DSC), 16+8 (MS first+subsequent layers)

Ref: K. Fukami, K. Fukagata & K. Taira,
     "Super-resolution reconstruction of turbulent flows with machine learning,"
     J. Fluid Mech. 870, 106-120, 2019.
     Official code: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fukami2019_DSCMS(nn.Module):
    """
    DSC/MS hybrid model — PyTorch port of the official Keras implementation.

    In LIFT-550:
        input  : degraded velocity field [B, 3, H, W]  (normalized)
        output : reconstructed DNS field [B, 3, H, W]  (normalized)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 dsc_filters: int = 32, ms_first: int = 16, ms_sub: int = 8):
        """
        Args:
            in_channels  : input channels (3 for u/v/w velocity)
            out_channels : output channels (3 for u/v/w DNS)
            dsc_filters  : filters in DSC branch (32 in official code)
            ms_first     : filters in first MS conv (16 in official code)
            ms_sub       : filters in subsequent MS convs (8 in official code)
        """
        super().__init__()
        C  = dsc_filters
        F1 = ms_first
        Fs = ms_sub

        # ── DSC branch ──────────────────────────────────────────────────────
        # Level 1: pool/8, conv×2, upsample×2
        self.dsc_pool1  = nn.MaxPool2d(kernel_size=8, stride=8)
        self.dsc_conv1a = nn.Conv2d(in_channels, C, 3, padding=1)
        self.dsc_conv1b = nn.Conv2d(C, C, 3, padding=1)
        self.dsc_up1    = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=False)

        # Level 2: pool/4, concat(up1, pool/4), conv×2, upsample×2
        self.dsc_pool2  = nn.MaxPool2d(kernel_size=4, stride=4)
        # input channels = C (from up1) + in_channels (from pool/4)
        self.dsc_conv2a = nn.Conv2d(C + in_channels, C, 3, padding=1)
        self.dsc_conv2b = nn.Conv2d(C, C, 3, padding=1)
        self.dsc_up2    = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=False)

        # Level 3: pool/2, concat(up2, pool/2), conv×2, upsample×2
        self.dsc_pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channels = C (from up2) + in_channels (from pool/2)
        self.dsc_conv3a = nn.Conv2d(C + in_channels, C, 3, padding=1)
        self.dsc_conv3b = nn.Conv2d(C, C, 3, padding=1)
        self.dsc_up3    = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=False)

        # Level 4: concat(up3, input), conv×2
        # input channels = C (from up3) + in_channels (original input)
        self.dsc_conv4a = nn.Conv2d(C + in_channels, C, 3, padding=1)
        self.dsc_conv4b = nn.Conv2d(C, C, 3, padding=1)
        # DSC output: [B, C, H, W]

        # ── MS branch ───────────────────────────────────────────────────────
        # Path 1: kernel=5
        self.ms1_conv1 = nn.Conv2d(in_channels, F1, 5, padding=2)
        self.ms1_conv2 = nn.Conv2d(F1, Fs, 5, padding=2)
        self.ms1_conv3 = nn.Conv2d(Fs, Fs, 5, padding=2)

        # Path 2: kernel=9
        self.ms2_conv1 = nn.Conv2d(in_channels, F1, 9, padding=4)
        self.ms2_conv2 = nn.Conv2d(F1, Fs, 9, padding=4)
        self.ms2_conv3 = nn.Conv2d(Fs, Fs, 9, padding=4)

        # Path 3: kernel=13
        self.ms3_conv1 = nn.Conv2d(in_channels, F1, 13, padding=6)
        self.ms3_conv2 = nn.Conv2d(F1, Fs, 13, padding=6)
        self.ms3_conv3 = nn.Conv2d(Fs, Fs, 13, padding=6)

        # MS merge: concat(path1, path2, path3, input) → conv → conv
        # input channels = 3*Fs + in_channels
        self.ms_merge1 = nn.Conv2d(3 * Fs + in_channels, Fs, 7, padding=3)
        self.ms_merge2 = nn.Conv2d(Fs, 3, 5, padding=2)
        # MS output: [B, 3, H, W]

        # ── Final merge ─────────────────────────────────────────────────────
        # concat(DSC_out, MS_out) → conv (no activation)
        # input channels = C + 3
        self.final_conv = nn.Conv2d(C + 3, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── DSC branch ──────────────────────────────────────────────────────
        # Level 1
        d1 = self.dsc_pool1(x)              # [B, C_in, H/8, W/8]
        d1 = F.relu(self.dsc_conv1a(d1))
        d1 = F.relu(self.dsc_conv1b(d1))
        d1 = self.dsc_up1(d1)              # [B, C, H/4, W/4]

        # Level 2
        skip2 = self.dsc_pool2(x)          # [B, C_in, H/4, W/4]
        d2 = torch.cat([d1, skip2], dim=1) # [B, C+C_in, H/4, W/4]
        d2 = F.relu(self.dsc_conv2a(d2))
        d2 = F.relu(self.dsc_conv2b(d2))
        d2 = self.dsc_up2(d2)              # [B, C, H/2, W/2]

        # Level 3
        skip3 = self.dsc_pool3(x)          # [B, C_in, H/2, W/2]
        d3 = torch.cat([d2, skip3], dim=1) # [B, C+C_in, H/2, W/2]
        d3 = F.relu(self.dsc_conv3a(d3))
        d3 = F.relu(self.dsc_conv3b(d3))
        d3 = self.dsc_up3(d3)              # [B, C, H, W]

        # Level 4
        d4 = torch.cat([d3, x], dim=1)     # [B, C+C_in, H, W]
        d4 = F.relu(self.dsc_conv4a(d4))
        d4 = F.relu(self.dsc_conv4b(d4))   # [B, C, H, W]

        # ── MS branch ───────────────────────────────────────────────────────
        m1 = F.relu(self.ms1_conv1(x))
        m1 = F.relu(self.ms1_conv2(m1))
        m1 = F.relu(self.ms1_conv3(m1))    # [B, Fs, H, W]

        m2 = F.relu(self.ms2_conv1(x))
        m2 = F.relu(self.ms2_conv2(m2))
        m2 = F.relu(self.ms2_conv3(m2))    # [B, Fs, H, W]

        m3 = F.relu(self.ms3_conv1(x))
        m3 = F.relu(self.ms3_conv2(m3))
        m3 = F.relu(self.ms3_conv3(m3))    # [B, Fs, H, W]

        ms = torch.cat([m1, m2, m3, x], dim=1)  # [B, 3*Fs+C_in, H, W]
        ms = F.relu(self.ms_merge1(ms))
        ms = F.relu(self.ms_merge2(ms))     # [B, 3, H, W]

        # ── Final merge ─────────────────────────────────────────────────────
        out = torch.cat([d4, ms], dim=1)    # [B, C+3, H, W]
        out = self.final_conv(out)          # [B, out_channels, H, W] — linear
        return out


if __name__ == "__main__":
    model = Fukami2019_DSCMS()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n = sum(p.numel() for p in model.parameters())
    print(f"Fukami2019_DSCMS | params: {n:,} | output: {tuple(y.shape)}")
    assert y.shape == x.shape, "Output shape mismatch"
    print("PASS")
