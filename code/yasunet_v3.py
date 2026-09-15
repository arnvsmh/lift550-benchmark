"""
yasunet_v3.py
=============
YASU-Net v3 — Physics-informed turbulence super-resolution.

Architecture improvements over v1:
  1. GroupNorm(8) replaces InstanceNorm throughout — removes Re-dependent
     batch statistics while keeping gradient stability at batch=32
  2. base_ch=32 (~8M params) — better param/sample ratio for 70K samples
  3. BlurPool downsampling — anti-aliased, preserves vortex structure
  4. Circular padding throughout — correct for periodic DNS boundaries
  5. Multi-scale bottleneck (k=3,5,7) — captures structures at multiple
     spatial scales, directly addresses DSCMS advantage
  6. Fourier skip connections at every U-Net level — passes phase AND
     magnitude through each skip, giving decoder spectral info at all scales
  7. Dual FiLM — conditions both spatial and Fourier-domain features on y+
  8. Post-decoder full-resolution ResBlocks (3 blocks) — inspired by Kim
     CycleGAN's full-resolution residual refinement strength

Design principles:
  - Every downsampling uses BlurPool (Gaussian pre-filter + stride-2 conv)
  - Every convolution uses circular padding (turbulence is periodic in x,z)
  - GroupNorm(8) on all feature maps — Re-agnostic normalization
  - FiLM applied after every encoder/decoder block AND in Fourier domain
  - Fourier skip: at each level, pass rfft2 of encoder features to decoder
    via learned 1x1 conv on [real, imag] concatenation

Input/output: [B, 3, 256, 256] float32 (normalized velocity fields)
y+: [B] float32 scalar (wall-normal distance: 15, 30, 50, 100)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class CircConv2d(nn.Module):
    """Conv2d with circular padding — correct for periodic DNS boundaries."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.pad  = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, [self.pad]*4, mode="circular")
        return self.conv(x)


class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling (Zhang 2019).
    Applies a fixed Gaussian blur then strides by 2.
    Prevents aliasing of vortex structures during downsampling.
    """
    def __init__(self, channels):
        super().__init__()
        # 3x3 binomial filter (approx Gaussian)
        k = torch.tensor([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=torch.float32) / 16.0
        k = k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", k)
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], mode="circular")
        x = F.conv2d(x, self.kernel, stride=2,
                     groups=self.channels, padding=0)
        return x


def GroupNorm(channels):
    """GroupNorm(8) — Re-agnostic, stable at batch=32."""
    n_groups = min(8, channels)
    # ensure channels divisible by n_groups
    while channels % n_groups != 0 and n_groups > 1:
        n_groups -= 1
    return nn.GroupNorm(n_groups, channels, affine=True)


# ─────────────────────────────────────────────────────────────────────────────
# FiLM Conditioning
# ─────────────────────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Sinusoidal y+ embedding -> MLP -> (gamma, beta) for FiLM modulation.
    Same embedding as v1, now used for both spatial and Fourier conditioning.
    """
    def __init__(self, n_channels, embed_dim=16, hidden_dim=128):
        super().__init__()
        freqs = torch.exp(
            torch.linspace(math.log(0.01), math.log(10.0), embed_dim // 2)
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * n_channels),
        )
        nn.init.normal_(self.mlp[-1].weight, 0.0, 0.01)
        nn.init.constant_(self.mlp[-1].bias[:n_channels], 1.0)   # gamma init=1
        nn.init.zeros_(self.mlp[-1].bias[n_channels:])            # beta  init=0

    def forward(self, yp):
        yp_n   = (yp - 15.0) / 85.0
        angles = yp_n[:, None] * self.freqs[None, :] * 2 * math.pi
        embed  = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        out    = self.mlp(embed.to(next(self.mlp.parameters()).dtype))
        C      = out.shape[-1] // 2
        g = out[:, :C].unsqueeze(-1).unsqueeze(-1)
        b = out[:, C:].unsqueeze(-1).unsqueeze(-1)
        return g, b


def film_apply(feat, film_gen, yp):
    g, b = film_gen(yp)
    return g * feat + b


# ─────────────────────────────────────────────────────────────────────────────
# Core ConvBlock with GroupNorm
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two CircConv layers + GroupNorm + GELU. No FiLM (applied externally)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            CircConv2d(in_ch, out_ch),
            GroupNorm(out_ch),
            nn.GELU(),
            CircConv2d(out_ch, out_ch),
            GroupNorm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Block
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """ConvBlock + FiLM, no downsampling (downsampling done separately)."""
    def __init__(self, in_ch, out_ch, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.film = FiLMGenerator(out_ch, embed_dim, hidden_dim)

    def forward(self, x, yp):
        return film_apply(self.conv(x), self.film, yp)


# ─────────────────────────────────────────────────────────────────────────────
# Fourier Skip Connection
# ─────────────────────────────────────────────────────────────────────────────

class FourierSkip(nn.Module):
    """
    Passes encoder features to decoder in frequency domain.
    Computes rfft2 of encoder features, concatenates real+imag,
    applies learned 1x1 conv, then passes to decoder alongside
    spatial skip. Gives decoder phase AND magnitude info at each scale.

    Novel: no prior turbulence SR model uses Fourier-domain skip connections.
    """
    def __init__(self, channels):
        super().__init__()
        # 1x1 conv on [real, imag] concatenation -> channels
        self.proj = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
        nn.init.normal_(self.proj.weight, 0.0, 0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] encoder feature map
        Returns:
            fourier_skip: [B, C, H, W] frequency-domain features
        """
        x_ft  = torch.fft.rfft2(x.float())   # [B, C, H, W//2+1] complex
        # Reconstruct to full spatial size via irfft2 for consistent shape
        real  = x_ft.real                     # [B, C, H, W//2+1]
        imag  = x_ft.imag
        # Pad/crop to match spatial dimensions
        B, C, H, W = x.shape
        # Use irfft2 of real and imag parts separately as proxy features
        feat_r = torch.fft.irfft2(x_ft.real.unsqueeze(-1).squeeze(-1)
                                   .to(torch.cfloat), s=(H, W))
        feat_i = torch.fft.irfft2((1j * x_ft.imag).to(torch.cfloat), s=(H, W))
        feat_r = feat_r.float()
        feat_i = feat_i.float()
        combined = torch.cat([feat_r, feat_i], dim=1)  # [B, 2C, H, W]
        return self.proj(combined.to(x.dtype))          # [B, C, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample + concatenate spatial skip + Fourier skip + ConvBlock + FiLM.
    Receives both spatial and frequency information from encoder.
    """
    def __init__(self, in_ch, skip_ch, out_ch, embed_dim=16, hidden_dim=128):
        super().__init__()
        # in_ch (upsampled) + skip_ch (spatial) + skip_ch (fourier)
        self.conv = ConvBlock(in_ch + 2 * skip_ch, out_ch)
        self.film = FiLMGenerator(out_ch, embed_dim, hidden_dim)

    def forward(self, x, spatial_skip, fourier_skip, yp):
        x = F.interpolate(x, size=spatial_skip.shape[-2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, spatial_skip, fourier_skip], dim=1)
        return film_apply(self.conv(x), self.film, yp)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale Bottleneck
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleFourierBottleneck(nn.Module):
    """
    Novel bottleneck combining:
      1. Spectral conv (Fourier Neural Operator) — global frequency processing
      2. Multi-scale spatial branches (k=3, k=5, k=7) — local multi-scale
      3. Dual FiLM — conditions both spatial and spectral paths on y+

    The multi-scale spatial branches directly address why DSCMS beats v1:
    DSCMS processes flow at multiple spatial scales, this bottleneck does too.
    Combined with the Fourier path, it's more powerful than either alone.
    """
    def __init__(self, channels, modes_h=8, modes_w=8,
                 embed_dim=16, hidden_dim=128):
        super().__init__()
        self.channels = channels
        self.modes_h  = modes_h
        self.modes_w  = modes_w

        # Spectral conv weights
        scale = 1.0 / (channels * channels)
        self.w_r = nn.Parameter(
            scale * torch.randn(channels, channels, modes_h, modes_w))
        self.w_i = nn.Parameter(
            scale * torch.randn(channels, channels, modes_h, modes_w))

        # Spatial residual (1x1)
        self.res_conv = nn.Conv2d(channels, channels, 1)

        # Multi-scale spatial branches (k=3, k=5, k=7)
        mid = channels // 4
        self.ms3 = nn.Sequential(
            CircConv2d(channels, mid, 3), GroupNorm(mid), nn.GELU(),
            CircConv2d(mid, mid, 3),      GroupNorm(mid), nn.GELU(),
        )
        self.ms5 = nn.Sequential(
            CircConv2d(channels, mid, 5), GroupNorm(mid), nn.GELU(),
            CircConv2d(mid, mid, 5),      GroupNorm(mid), nn.GELU(),
        )
        self.ms7 = nn.Sequential(
            CircConv2d(channels, mid, 7), GroupNorm(mid), nn.GELU(),
            CircConv2d(mid, mid, 7),      GroupNorm(mid), nn.GELU(),
        )

        # Merge: spectral(channels) + ms3(mid) + ms5(mid) + ms7(mid) -> channels
        self.merge = nn.Sequential(
            nn.Conv2d(channels + 3 * mid, channels, 1),
            GroupNorm(channels),
            nn.GELU(),
        )

        # Dual FiLM: spatial + Fourier
        self.film_spatial  = FiLMGenerator(channels, embed_dim, hidden_dim)
        self.film_fourier  = FiLMGenerator(channels, embed_dim, hidden_dim)

    def _cmult(self, x, wr, wi):
        xr = x.real.float()
        xi = x.imag.float()
        wr = wr.float()
        wi = wi.float()
        return torch.complex(
            torch.einsum("bixy,oixy->boxy", xr, wr)
          - torch.einsum("bixy,oixy->boxy", xi, wi),
            torch.einsum("bixy,oixy->boxy", xr, wi)
          + torch.einsum("bixy,oixy->boxy", xi, wr),
        )

    def forward(self, x, yp):
        B, C, H, W = x.shape

        # ── Spectral path ────────────────────────────────────────────────────
        x_ft   = torch.fft.rfft2(x.float())
        out_ft = torch.zeros(B, C, H, W//2+1,
                             dtype=torch.cfloat, device=x.device)
        mh = min(self.modes_h, H//2)
        mw = min(self.modes_w, W//2+1)
        out_ft[:, :, :mh,  :mw] = self._cmult(
            x_ft[:, :, :mh,  :mw],
            self.w_r[:, :, :mh, :mw], self.w_i[:, :, :mh, :mw])
        out_ft[:, :, -mh:, :mw] = self._cmult(
            x_ft[:, :, -mh:, :mw],
            self.w_r[:, :, :mh, :mw], self.w_i[:, :, :mh, :mw])
        spectral = torch.fft.irfft2(out_ft, s=(H, W)).to(x.dtype)
        spectral = spectral + self.res_conv(x)

        # Apply Fourier FiLM — conditions spectral path on y+
        spectral = film_apply(spectral, self.film_fourier, yp)

        # ── Multi-scale spatial paths ────────────────────────────────────────
        ms3 = self.ms3(x)
        ms5 = self.ms5(x)
        ms7 = self.ms7(x)

        # ── Merge all paths ──────────────────────────────────────────────────
        merged = self.merge(torch.cat([spectral, ms3, ms5, ms7], dim=1))

        # Apply spatial FiLM — conditions merged output on y+
        return film_apply(merged, self.film_spatial, yp)


# ─────────────────────────────────────────────────────────────────────────────
# Full-resolution Residual Refinement Block (Kim-inspired)
# ─────────────────────────────────────────────────────────────────────────────

class ResRefinementBlock(nn.Module):
    """
    Full-resolution ResNet block with GroupNorm + CircConv.
    Applied post-decoder for full-resolution refinement.
    Inspired by Kim CycleGAN's success with deep full-resolution residual paths.
    Key: no spatial compression, refines the full-resolution feature map.
    """
    def __init__(self, channels, embed_dim=16, hidden_dim=128):
        super().__init__()
        self.block = nn.Sequential(
            CircConv2d(channels, channels, 3),
            GroupNorm(channels),
            nn.GELU(),
            CircConv2d(channels, channels, 3),
            GroupNorm(channels),
        )
        self.film = FiLMGenerator(channels, embed_dim, hidden_dim)

    def forward(self, x, yp):
        out = x + self.block(x)    # residual connection
        return film_apply(out, self.film, yp)


# ─────────────────────────────────────────────────────────────────────────────
# YASU-Net v2
# ─────────────────────────────────────────────────────────────────────────────

class YASUNetV2(nn.Module):
    """
    YASU-Net v2 — y+-conditioned Fourier U-Net with:
      - GroupNorm(8) throughout
      - BlurPool anti-aliased downsampling
      - Circular padding throughout
      - Multi-scale Fourier bottleneck (k=3,5,7 + spectral conv)
      - Fourier skip connections at every U-Net level
      - Dual FiLM (spatial + Fourier domain)
      - Post-decoder full-resolution residual refinement (6 ResBlocks)

    Input:  [B, 3, 256, 256] normalized degraded velocity field
    y+:     [B] float32 wall-normal distance (15, 30, 50, 100)
    Output: [B, 3, 256, 256] normalized DNS reconstruction
    """

    def __init__(
        self,
        in_channels  = 3,
        out_channels = 3,
        base_ch      = 32,
        embed_dim    = 16,
        hidden_dim   = 128,
        modes_h      = 10,    # v3: was 8 — more spectral coverage, params kept sane for 70K samples
        modes_w      = 10,    # v3: was 8
        n_refine     = 6,     # v3: was 3 — deeper full-res refinement (counters Kim)
    ):
        super().__init__()
        b   = base_ch
        ekw = dict(embed_dim=embed_dim, hidden_dim=hidden_dim)

        # ── Encoder ──────────────────────────────────────────────────────────
        # enc0: full res,  b   channels  [B, b,   256, 256]
        # enc1: /2,        b*2 channels  [B, b*2, 128, 128]
        # enc2: /4,        b*4 channels  [B, b*4,  64,  64]
        # enc3: /8,        b*8 channels  [B, b*8,  32,  32]
        self.enc0      = EncoderBlock(in_channels, b,   **ekw)
        self.blur0     = BlurPool2d(b)
        self.enc1      = EncoderBlock(b,   b*2, **ekw)
        self.blur1     = BlurPool2d(b*2)
        self.enc2      = EncoderBlock(b*2, b*4, **ekw)
        self.blur2     = BlurPool2d(b*4)
        self.enc3      = EncoderBlock(b*4, b*8, **ekw)
        self.blur3     = BlurPool2d(b*8)

        # ── Fourier skip connections ──────────────────────────────────────────
        self.fskip0    = FourierSkip(b)
        self.fskip1    = FourierSkip(b*2)
        self.fskip2    = FourierSkip(b*4)
        self.fskip3    = FourierSkip(b*8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Input: [B, b*8, 16, 16] after blur3
        self.bottleneck = MultiScaleFourierBottleneck(
            b*8, modes_h, modes_w, **ekw)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Each decoder gets: upsampled + spatial_skip + fourier_skip
        self.dec3      = DecoderBlock(b*8, b*8, b*4, **ekw)
        self.dec2      = DecoderBlock(b*4, b*4, b*2, **ekw)
        self.dec1      = DecoderBlock(b*2, b*2, b,   **ekw)
        self.dec0      = DecoderBlock(b,   b,   b,   **ekw)

        # ── Post-decoder full-resolution refinement ───────────────────────────
        self.refine    = nn.ModuleList([
            ResRefinementBlock(b, **ekw) for _ in range(n_refine)
        ])

        # ── Output head ──────────────────────────────────────────────────────
        self.head      = nn.Conv2d(b, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, yp):
        # ── Encoder + Fourier skips ──────────────────────────────────────────
        s0 = self.enc0(x,            yp)   # [B, b,   256, 256]
        f0 = self.fskip0(s0)               # [B, b,   256, 256]
        s1 = self.enc1(self.blur0(s0), yp) # [B, b*2, 128, 128]
        f1 = self.fskip1(s1)               # [B, b*2, 128, 128]
        s2 = self.enc2(self.blur1(s1), yp) # [B, b*4,  64,  64]
        f2 = self.fskip2(s2)               # [B, b*4,  64,  64]
        s3 = self.enc3(self.blur2(s2), yp) # [B, b*8,  32,  32]
        f3 = self.fskip3(s3)               # [B, b*8,  32,  32]

        # ── Bottleneck ────────────────────────────────────────────────────────
        bt = self.bottleneck(self.blur3(s3), yp)  # [B, b*8, 16, 16]

        # ── Decoder ──────────────────────────────────────────────────────────
        d3 = self.dec3(bt, s3, f3, yp)   # [B, b*4,  32,  32]
        d2 = self.dec2(d3, s2, f2, yp)   # [B, b*2,  64,  64]
        d1 = self.dec1(d2, s1, f1, yp)   # [B, b,   128, 128]
        d0 = self.dec0(d1, s0, f0, yp)   # [B, b,   256, 256]

        # ── Full-resolution refinement ────────────────────────────────────────
        out = d0
        for blk in self.refine:
            out = blk(out, yp)

        return self.head(out)              # [B, 3, 256, 256]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = YASUNetV2().to(device)
    print(f"Parameters: {model.count_parameters():,}")

    x  = torch.randn(2, 3, 256, 256, device=device)
    yp = torch.tensor([15.0, 100.0], device=device)

    # Forward pass
    with torch.no_grad():
        y = model(x, yp)
    assert y.shape == (2, 3, 256, 256), f"Wrong output shape: {y.shape}"
    assert not torch.isnan(y).any(),    "NaN in output"
    assert not torch.isinf(y).any(),    "Inf in output"
    print(f"Forward: {tuple(x.shape)} -> {tuple(y.shape)}")

    # Mixed precision
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type=="cuda")):
        y_amp = model(x, yp)
    assert y_amp.shape == (2, 3, 256, 256)
    print(f"Mixed precision: OK")

    # Gradient flow
    model.train()
    pred = model(x, yp)
    loss = F.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    if no_grad:
        print(f"WARNING: no gradient: {no_grad}")
    else:
        print(f"Gradient flow: OK")

    print(f"\nAll checks passed.")
    print(f"Parameters: {model.count_parameters():,}")
