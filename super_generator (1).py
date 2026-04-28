import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from .ifnet import IFNet


# =====================================================================================
# SS2D (Full Implementation – 4 Direction Scanning)
# =====================================================================================

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand  = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )

        self.act = nn.SiLU()

        self.x_proj_weight = nn.Parameter(
            torch.stack([
                nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False).weight
                for _ in range(4)
            ], dim=0)
        )

        self.dt_projs_weight = nn.Parameter(
            torch.stack([
                nn.Linear(self.dt_rank, self.d_inner).weight
                for _ in range(4)
            ], dim=0)
        )

        self.dt_projs_bias = nn.Parameter(torch.zeros(4, self.d_inner))

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        )
        self.A_logs = nn.Parameter(torch.log(A).repeat(4, 1, 1))
        self.Ds     = nn.Parameter(torch.ones(4, self.d_inner))

        self.selective_scan = selective_scan_fn
        self.out_norm  = nn.LayerNorm(self.d_inner)
        self.out_proj  = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout   = nn.Dropout(dropout) if dropout > 0 else None

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W

        x_hw = torch.stack([
            x.reshape(B, C, L),
            x.transpose(2, 3).contiguous().reshape(B, C, L)
        ], dim=1)

        xs = torch.cat([x_hw, torch.flip(x_hw, dims=[-1])], dim=1)  # 4 dirs

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.reshape(B, 4, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.reshape(B, 4, -1, L), self.dt_projs_weight
        )

        xs  = xs.reshape(B, -1, L).float()
        dts = dts.reshape(B, -1, L).float()
        Bs  = Bs.float().reshape(B, 4, -1, L)
        Cs  = Cs.float().reshape(B, 4, -1, L)
        As  = -torch.exp(self.A_logs.float()).reshape(-1, self.d_state)
        Ds  = self.Ds.float().reshape(-1)
        bias = self.dt_projs_bias.float().reshape(-1)

        out = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=bias, delta_softplus=True, return_last_state=False,
        )
        out   = out.reshape(B, 4, -1, L)
        inv   = torch.flip(out[:, 2:4], dims=[-1])
        wh    = out[:, 1].reshape(B, -1, W, H).transpose(2, 3).reshape(B, -1, L)
        invwh = inv[:, 1].reshape(B, -1, W, H).transpose(2, 3).reshape(B, -1, L)
        return out[:, 0], inv[:, 0], wh, invwh

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).reshape(B, H, W, self.d_inner)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# =====================================================================================
# Channel Attention (SE-style)
# =====================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =====================================================================================
# Conv Block (GroupNorm)
# =====================================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =====================================================================================
# VSSBlock (Spatial Mamba)
# =====================================================================================

class VSSBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm  = nn.LayerNorm(hidden_dim)
        self.mamba = SS2D(hidden_dim)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        x = x.permute(0, 2, 3, 1)
        x = self.mamba(self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return identity + x * self.scale.view(1, -1, 1, 1)


# =====================================================================================
# Rotational Temporal Mamba
# =====================================================================================

class RotationalTemporalMamba(nn.Module):
    def __init__(self, in_channels=6, num_frames=2):
        super().__init__()
        self.color_channels = in_channels // num_frames
        self.num_frames     = num_frames

        self.norm_tw  = nn.LayerNorm(self.color_channels)
        self.mamba_tw = SS2D(self.color_channels)

        self.norm_th  = nn.LayerNorm(self.color_channels)
        self.mamba_th = SS2D(self.color_channels)

        self.scale = nn.Parameter(torch.ones(self.color_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        x_time   = x.reshape(B, self.num_frames, self.color_channels, H, W)
        identity = x_time

        # Time-Width
        x_tw = x_time.permute(0, 3, 2, 1, 4)
        x_tw = x_tw.reshape(B * H, self.color_channels, self.num_frames, W)
        x_tw = x_tw.permute(0, 2, 3, 1)
        m_tw = self.mamba_tw(self.norm_tw(x_tw))
        m_tw = m_tw.permute(0, 3, 1, 2)
        m_tw = m_tw.reshape(B, H, self.color_channels, self.num_frames, W)
        m_tw = m_tw.permute(0, 3, 2, 1, 4)

        # Time-Height
        x_th = x_time.permute(0, 4, 2, 1, 3)
        x_th = x_th.reshape(B * W, self.color_channels, self.num_frames, H)
        x_th = x_th.permute(0, 2, 3, 1)
        m_th = self.mamba_th(self.norm_th(x_th))
        m_th = m_th.permute(0, 3, 1, 2)
        m_th = m_th.reshape(B, W, self.color_channels, self.num_frames, H)
        m_th = m_th.permute(0, 3, 2, 4, 1)

        out = identity + (m_tw + m_th) * self.scale.view(1, 1, -1, 1, 1)
        return out.reshape(B, C, H, W)


# =====================================================================================
# SuperGenerator — PSNR-Optimised U-Net with Mamba Blocks
# =====================================================================================

class SuperGenerator(nn.Module):
    """
    U-Net style generator with:
      - RotationalTemporalMamba for cross-frame feature exchange
      - VSSBlock (spatial Mamba) in every encoder stage
      - SE-style channel attention at skip connections

    When called from FlowGuidedVFI, `x` contains warped frames [wf1, wf3]
    and `baseline` is α·wf1 + (1-α)·wf3 provided by IFNet.
    When called standalone, `baseline` defaults to 0.5·(f1+f3).
    """

    def __init__(
        self,
        in_channels=6,
        out_channels=3,
        features=[48, 96, 192, 384],
        num_frames=2,
    ):
        super().__init__()

        self.temporal = RotationalTemporalMamba(in_channels, num_frames)

        self.encoder = nn.ModuleList()
        self.pool    = nn.MaxPool2d(2)

        current = in_channels
        for f in features:
            self.encoder.append(
                nn.Sequential(ConvBlock(current, f), VSSBlock(f))
            )
            current = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.up      = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attn    = nn.ModuleList()

        rev     = features[::-1]
        current = features[-1] * 2
        for f in rev:
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(current, f, 3, padding=1)
            ))
            self.attn.append(ChannelAttention(f))
            self.decoder.append(ConvBlock(f * 2, f))
            current = f

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x, baseline=None):
        """
        Args:
            x        : (B, 6, H, W)  concatenated frame pair [wf1|wf3] or [f1|f3]
            baseline : (B, 3, H, W)  optional pre-computed baseline.
                       If None, falls back to 0.5*(f1+f3) for standalone use.
        """
        if baseline is None:
            baseline = 0.5 * (x[:, :3] + x[:, 3:])

        x = self.temporal(x)

        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for i in range(len(self.up)):
            x    = self.up[i](x)
            skip = self.attn[i](skips[i])
            x    = torch.cat([x, skip], dim=1)
            x    = self.decoder[i](x)

        residual = self.final_conv(x)
        out      = baseline + residual
        out      = torch.clamp(out, -1.0, 1.0)
        return out


# =====================================================================================
# FlowGuidedVFI — Full Pipeline: IFNet + SuperGenerator
# =====================================================================================

class FlowGuidedVFI(nn.Module):
    """
    Complete video frame interpolation pipeline.

    Stage 1 — IFNet:
        Estimates bidirectional optical flows (f1→t, f3→t) at 3 pyramid
        scales (1/4 → 1/2 → 1) and produces per-pixel blend weight alpha.

    Stage 2 — Warp:
        wf1 = warp(f1, flow[:, :2])
        wf3 = warp(f3, flow[:, 2:])
        baseline = alpha * wf1 + (1 - alpha) * wf3

    Stage 3 — SuperGenerator:
        Sees [wf1, wf3] — a spatially-aligned frame pair — and predicts a
        residual on top of the IFNet baseline.  The RotationalTemporalMamba
        now processes temporally-coherent features, making its job easier.

    Training returns all intermediate artefacts for auxiliary losses.
    Inference: call `interpolate(f1, f3)` for a single-value return.

    Returns (training forward):
        out       (B, 3, H, W)   final interpolated frame, clamped [-1, 1]
        flow      (B, 4, H, W)   final bidirectional flow
        wf1       (B, 3, H, W)   warped frame 1
        wf3       (B, 3, H, W)   warped frame 3
        flow_list list[Tensor]   per-block intermediate flows  (length 3)
                                   used for multi-scale auxiliary warp loss
    """

    def __init__(
        self,
        out_channels=3,
        features=[48, 96, 192, 384],
        num_frames=2,
    ):
        super().__init__()
        self.ifnet     = IFNet()
        self.generator = SuperGenerator(
            in_channels=6,
            out_channels=out_channels,
            features=features,
            num_frames=num_frames,
        )

    def forward(self, f1, f3):
        """
        Args:
            f1 : (B, 3, H, W)  frame before the target
            f3 : (B, 3, H, W)  frame after  the target
        """
        # ── Stage 1: optical flow ───────────────────────────────────────
        flow, alpha, wf1, wf3, flow_list = self.ifnet(f1, f3)

        # ── Stage 2: IFNet-guided baseline ─────────────────────────────
        baseline = alpha * wf1 + (1.0 - alpha) * wf3   # (B, 3, H, W)

        # ── Stage 3: residual refinement on warped frames ───────────────
        x   = torch.cat([wf1, wf3], dim=1)             # (B, 6, H, W)
        out = self.generator(x, baseline=baseline)

        return out, flow, wf1, wf3, flow_list

    @torch.no_grad()
    def interpolate(self, f1, f3):
        """Convenience method for inference — returns only the output frame."""
        out, *_ = self.forward(f1, f3)
        return out