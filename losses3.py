import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# Charbonnier Loss (PSNR-friendly pixel loss)
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


# ============================================================
# Perceptual Loss (VGG16 relu2_2, with correct ImageNet norm)
# ============================================================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred   = ((pred   + 1) / 2 - self.mean) / self.std
        target = ((target + 1) / 2 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred), self.vgg(target))


# ============================================================
# Edge Loss (Sobel-based, unnormalized)
# ============================================================

class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, x):
        gx = F.conv2d(x, self.sobel_x.expand(x.size(1), -1, -1, -1), padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.sobel_y.expand(x.size(1), -1, -1, -1), padding=1, groups=x.size(1))
        return gx, gy

    def forward(self, pred, target):
        gx_pred, gy_pred = self.gradient(pred)
        gx_gt,   gy_gt   = self.gradient(target)
        return F.l1_loss(gx_pred, gx_gt) + F.l1_loss(gy_pred, gy_gt)


# ============================================================
# SSIM Loss (avg_pool approximation, consistent with original)
# ============================================================

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        pred   = (pred   + 1) / 2
        target = (target + 1) / 2
        return 1 - self._ssim(pred, target)

    def _ssim(self, img1, img2):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ws, pad = self.window_size, self.window_size // 2
        mu1 = F.avg_pool2d(img1, ws, 1, pad)
        mu2 = F.avg_pool2d(img2, ws, 1, pad)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1  = F.avg_pool2d(img1 * img1, ws, 1, pad) - mu1_sq
        sigma2  = F.avg_pool2d(img2 * img2, ws, 1, pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, ws, 1, pad) - mu12
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1 + sigma2 + C2))
        return ssim_map.mean()


# ============================================================
# Frequency Loss (FFT-based high-frequency detail recovery)
# ============================================================

class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft   = torch.fft.rfft2(pred,   norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


# ============================================================
# Flow Smoothness Loss (TV regularization on optical flow)
# ============================================================

class FlowSmoothnessLoss(nn.Module):
    """
    Total-variation regularizer on the optical flow field.
    Penalizes sharp spatial discontinuities in flow — prevents
    noisy/degenerate flow estimates in textureless regions.

    A small weight (0.01–0.05) is sufficient; it acts as a soft
    prior rather than a strong constraint.
    """

    def forward(self, flow):
        """
        Args:
            flow : (B, 4, H, W)  bidirectional optical flow in pixel units
        """
        dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
        dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
        return dx.mean() + dy.mean()


# ============================================================
# Combined Loss (flow-aware version)
# ============================================================

class CombinedLoss(nn.Module):
    """
    All reconstruction losses for the generator, now extended with:

      lambda_warp        : L1 loss between final warped frames and GT.
                           This supervises IFNet directly and is the single
                           biggest contributor to IFNet learning speed.
                           Weight 10.0 is strong but appropriate — IFNet
                           has no other direct signal.

      lambda_flow_smooth : TV regularization on the optical flow field.
                           Keep very small (0.01) — enough to prevent noise
                           without blocking motion-discontinuity flows.

    Call signature
    --------------
    loss = criterion(fake, real)                    # standalone (no flow)
    loss = criterion(fake, real,                    # full flow-guided
                     wf1=wf1, wf3=wf3, flow=flow)
    """

    def __init__(
        self,
        lambda_pixel=100.0,
        lambda_perceptual=0.1,
        lambda_edge=5.0,
        lambda_ssim=0.5,
        lambda_freq=1.0,
        lambda_warp=10.0,
        lambda_flow_smooth=0.01,
    ):
        super().__init__()

        self.lambda_pixel       = lambda_pixel
        self.lambda_perceptual  = lambda_perceptual
        self.lambda_edge        = lambda_edge
        self.lambda_ssim        = lambda_ssim
        self.lambda_freq        = lambda_freq
        self.lambda_warp        = lambda_warp
        self.lambda_flow_smooth = lambda_flow_smooth

        self.pixel       = CharbonnierLoss()
        self.perceptual  = PerceptualLoss()
        self.edge        = EdgeAwareLoss()
        self.ssim        = SSIMLoss()
        self.freq        = FrequencyLoss()
        self.flow_smooth = FlowSmoothnessLoss()

    def forward(self, fake, real, wf1=None, wf3=None, flow=None):
        """
        Args:
            fake  : (B, 3, H, W)  generator output
            real  : (B, 3, H, W)  ground-truth frame
            wf1   : (B, 3, H, W)  warped frame 1 from IFNet  (optional)
            wf3   : (B, 3, H, W)  warped frame 3 from IFNet  (optional)
            flow  : (B, 4, H, W)  bidirectional flow from IFNet  (optional)
        """
        # ── Core reconstruction on final output ──────────────────────
        loss  = self.lambda_pixel      * self.pixel(fake, real)
        loss += self.lambda_perceptual * self.perceptual(fake, real)
        loss += self.lambda_edge       * self.edge(fake, real)
        loss += self.lambda_ssim       * self.ssim(fake, real)
        loss += self.lambda_freq       * self.freq(fake, real)

        # ── Auxiliary warp supervision (trains IFNet directly) ────────
        if wf1 is not None and wf3 is not None:
            # Symmetrical: both warped frames should match GT
            warp_loss = F.l1_loss(wf1, real) + F.l1_loss(wf3, real)
            loss += self.lambda_warp * warp_loss

        # ── Flow smoothness (TV regularization) ───────────────────────
        if flow is not None:
            loss += self.lambda_flow_smooth * self.flow_smooth(flow)

        return loss