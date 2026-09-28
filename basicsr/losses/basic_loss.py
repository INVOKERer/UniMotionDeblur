import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

# from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

from ..metrics.psnr_ssim import calculate_ssim_pt
from .loss_util import weighted_loss
import kornia
_reduction_modes = ["none", "mean", "sum"]


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


@weighted_loss
def huber_loss(pred, target, delta=0.01):
    abs_error = torch.abs(pred - target)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    losses = 0.5 * torch.pow(quadratic, 2) + linear
    return losses


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean") -> None:
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        return self.loss_weight * F.cross_entropy(
            pred, target, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += self.loss_weight * l1_loss(
                predi, target, weight, reduction=self.reduction
            )
            return loss / len(pred)
        else:
            return self.loss_weight * l1_loss(
                pred, target, weight, reduction=self.reduction
            )
@LOSS_REGISTRY.register()
class FourierLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FourierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        # self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target, weight=None):
        if weight is None:
            weight = 1.
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = (torch.fft.rfft2(predi) - torch.fft.rfft2(target)) * weight
                # diff = torch.fft.fftn(predi, dim=[-2,-1]) - torch.fft.fftn(target, dim=[-2,-1])
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * loss_freq
            return loss / len(pred)
        else:
            diff = (torch.fft.rfft2(pred) - torch.fft.rfft2(target)) * weight
            # diff = torch.rfft(pred, signal_ndim=2, normalized=False, onesided=False) - torch.rfft(target, signal_ndim=2, normalized=False, onesided=False)
            # diff = torch.fft.fft(torch.fft.rfft(pred, dim=-1), dim=-2) - torch.fft.fft(torch.fft.rfft(target, dim=-1), dim=-2)
            loss = torch.mean(torch.abs(diff))
            # print(loss)
            return self.loss_weight * loss
@LOSS_REGISTRY.register()
class MultiScaleFourierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_weight_scale=[1., 1., 1.], reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.freq_loss = FourierLoss(loss_weight, reduction)
        # self.l1_loss = L1Loss(loss_weight, reduction)
        self.loss_weight_scale = loss_weight_scale
    def forward(self, pred, target, weight=None):
        tar = target
        loss = 0.
        if weight is None:
            weight = 1.
        if isinstance(pred, list):
            tars = kornia.geometry.transform.build_pyramid(tar, len(pred))
            for (i, predi) in enumerate(pred[::-1]):

                loss += self.freq_loss(predi, tars[i]) * weight * self.loss_weight_scale[i]
                # loss += self.l1_loss(predi, tars[i]) * weight * self.loss_weight_scale[i]
                # tar = F.interpolate(tar, scale_factor=0.5)
        else:
            loss += self.freq_loss(pred, tar) * weight * self.loss_weight_scale[0]
            # loss += self.l1_loss(pred, tar) * weight * self.loss_weight_scale[0]
        return loss
@LOSS_REGISTRY.register()
class MultiScaleL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_weight_scale=[1., 1., 1.], reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        # self.freq_loss = FourierLoss(loss_weight_freq, reduction)
        self.l1_loss = L1Loss(loss_weight, reduction)
        self.loss_weight_scale = loss_weight_scale
    def forward(self, pred, target, weight=None):
        tar = target
        loss = 0.
        if weight is None:
            weight = 1.
        if isinstance(pred, list):
            tars = kornia.geometry.transform.build_pyramid(tar, len(pred))
            for (i, predi) in enumerate(pred[::-1]):

                # loss += self.freq_loss(predi, tars[i]) * weight * self.loss_weight_scale[i]
                loss += self.l1_loss(predi, tars[i]) * weight * self.loss_weight_scale[i]
                # tar = F.interpolate(tar, scale_factor=0.5)
        else:
            # loss += self.freq_loss(pred, tar) * weight * self.loss_weight_scale[0]
            loss += self.l1_loss(pred, tar) * weight * self.loss_weight_scale[0]
        return loss
@LOSS_REGISTRY.register()
class MultiScaleFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_weight_freq=1.0, loss_weight_scale=[1., 1., 1.], reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.freq_loss = FourierLoss(loss_weight_freq, reduction)
        self.l1_loss = L1Loss(loss_weight, reduction)
        self.loss_weight_scale = loss_weight_scale
    def forward(self, pred, target, weight=None):
        tar = target
        loss = 0.
        if weight is None:
            weight = 1.
        if isinstance(pred, list):
            tars = kornia.geometry.transform.build_pyramid(tar, len(pred))
            for (i, predi) in enumerate(pred[::-1]):

                loss += self.freq_loss(predi, tars[i]) * weight * self.loss_weight_scale[i]
                loss += self.l1_loss(predi, tars[i]) * weight * self.loss_weight_scale[i]
                # tar = F.interpolate(tar, scale_factor=0.5)
        else:
            loss += self.freq_loss(pred, tar) * weight * self.loss_weight_scale[0]
            loss += self.l1_loss(pred, tar) * weight * self.loss_weight_scale[0]
        return loss
@LOSS_REGISTRY.register()
class SmoothL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(SmoothL1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * F.smooth_l1_loss(
            pred, target, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class HuberLoss(nn.Module):
    """Huber (Smooth L1) loss.

    Args:
        loss_weight (float): Loss weight for Huber loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, delta=0.01, reduction="mean"):
        super(HuberLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * huber_loss(
            pred, target, weight, delta=self.delta, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """SSIM (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
    """

    def __init__(
        self,
        ssim_weight=0.1,
        mse_weight=1.0,
        crop_border=0,
        reduction="mean",
        test_y_channel=False,
    ):
        super(SSIMLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.ssim_weight * (
            1
            - calculate_ssim_pt(
                pred,
                target,
                crop_border=self.crop_border,
                test_y_channel=self.test_y_channel,
                image_range=1,
            )[0].mean()
        ) + self.mse_weight * huber_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class SSIMMSELoss(nn.Module):
    """SSIM (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
    """

    def __init__(
        self,
        ssim_weight=0.1,
        mse_weight=1.0,
        crop_border=0,
        reduction="mean",
        test_y_channel=False,
    ):
        super(SSIMMSELoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        return self.ssim_weight * (
            1
            - calculate_ssim_pt(
                pred,
                target,
                crop_border=self.crop_border,
                test_y_channel=self.test_y_channel,
                image_range=1,
                float64=False,
            ).mean()
        ) + self.mse_weight * mse_loss(pred, target, None, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction
        )


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. Supported ones are: mean | sum"
            )
        super(WeightedTVLoss, self).__init__(
            loss_weight=loss_weight, reduction=reduction
        )

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target, weight=None, **kwargs):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
