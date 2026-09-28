import math
import os

import cv2
import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY

import lpips
# alex = lpips.LPIPS(net='alex').cuda()

@METRIC_REGISTRY.register()
def calculate_lpips(
    img,
    img2,
    crop_border,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )

    lpips_ = []
    img = torch.from_numpy(img)
    img2 = torch.from_numpy(img2)
    img = img.cuda()
    img2 = img2.cuda()
    batch = img.shape[0]
    for i in range(batch):
        img1 = img[i, ...].unsqueeze(0)
        imgx = img2[i, ...].unsqueeze(0)


        if crop_border != 0:
            img1 = img[..., crop_border:-crop_border, crop_border:-crop_border]
            imgx = imgx[..., crop_border:-crop_border, crop_border:-crop_border]

        lpips_.append(alex(img1, imgx, normalize=True).item())


    return np.array(lpips_).sum()
