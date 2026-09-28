## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import sys
sys.path.append(".")
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

def reorder_image(img, input_order="BHWC"):
    """Reorder images to 'BHWC' order.

    If the input_order is (h, w), return (1, h, w, 1);
    If the input_order is (c, h, w), return (1, h, w, c);
    If the input_order is (h, w, c), return (1, h, w, c);
    If the input_order is (b, h, w, c), return (b, h, w, c);
    If the input_order is (b, c, h, w), return (b, h, w, c);

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'BHWC' or 'BCHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'BHWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
        img = img[None, ...]
    if input_order == "BCHW":
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)
            img = img[None, ...]
        elif len(img.shape) == 4:
            img = img.transpose(0, 2, 3, 1)
    return img

def _ssim(img, img2, image_range=255):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * image_range) ** 2
    c2 = (0.03 * image_range) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(), cs_map.mean()
def calculate_ssim(
    img,
    img2,
    crop_border,
    input_order="BCHW",
    test_y_channel=False,
    image_range=255,
    **kwargs,
):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."
    if input_order not in ["BHWC", "BCHW"]:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"'
        )
    imgs = reorder_image(img, input_order=input_order)
    imgs2 = reorder_image(img2, input_order=input_order)
    dtype = np.uint8 if image_range == 255 else np.uint16

    ssims = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i, ...]
        img2 = imgs2[i, ...]

        if image_range != 1:
            img = (img * float(image_range)).round().astype(dtype)
            img2 = (img2 * float(image_range)).round().astype(dtype)
        if img.shape[-1] == img2.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        # if test_y_channel and img.shape[-1] == img2.shape[-1] == 3:
        #     img = to_y_channel(img, image_range)
        #     img2 = to_y_channel(img2, image_range)

        img = img.astype(np.float64)
        img2 = img2.astype(np.float64)
        ssims_ = []
        for j in range(img.shape[2]):
            ssim, _ = _ssim(img[..., j], img2[..., j], image_range)
            ssims_.append(ssim)
        ssims.append(np.array(ssims_).mean())
    return np.array(ssims).sum()

def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  # c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)

def compute_psnr_(image_true, image_test, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64)
  return 10 * np.log10((data_range ** 2) / err)
def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True,
                                               use_sample_covariance=False, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    # h_gt, w_gt = tar_img.shape[:2]
    # h, w = prd_img.shape[:2]
    # h_pad = max(0, h - h_gt)
    # w_pad = max(0, w - w_gt)
    # h_pad, w_pad = 64, 64
    # # print(prd_img.shape, tar_img.shape)
    # pad_mode = 'center'
    # if pad_mode == 'center':
    #     h1 = h_pad//2
    #     w1 = w_pad // 2
    #     prd_img = prd_img[h1:h_gt+h1, w1:w_gt+w1, :]
    # else:
    #     prd_img = prd_img[:h_gt, :w_gt, :]
    # print(prd_img.shape, tar_img.shape)
    SSIM_ = 0. # calculate_ssim(prd_img, tar_img, crop_border=0)
    PSNR_ = psnr_loss(prd_img, tar_img)
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0


    try:
        prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)
    except:
        print('x: ' + prd.split('/')[-1])
        cr1 = np.ones_like(prd_img)
    # print(prd_img.shape, tar_img.shape, cr1.shape, cr1.max(), cr1.min())
    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    # print(tar_img.shape, prd_img.shape)
    SSIM = compute_ssim(tar_img, prd_img, cr1)

    return [PSNR, SSIM, PSNR_, SSIM_]

# datasets = ['RealBlur_R', 'RealBlur_J']
datasets = ['FoundIR']
# datasets = ['RealBlur_R']
# datasets = ['RealBlur_J']
# compare_methods = ['MIMOUNet', 'DeepRFT', 'NAFNet64', 'LoFormer', 'PromptIR', 'MoCEIR', 'BioIR', 'EVSSM']
compare_methods = ['UniDeblur-B-kerlearnings123', 'NAFNet64']
for method in compare_methods:
    print(method)
    for dataset in datasets:

        file_path = os.path.join('/home/ubuntu/150t/personal_data/mxt/MXT/ckpts/uniDeblur', method, 'visualization', dataset)
        # file_path = os.path.join('/data/mxt_data/uniDeblur_dataset/test/MotionBlur', dataset, 'blur')
        # print(file_path)
        # gt_path = os.path.join('/data/mxt_data/uniDeblur_dataset/test/LowLightMotionBlur', dataset, 'sharp')
        gt_path = os.path.join('/data/mxt_data/uniDeblur_dataset/test/MotionBlur', dataset, 'sharp')

        path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
        gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
        # print(len(path_list), len(gt_list))
        assert len(path_list) != 0, "Predicted files not found"
        assert len(gt_list) != 0, "Target files not found"

        psnr, ssim, psnr_, ssim_ = [], [], [], []
        img_files =[(i, j) for i,j in zip(gt_list,path_list)]
        # try:
        # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # for filename in img_files:
        #     # print(filename)
        #     PSNR_SSIM = proc(filename)
        #     psnr.append(PSNR_SSIM[0])
        #     ssim.append(PSNR_SSIM[1])
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
                # print(img_files)
                psnr.append(PSNR_SSIM[0])
                ssim.append(PSNR_SSIM[1])
                psnr_.append(PSNR_SSIM[2])
                ssim_.append(PSNR_SSIM[3])
        # except:
        #     print('x')
        #     psnr.append(0.)
        #     ssim.append(0.)

        avg_psnr = sum(psnr)/len(psnr)
        avg_ssim = sum(ssim)/len(ssim)
        avg_psnr_ = sum(psnr_) / len(psnr)
        avg_ssim_ = sum(ssim_) / len(ssim)
        print('For {:s} dataset PSNR: {:f} SSIM: {:f} PSNR_: {:f} SSIM_: {:f}\n'.format(dataset, avg_psnr, avg_ssim, avg_psnr_, avg_ssim_))
