import os.path

from torch.utils import data as data
import glob
import torch
import torch.nn.functional as F
import sys
import cv2
sys.path.append('/data/mxt_code/AllinOne')
import numpy as np
import math
import random
from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory
# from basicsr.utils.img_process_util import filter2D
# from motion_blur.blur_image import *
# from .data_util import prctile_norm
import os
import yaml
from collections import OrderedDict
import time
from os import path as osp
from natsort import natsorted
def paired_paths_from_meta_info_file2(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    # lq_names = []
    # gt_names = []

    paths = []
    with open(meta_info_file, "r") as fin:
        for line in fin:
            elements = line.strip().split()
            if len(elements) == 2:
                # lq_names.append(elements[0])
                # gt_names.append(elements[1])
                input_path = osp.join(input_folder, elements[0])
                gt_path = osp.join(gt_folder, elements[1])
                paths.append(
                    dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
                )
        # lq_names = [line.strip().split(" ")[0] for line in fin]
        # gt_names = [line.strip().split(" ")[1] for line in fin]

    # paths = []
    # for lq_name, gt_name in zip(lq_names, gt_names):
    #     # basename, ext = osp.splitext(osp.basename(gt_name))
    #     # input_name = f"{filename_tmpl.format(basename)}{ext}"
    #     input_path = osp.join(input_folder, lq_name)
    #     gt_path = osp.join(gt_folder, gt_name)
    #     paths.append(
    #         dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
    #     )
    return paths
def filter2D(img, kernel, pad_size=None):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if pad_size is None:
        if k % 2 == 1:
            img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
        else:
            raise ValueError("Wrong kernel size")
    else:
        if pad_size>0:
            img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode="reflect")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)

def image_degrade(img_gt):
    # if len(img_gt.shape) == 2:
    #     c = 1
    #     img_gt = torch.from_numpy(img_gt).unsqueeze(0).float().contiguous() / 255.
    # else:
    #     c = 3
    #     img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous() / 255.
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]

    trajectory = Trajectory(canvas=65, max_len=60, expl=np.random.choice(params)).fit()
    psfs = PSF(canvas=65, trajectory=trajectory).fit()
    # part = np.random.choice([1, 2, 3])
    psf = psfs[-1]
    # print(psf.shape)
    # BGR to RGB, HWC to CHW, numpy to tensor
    # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
    psf[np.where(psf <= 1e-5)] = 0.
    # if np.random.uniform(0., 1.) < 0.5:
    #     idx = np.where(psf > 1e-5)
    #     psf[idx] = np.exp(psf[idx])
    #
    # else:
    psf = np.abs(psf)
    psf = psf / np.sum(psf)
    psf = torch.FloatTensor(psf)
    # print(psf.shape)
    img_lq = filter2D(img_gt.cuda().unsqueeze(0), psf.cuda()).squeeze(0)
    img_lq = torch.clamp(img_lq, 0., 1.)
    img_lq = torch.clamp((img_lq.cpu() * 255.0).round(), 0, 255)
    # img_lq = degradation(img_gt.unsqueeze(0), kernel, kernel2, sinc_kernel, jpeger, usm_sharpener, second_blur, opt).squeeze(0)

    return img_lq, psf
if __name__ == '__main__':
    import cv2
    import torch
    import numpy as np
    import math
    import random
    import tqdm
    import time

    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21

    root = '/home/ubuntu/uniDeblurDataset/LFDOF/train_data'
    # root = '/data/mxt_data/uniDeblur_dataset/test/DefocusBlur/RealDOF'
    img_dir = os.path.join(root, 'input')
    sharp_dir = os.path.join(root, 'ground_truth')
    meta_file = '/home/ubuntu/uniDeblurDataset/LFDOF/train_data/meta_info_train.txt'
    root_out = '/data/mxt_data/LFDOF/train_data'
    out_dir = os.path.join(root_out, 'motion_blur')
    os.makedirs(out_dir, exist_ok=True)
    out_dir2 = os.path.join(root_out, 'motion_sharp')
    os.makedirs(out_dir2, exist_ok=True)
    out_dir3 = os.path.join(root_out, 'motion_ker')
    os.makedirs(out_dir3, exist_ok=True)
    # img_dir = '/data/mxt_data/SR/DF2K/HR'
    # out_dir = '/data/mxt_data/SR/DF2K/input_cropsX'
    # os.makedirs(out_dir, exist_ok=True)
    # out_dir2 = '/data/mxt_data/SR/DF2K/target_cropsX'
    # os.makedirs(out_dir2, exist_ok=True)
    # img_file_list = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
    paths = paired_paths_from_meta_info_file2(
        [img_dir, sharp_dir],
        ["lq", "gt"],
        meta_file,
        "{}",
    )
    cut_patches = True
    pad_size = 32
    patch_size = 624 + 2 * pad_size
    # overlap = 256 # train
    overlap = 200 # test
    p_max = 0
    for path in tqdm.tqdm(paths):
        # img_file = '/data/mxt_data/SR/SR/img012.png'
        img_file = path["lq_path"]
        sharp_file = path["gt_path"]
        img_file_name = os.path.basename(img_file)

        # if os.path.exists(out_img_file):
        #     continue
        img_gt = cv2.imread(img_file)
        img_gt_shape = img_gt.shape
        if len(img_gt.shape) == 2:
            c = 1
            img_gt = torch.from_numpy(img_gt).unsqueeze(0).float().contiguous()
        else:
            c = 3
            img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_gt_ = cv2.imread(sharp_file)
        if not cut_patches:
            if img_gt_shape[0] < 512 or img_gt_shape[1] < 512:
                # os.system('rm {}'.format(img_file))
                # print('rm {}'.format(img_file))
                # time.sleep(0.001)
                continue
            out_img_file = os.path.join(out_dir, img_file_name)
            if os.path.exists(out_img_file):
                continue
            img_lq, psf = image_degrade(img_gt / 255.)
            if c == 1:
                img_lq = img_lq.squeeze(0).cpu().detach().numpy()
            else:
                img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
            if img_lq.shape != img_gt_shape:
                print(img_lq.shape, img_gt_shape)

            out_gt_file = os.path.join(out_dir2, img_file_name)
            out_ker_file = os.path.join(out_dir3, img_file_name[:-4]+'.npy')
            # print(out_img_file, out_ker_file, out_gt_file)
            cv2.imwrite(out_gt_file, img_gt_[pad_size:-pad_size, pad_size:-pad_size, :].astype('uint8'))
            cv2.imwrite(out_img_file, img_lq[pad_size:-pad_size, pad_size:-pad_size, :].astype('uint8'))
            # cv2.imwrite(out_ker_file, (psf.cpu().detach().numpy()*255.).astype('uint8'))
            np.save(out_ker_file, (psf.cpu().detach().numpy()))
        else:
            filename = os.path.splitext(os.path.split(img_file)[-1])[0]


            num_patch = 0
            # print(os.path.join(out_dir, filename + '-' + str(num_patch) + '.png'))
            if os.path.exists(os.path.join(out_dir, filename + '-' + str(num_patch+1) + '.png')):
                print(filename+' exist')
                continue
            w, h = img_gt.shape[-2:]
            if w < patch_size or h < patch_size:
                continue
            if w > p_max and h > p_max:
                w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=np.int32))
                h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=np.int32))
                w1.append(w - patch_size)
                h1.append(h - patch_size)
                # print(h1, w1)
                for i in w1:
                    for j in h1:
                        num_patch += 1

                        # lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                        hr_patch = img_gt[:, i:i + patch_size, j:j + patch_size]
                        if hr_patch.shape[-2] != patch_size or hr_patch.shape[-1] != patch_size:
                            print(hr_patch.shape, filename)
                            continue
                        img_lq, psf = image_degrade(hr_patch / 255.)
                        img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
                        img_lq = img_lq[pad_size: -pad_size, pad_size: -pad_size, :]
                        gt_patch_ = img_gt_[i:i + patch_size, j:j + patch_size, :]
                        gt_patch = gt_patch_[pad_size:-pad_size, pad_size:-pad_size, :]
                        # if c == 1:
                        #     img_lq = img_lq.squeeze(0).cpu().detach().numpy()
                        #     gt_patch = gt_patch.squeeze(0).cpu().detach().numpy()
                        # else:
                        #     img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
                        #     gt_patch = gt_patch.cpu().detach().permute(1, 2, 0).numpy()
                        lr_savename = os.path.join(out_dir, filename + '-' + str(num_patch) + '.png')
                        hr_savename = os.path.join(out_dir2, filename + '-' + str(num_patch) + '.png')
                        # print(lr_savename)
                        cv2.imwrite(lr_savename, img_lq.astype('uint8'))
                        cv2.imwrite(hr_savename, gt_patch.astype('uint8'))

                        out_ker_file = os.path.join(out_dir3, filename + '-' + str(num_patch) + '.npy')
                        np.save(out_ker_file, (psf.cpu().detach().numpy()))

            # else:
            #     img_lq = image_degrade(img_gt / 255.)
            #     if c == 1:
            #         img_lq = img_lq.squeeze(0).cpu().detach().numpy()
            #         img_gt = img_gt.squeeze(0).cpu().detach().numpy()
            #     else:
            #         img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
            #         img_gt = img_gt.cpu().detach().permute(1, 2, 0).numpy()
            #     lr_savename = os.path.join(out_dir, filename + '.png')
            #     hr_savename = os.path.join(out_dir2, filename + '.png')
            #
            #     cv2.imwrite(lr_savename, img_lq.astype('uint8'))
            #     cv2.imwrite(hr_savename, img_gt.astype('uint8'))
        # # TODO: kernel range is now hard-coded, should be in the configure file
        # pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        # pulse_tensor[10, 10] = 1
        # kernel_size = random.choice(kernel_range)
        # if np.random.uniform() < opt['sinc_prob']:
        #     # this sinc filter setting is for kernels ranging from [7, 21]
        #     if kernel_size < 13:
        #         omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     else:
        #         omega_c = np.random.uniform(np.pi / 5, np.pi)
        #     kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        # else:
        #     kernel = random_mixed_kernels(
        #         opt['kernel_list'],
        #         opt['kernel_prob'],
        #         kernel_size,
        #         opt['blur_sigma'],
        #         opt['blur_sigma'], [-math.pi, math.pi],
        #         opt['betag_range'],
        #         opt['betap_range'],
        #         noise_range=None)
        # # pad kernel
        # pad_size = (21 - kernel_size) // 2
        # kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        #
        # # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        # kernel_size = random.choice(kernel_range)
        # if np.random.uniform() < opt['sinc_prob2']:
        #     if kernel_size < 13:
        #         omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     else:
        #         omega_c = np.random.uniform(np.pi / 5, np.pi)
        #     kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        # else:
        #     kernel2 = random_mixed_kernels(
        #         opt['kernel_list2'],
        #         opt['kernel_prob2'],
        #         kernel_size,
        #         opt['blur_sigma2'],
        #         opt['blur_sigma2'], [-math.pi, math.pi],
        #         opt['betag_range2'],
        #         opt['betap_range2'],
        #         noise_range=None)
        #
        # # pad kernel
        # pad_size = (21 - kernel_size) // 2
        # kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        #
        # # ------------------------------------- the final sinc kernel ------------------------------------- #
        # if np.random.uniform() < opt['final_sinc_prob']:
        #     kernel_size = random.choice(kernel_range)
        #     omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        #     sinc_kernel = torch.FloatTensor(sinc_kernel)
        # else:
        #     sinc_kernel = pulse_tensor
        #
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        # kernel = torch.FloatTensor(kernel)
        # kernel2 = torch.FloatTensor(kernel2)
        # img_lq = degradation(img_gt.unsqueeze(0), kernel, kernel2, sinc_kernel, jpeger, usm_sharpener, opt).squeeze(0)
        # if c == 1:
        #     img_lq = img_lq.squeeze(0).cpu().detach().numpy()
        # else:
        #     img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
        # print(img_lq.max())
        # out_img_file = os.path.join(out_dir, img_file_name)
        # cv2.imwrite(out_img_file, img_lq.astype('uint8'))