## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import sys
import utils
sys.path.append("../")
# sys.path.append(".")
import torch.nn as nn


import numpy as np
from basicsr.archs.uniDeblur_arch import uniDeblur_vis as Net
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.archs.utils_deblur import *

import cv2
from tqdm import tqdm
import torch
import kornia
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
from collections import OrderedDict
import time
from os import path as osp
from natsort import natsorted
def ker_topk(ker, k_len=61, kernel_size=65):
    # print(ker.shape)
    ker = torch.mean(ker, dim=1, keepdim=True)

    ker = ker.view(-1, kernel_size * kernel_size)
    _, idx = torch.topk(torch.abs(ker), k=k_len, dim=-1)
    z = torch.zeros_like(ker)
    # print(idx)
    # print(kernels_reblur[:, idx])
    z[:, idx] = 1.
    ker = ker * z
    ker = ker.view(-1, kernel_size, kernel_size)
    # ker = ker.view(-1, kernel_size, kernel_size)
    return ker

def convet_to_freq_torch(x):
    x = torch.fft.fft2(x, dim=[-2, -1])
    x = torch.fft.fftshift(x, dim=[-2, -1])
    x = torch.log(torch.abs(x)+1e-7)
    x = (x - x.min()) / (x.max() - x.min() + 1e-7)
    return x

def convet_to_freq_numpy(x):
    x = np.fft.fft2(x, axes=[0, 1])
    x = np.fft.fftshift(x, axes=[0, 1])
    x = np.log(np.abs(x)+1e-7)
    x = (x - x.min()) / (x.max() - x.min() + 1e-7)
    return x

def fft_relu(inp, result_dir, file_, ext_, cvt2torch=True):
    if cvt2torch:
        inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)
        # inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0)
    inp = torch.mean(inp, dim=1, keepdim=True)
    x = torch.fft.fft2(inp)
    # otf_gt_reblur_ = x / otf_clamp(y)
    # ker_reblur_fft = torch.log(torch.fft.fftshift(torch.abs(otf_gt_reblur_) + 1e-7, dim=[-2, -1]))
    # ker_reblur_fft = torch.angle(torch.fft.ifftshift(otf_gt_reblur_, dim=[-2, -1])) / torch.pi
    x_ori = x
    x.real = torch.relu(x.real)
    x.imag = torch.relu(x.imag)
    ker_relu_fft = torch.log(torch.fft.fftshift(torch.abs(x - x_ori / 2.) + 1e-7, dim=[-2, -1]))
    # ker_relu_fft = torch.angle(torch.fft.fftshift(x-x_ori/2., dim=[-2, -1])) / torch.pi
    x_mean = kornia.enhance.normalize_min_max(ker_relu_fft, 0., 1.)
    x_mean = x_mean.squeeze(1)
    x_mean = x_mean.cpu().detach().squeeze(0).numpy()
    out_way = os.path.join(result_dir,
                           os.path.splitext(os.path.split(file_)[-1])[0]+ext_ + '_fftrelu_fft.png')
    sns_plot = plt.figure()
    sns.heatmap(x_mean, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
    plt.close()

    x = torch.fft.ifft2(x).abs() - inp / 2.
    # x = torch.log(torch.clamp(x, 1e-6))
    psf = torch.fft.ifftshift(x, dim=[-2, -1])

    psf = torch.nn.functional.pad(psf, (0, 1, 0, 1))
    kernel = kornia.geometry.center_crop(psf, [65, 65])
    x_mean = kornia.enhance.normalize_min_max(kernel, 0., 1.)
    x_mean = x_mean.squeeze(1)
    x_mean = x_mean.cpu().detach().squeeze(0).numpy()
    out_way = os.path.join(result_dir,
                           os.path.splitext(os.path.split(file_)[-1])[0] +ext_+ '_fftrelu' + '.png')
    sns_plot = plt.figure()

    sns.heatmap(x_mean, cmap='Greys_r', linewidths=0., vmin=0., vmax=1.,
                xticklabels=False, yticklabels=False, cbar=False,
                square=True)  # Reds_r .invert_yaxis()
    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
    plt.close()
    return # x_mean, ker_relu_fft

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
def paired_paths_from_folder(folders, keys, filename_tmpl, multi=False):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
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

    # if multi:
    input_paths = natsorted(os.listdir(input_folder))
    # else:
    #     input_paths = list(scandir(input_folder))
    #     input_paths = list(scandir(input_folder))
    # gt_paths = list(scandir(gt_folder))
    gt_paths = natsorted(os.listdir(gt_folder))

    assert len(input_paths) == len(gt_paths), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_paths)}, {len(gt_paths)}."
    )
    paths = []
    input_paths_set = set(input_paths)
    for gt_path in gt_paths:

        basename, ext = osp.splitext(osp.basename(gt_path))

        # print(gt_path, filename_tmpl, f"{filename_tmpl.format(basename)}{ext}")
        if multi:
            input_name = f"{filename_tmpl.format(basename)}"
        else:
            input_name = f"{filename_tmpl.format(basename)}{ext}"
        if "_HR_" in input_name:
            input_name = input_name.replace("_HR_", "_LRBI_")
        input_path = osp.join(input_folder, input_name)
        assert (
                len(set([input_name]) & input_paths_set) == 1
        ), f"{input_name} is not in {input_key}_paths."
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
        )
    return paths
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper
def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, "r") as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


psnr_val_rgb = []
ssim_val_rgb = []
ssim = 0
psnr = 0
factor = 8
all_time = 0.
crop_size = 256
get_fft_relu = True
get_psnr = True
save_result = True
# files = [os.path.join(inp_dir, x+'.png') for x in img_select]
img_end = 5

opt_path = '/data/mxt_code/AllinOne/options/all_in_one/test/test_uniDeblurB.yml'
result_root = '/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/uniDeblur/features_ker_uniDeblur-B'
# dataset_name = 'MotionBlur'
os.makedirs(result_root,exist_ok=True)
opt = yaml_load(opt_path)
strict = True
save_lens = 8
# opt = deepcopy(opt)
# model_restoration = MODEL_REGISTRY.get(opt["model_type"])(opt)
if opt['path']['pretrain_network_g'] is not None:

    s = opt['network_g'].pop('type')
    ##########################
    # print(x['network_g'])
    model_restoration = Net(**opt['network_g'])
    checkpoint = torch.load(opt['path']['pretrain_network_g'])
    # model_restoration.load_state_dict(checkpoint['params'])
    try:
        model_restoration.load_state_dict(checkpoint['params_ema'], strict=strict)
        state_dict = checkpoint["params_ema"]
        # for k, v in state_dict.items():
        #     print(k)
    except:
        try:
            model_restoration.load_state_dict(checkpoint["state_dict"], strict=strict)
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # print(k)
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_restoration.load_state_dict(new_state_dict, strict=strict)
else:
    print('inference for time')
print("===>Testing using weights: ", opt['path']['pretrain_network_g'])
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
paths_all = []
for phase, dataset_opt in opt["datasets"].items():
    # if phase.split("_")[0] == "train":
    if phase.split("_")[0] == "test":
        lq_folder = dataset_opt['dataroot_lq']
        gt_folder = dataset_opt['dataroot_gt']
        meta_file = dataset_opt.get("meta_info_file", None)
        # print(lq_path, gt_path)
        # input_paths = os.listdir(lq_path)
        if meta_file is not None:
            paths = paired_paths_from_meta_info_file2(
                [lq_folder, gt_folder],
                ["lq", "gt"],
                 meta_file,
                "{}",
            )
        else:
            multi = opt["multi"] if "multi" in opt else False
            paths = paired_paths_from_folder(
                [lq_folder, gt_folder],
                ["lq", "gt"],
                "{}",
                multi=False,
            )
        # step = dataset_opt.get("step", None)
        # if step is not None:
        #     paths = paths[::step]

        random.shuffle(paths)
        color_fft = 'Blues_r'
        color_feature = 'Reds'
        color_ker = 'Greys_r'
        dataset_name = dataset_opt['name']
        result_dir_1 = os.path.join(result_root, dataset_name)
        os.makedirs(result_dir_1, exist_ok=True)
        # color_fft = 'Blues_r'
        # color_feature = 'Reds'
        # color_ker = 'Greys_r'


        with torch.no_grad():
            kernel_pre = 0
            for path in tqdm(paths[:img_end]): # img_end [:1]
            # for file_ in files[:]:
                file_ = path["lq_path"]
                result_dir = os.path.join(result_dir_1, os.path.splitext(os.path.basename(file_))[0])
                os.makedirs(result_dir, exist_ok=True)

                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                img = np.float32(utils.load_img(path["lq_path"]))/255.
                img = torch.from_numpy(img).permute(2,0,1)
                input_ = img.unsqueeze(0).cuda()
                # print(input_.shape, input_.dtype)
                input_ = kornia.geometry.center_crop(input_, [crop_size, crop_size])
                # Padding in case images are not multiples of 8
                # h,w = input_.shape[2], input_.shape[3]
                # H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                # padh = H-h if h%factor!=0 else 0
                # padw = W-w if w%factor!=0 else 0
                # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
                gt_file_way = path["gt_path"]
                rgb_gt = np.float32(utils.load_img(gt_file_way)) / 255.
                rgb_gt = torch.from_numpy(rgb_gt).permute(2, 0, 1)
                rgb_gt = rgb_gt.unsqueeze(0).cuda()
                rgb_gt = kornia.geometry.center_crop(rgb_gt, [crop_size, crop_size])
                torch.cuda.synchronize()
                start = time.time()
                restored = model_restoration(input_, rgb_gt)
                torch.cuda.synchronize()
                end = time.time()
                all_time += end - start
                blur_fft = torch.fft.rfft2(input_)
                sharp_fft = torch.fft.rfft2(rgb_gt)
                otf_gt_deblur = sharp_fft / otf_clamp(blur_fft, eps=1e-2)
                otf_gt_reblur = blur_fft / otf_clamp(sharp_fft, eps=1e-2)

                blur_fft2 = torch.fft.fft2(input_)
                sharp_fft2 = torch.fft.fft2(rgb_gt)
                otf_gt_deblur_ = sharp_fft2 / otf_clamp(blur_fft2)
                otf_gt_reblur_ = blur_fft2 / otf_clamp(sharp_fft2)
                ker_deblur_fft = torch.log(
                    torch.fft.fftshift(torch.abs(otf_gt_deblur_) + 1e-7, dim=[-2, -1]))
                ker_reblur_fft = torch.log(
                    torch.fft.fftshift(torch.abs(otf_gt_reblur_) + 1e-7, dim=[-2, -1]))
                ker_deblur = convert_otf2psf(otf_gt_deblur, [65, 65])
                ker_reblur = convert_otf2psf(otf_gt_reblur, [65, 65])
                # ker_deblur = ker_topk(ker_deblur, 61, 65)
                # ker_reblur = ker_topk(ker_reblur, 61, 65)
                if isinstance(restored, list):
                    restoredx = restored[-1]
                elif isinstance(restored, dict):
                    # print(restored.keys())
                    # if 'kernel' in restored.keys():
                    #     kernel = restored['kernel'][1]
                        # print(torch.sum(torch.abs(kernel-kernel_pre)))
                        # kernel_pre = kernel
                    # if 'mask' in restored.keys():
                    #     mask = restored['mask']
                    # mask = kornia.enhance.normalize_min_max(mask, 0., 1.)
                    # restored_deconv = restored['deconv_img']
                    # restored_reblur = restored['reblur_img']
                    # kernel = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                    # restored = restored['img']  # ['x_deblur']
                    # restoredx = restored['img']
                    if isinstance(restored['restored'], list):
                        restoredx = restored['restored'][-1]
                    else:
                        restoredx = restored['restored']

                # Unpad images to original dimensions

                # restored = restored[:,:,:h,:w]

                restoredx = torch.clamp(restoredx,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                # restored_deconvx = restored_deconv.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                input_img = input_.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                # restored_reblurx = restored_reblur.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                restored_img = img_as_ubyte(restoredx)
                # restored_deconv_img = img_as_ubyte(restored_deconvx)
                # restored_reblur_img = img_as_ubyte(restored_reblurx)
                if get_psnr:
                    gt_file_way = path["gt_path"]
                    if os.path.exists(gt_file_way):
                        rgb_gt = cv2.imread(gt_file_way)
                    else:
                        rgb_gt = cv2.imread(gt_file_way[:-3] + 'png')
                    rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)
                    rgb_gt = torch.from_numpy(rgb_gt / 255.).permute(2, 0, 1)
                    rgb_gt = rgb_gt.unsqueeze(0)
                    rgb_gt = kornia.geometry.center_crop(rgb_gt, [crop_size, crop_size])
                    res_gt = rgb_gt - input_.cpu()
                    rgb_gt = rgb_gt.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    rgb_gt = img_as_ubyte(rgb_gt)
                    if get_psnr:
                        PSNR = psnr_loss(restored_img, rgb_gt)
                        # PSNR2 = psnr_loss(restored_deconv_img, rgb_gt)
                        psnr_val_rgb.append(PSNR)
                        # ssim_val_rgb.append(PSNR2)
                    # if args.get_ssim:
                    #     SSIM = calculate_ssim(restored_img, rgb_gt, crop_border=0)

                    # print('PSNR: ', sum(psnr_val_rgb) / len(psnr_val_rgb))
                if save_result:
                    if isinstance(restored['restored'], list):
                        for i, restoredi in enumerate(restored['restored']):
                            res = restoredi - input_
                            res = kornia.enhance.normalize_min_max(res, 0., 1.)
                            res = res.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                            res = np.mean(res, axis=-1)
                            restoredi = torch.clamp(restoredi, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                            utils.save_img((os.path.join(result_dir, 'restored_' + str(i) + '.png')),
                                           img_as_ubyte(restoredi))

                            out_way = os.path.join(result_dir, 'res_' + str(i) + '.png')
                            sns_plot = plt.figure()
                            sns.heatmap(res, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                    res_gt = kornia.enhance.normalize_min_max(res_gt, 0., 1.)
                    res_gt = res_gt.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    res_gt = np.mean(res_gt, axis=-1)
                    out_way = os.path.join(result_dir, 'res_gt.png')
                    sns_plot = plt.figure()
                    sns.heatmap(res_gt, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                    plt.close()

                    x_f = convet_to_freq_numpy(res_gt)
                    out_way = os.path.join(result_dir, 'res_gt_fft.png')
                    sns_plot = plt.figure()
                    sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                    plt.close()
                    # else:
                    #     utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                    #                    img_as_ubyte(restoredx))
                    if True:
                        kernel = ker_deblur # convert_otf2psf(otf_gt_deblur, [65, 65]) # restored['ker_deblur']
                        x_mean = torch.mean(kernel, dim=1)
                        # x_mean = torch.clamp(x_mean, -1., 1.)
                        # print(kernel.max(), kernel.min())
                        # x_mean = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                        # x_mean = x_mean.squeeze(1)
                        # x_mean = torch.mean(kernel, dim=1)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                        out_way = os.path.join(result_dir, 'ker_deblur' + '.png')
                        sns_plot = plt.figure()

                        sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=-0.25, vmax=0.25,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()

                        x_mean = kornia.enhance.normalize_min_max(ker_deblur_fft, 0., 1.)
                        # x_mean = x_mean.squeeze(1)
                        x_mean = torch.mean(x_mean, dim=1)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                        out_way = os.path.join(result_dir, 'ker_deblur_fft.png')
                        sns_plot = plt.figure()
                        sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()
                    if True:
                        kernel = ker_reblur # convert_otf2psf(otf_gt_reblur, [65, 65])
                        x_mean = torch.mean(kernel, dim=1)
                        # print(kernel.max(), kernel.min())
                        # x_mean = torch.clamp(x_mean, 0., 1.)
                        # x_mean = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                        # x_mean = x_mean.squeeze(1)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                        out_way = os.path.join(result_dir, 'ker_reblur' + '.png')
                        sns_plot = plt.figure()

                        sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=0.25,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()


                        x_mean = kornia.enhance.normalize_min_max(ker_reblur_fft, 0., 1.)
                        x_mean = torch.mean(x_mean, dim=1)
                        # x_mean = x_mean.squeeze(1)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                        out_way = os.path.join(result_dir,'ker_reblur_fft.png')
                        sns_plot = plt.figure()
                        sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()
                # if 'feature_visuals' in restored.keys():
                #     feature_visuals = restored['feature_visuals']

                    if 'kernel' in restored.keys():
                        result_dir_ = os.path.join(result_dir, 'kernel')
                        os.makedirs(result_dir_, exist_ok=True)
                        for i, kernel in enumerate(restored['kernel']):
                            # kernel = feature_visual['kernel']
                            # print(kernel.shape)
                            # x_mean = ker_topk(kernel)
                            x_mean = torch.mean(kernel, dim=1)

                            # x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                            x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                            # print(x_mean.shape)
                            out_way = os.path.join(result_dir_, 'mean_kernel_' + str(i) + '.png')
                            sns_plot = plt.figure()

                            sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=-0.25, vmax=0.25,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()

                            # kernel = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                            kernelx = kernel.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                            for k in range(save_lens):
                                out_way = os.path.join(result_dir_, 'kernel_' + str(i)+'_' + str(k) + '.png')
                                sns_plot = plt.figure()
                                # print(kernelx[:, :, k].max())
                                ker = kernelx[:, :, k]
                                # print(ker.min(), ker.max())
                                # ker = (ker-ker.min()) / (ker.max()-ker.min()+1e-7)
                                sns.heatmap(ker, cmap=color_ker, linewidths=0., vmin=-0.25, vmax=0.25,
                                            xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                            if 'kernel_fft' in restored.keys():
                                kernel = restored['kernel_fft'][i]
                                # print(kernel.shape)
                                # x_mean = ker_topk(kernel)
                                x_mean = torch.mean(kernel, dim=1)
                                x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                                x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                                # print(x_mean.shape)
                                out_way = os.path.join(result_dir_, 'mean_kernel_fft' + '.png')
                                sns_plot = plt.figure()

                                sns.heatmap(x_mean, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()

                                # kernel = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                                kernelx = kernel.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                                for k in range(save_lens):
                                    out_way = os.path.join(result_dir_, 'kernel_fft_' + str(i) +'_' + str(k) + '.png')
                                    sns_plot = plt.figure()
                                    # print(kernelx[:, :, k].max())
                                    ker = kernelx[:, :, k]
                                    ker = (ker-ker.min()) / (ker.max()-ker.min()+1e-7)
                                    sns.heatmap(ker, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                                    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                    plt.close()
                    if 'x_in' in restored.keys():
                        result_dir_ = os.path.join(result_dir, 'x_in')
                        os.makedirs(result_dir_, exist_ok=True)
                        for i, feature in enumerate(restored['x_in']):
                        # if 'feature' in feature_visual.keys():
                        #     feature = feature_visual['feature']
                            x_mean = torch.mean(feature, dim=1)
                            x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                            x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                            out_way = os.path.join(result_dir_,'mean_' + str(i) + '.png')
                            sns_plot = plt.figure()

                            sns.heatmap(x_mean, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()

                            x_f = convet_to_freq_numpy(x_mean)
                            out_way = os.path.join(result_dir_, 'mean_fft_' + str(i)+'.png')
                            sns_plot = plt.figure()
                            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            feature_ori = feature
                            # feature = kornia.enhance.normalize_min_max(feature, 0., 1.)
                            xs = feature.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                            xs_ori = xs
                            for k in range(save_lens):
                                out_way = os.path.join(result_dir_, str(i) +'_' + str(k) + '.png')
                                sns_plot = plt.figure()
                                # print(kernelx[:, :, k].max())
                                x = xs[:, :, k]

                                if get_fft_relu:
                                    fft_relu(x, result_dir_, file_, ext_=str(i)+'_' + str(k), cvt2torch=True)
                                x = (x - x.min()) / (x.max() - x.min() + 1e-7)
                                sns.heatmap(x, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                                x_f = convet_to_freq_numpy(xs[:, :, k])
                                out_way = os.path.join(result_dir_, 'fft_' + str(i) +'_' + str(k)+'.png')
                                sns_plot = plt.figure()
                                sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                    if 'x_out' in restored.keys():
                        result_dir_ = os.path.join(result_dir, 'x_out')
                        os.makedirs(result_dir_, exist_ok=True)
                        for i, feature in enumerate(restored['x_out']):
                            # feature = feature_visual['deconv_feature'] # + feature_visual['feature']) * 0.5
                            x_mean = torch.mean(feature, dim=1)
                            x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                            x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                            out_way = os.path.join(result_dir_,'mean_' + str(i) + '.png')
                            sns_plot = plt.figure()

                            sns.heatmap(x_mean, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            x_f = convet_to_freq_numpy(x_mean)
                            out_way = os.path.join(result_dir_,'mean_fft_' + str(i)+'.png')
                            sns_plot = plt.figure()
                            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            # feature = feature - feature_ori
                            # feature = kornia.enhance.normalize_min_max(feature, 0., 1.)
                            xs = feature.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                            for k in range(save_lens):
                                out_way = os.path.join(result_dir_,str(i) +'_' + str(k) + '.png')
                                sns_plot = plt.figure()
                                # print(kernelx[:, :, k].max())
                                x = xs[:, :, k]
                                x = (x - x.min()) / (x.max() - x.min() + 1e-7)
                                sns.heatmap(x, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                                x_f = convet_to_freq_numpy(xs[:, :, k])
                                out_way = os.path.join(result_dir_,'fft_' + str(i) +'_' + str(k)+'.png')
                                sns_plot = plt.figure()
                                sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()

                    if 'x_dec' in restored.keys():
                        result_dir_ = os.path.join(result_dir, 'x_dec')
                        os.makedirs(result_dir_, exist_ok=True)
                        for i, feature in enumerate(restored['x_dec']):
                            # feature = feature_visual['deconv_feature'] # + feature_visual['feature']) * 0.5
                            x_mean = torch.mean(feature, dim=1)
                            x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                            x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                            out_way = os.path.join(result_dir_, 'mean_' + str(i) + '.png')
                            sns_plot = plt.figure()

                            sns.heatmap(x_mean, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            x_f = convet_to_freq_numpy(x_mean)
                            out_way = os.path.join(result_dir_, 'mean_fft_' + str(i)+'.png')
                            sns_plot = plt.figure()
                            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            # feature = feature - feature_ori
                            # feature = kornia.enhance.normalize_min_max(feature, 0., 1.)
                            xs = feature.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                            for k in range(save_lens):
                                out_way = os.path.join(result_dir_,str(i) +'_' + str(k) + '.png')
                                sns_plot = plt.figure()
                                # print(kernelx[:, :, k].max())
                                x = xs[:, :, k]
                                x = (x - x.min()) / (x.max() - x.min() + 1e-7)
                                sns.heatmap(x, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                                x_f = convet_to_freq_numpy(xs[:, :, k])
                                out_way = os.path.join(result_dir_,'_fft_' + str(i) +'_' + str(k)+'.png')
                                sns_plot = plt.figure()
                                sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()

                    if 'x_enc' in restored.keys():
                        result_dir_ = os.path.join(result_dir, 'x_enc')
                        os.makedirs(result_dir_, exist_ok=True)
                        for i, feature in enumerate(restored['x_dec']):
                            # feature = feature_visual['deconv_feature'] # + feature_visual['feature']) * 0.5
                            x_mean = torch.mean(feature, dim=1)
                            x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                            x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                            out_way = os.path.join(result_dir_, 'mean_' + str(i) + '.png')
                            sns_plot = plt.figure()

                            sns.heatmap(x_mean, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            x_f = convet_to_freq_numpy(x_mean)
                            out_way = os.path.join(result_dir_, 'mean_fft_' + str(i)+'.png')
                            sns_plot = plt.figure()
                            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            # feature = feature - feature_ori
                            # feature = kornia.enhance.normalize_min_max(feature, 0., 1.)
                            xs = feature.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                            for k in range(save_lens):
                                out_way = os.path.join(result_dir_,str(i) +'_' + str(k) + '.png')
                                sns_plot = plt.figure()
                                # print(kernelx[:, :, k].max())
                                x = xs[:, :, k]
                                x = (x - x.min()) / (x.max() - x.min() + 1e-7)
                                sns.heatmap(x, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                                x_f = convet_to_freq_numpy(xs[:, :, k])
                                out_way = os.path.join(result_dir_,'_fft_' + str(i) +'_' + str(k)+'.png')
                                sns_plot = plt.figure()
                                sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                            xticklabels=False, yticklabels=False, cbar=False,
                                            square=True)  # Reds .invert_yaxis()
                                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                                plt.close()
                    # if 'mask' in restored.keys():
                    #     maskx = mask.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    #     for k in range(maskx.shape[-1]):
                    #         out_way = os.path.join(result_dir,
                    #                                os.path.splitext(os.path.split(file_)[-1])[0] + '_mask_' + str(k) + '.png')
                    #         sns_plot = plt.figure()
                    #         # print(maskx[:, :, k].max())
                    #         sns.heatmap(maskx[:, :, k], cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                    #                     xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                    #         sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                    #         plt.close()
                #     img_ = kornia.tensor_to_image(out_weiner)
                    #
                    #     cv2.imwrite((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_deblur_xweiner' + str(k) + '.png')),
                    #                    np.uint8(img_[:, :, ::-1]*255.))
                    # residual = rgb_gt - input_img
                    # utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'_blur_pattern.png')),
                    #                img_as_ubyte(residual))
                    utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + 'gt.png')),
                                   rgb_gt)
                    x_f = convet_to_freq_numpy(np.mean(rgb_gt, axis=-1))
                    out_way = os.path.join(result_dir,
                                           os.path.splitext(os.path.split(file_)[-1])[0] + '_rgb_gt_fft.png')
                    sns_plot = plt.figure()
                    sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                    plt.close()
                    utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_blur.png')),
                                   img_as_ubyte(input_img))
                    x_f = convet_to_freq_numpy(np.mean(input_img, axis=-1))
                    out_way = os.path.join(result_dir,
                                           os.path.splitext(os.path.split(file_)[-1])[0] + '_blur_fft.png')
                    sns_plot = plt.figure()
                    sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                    plt.close()
                    # utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_blure.png')),
                    #                img_as_ubyte(restored_reblur_img))
        print('average_time: ', all_time / img_end)
        if get_psnr:
            psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
            print("PSNR: %f" % psnr)
            # ssim = sum(ssim_val_rgb) / len(ssim_val_rgb)
            # print("PSNR2: %f" % ssim)

