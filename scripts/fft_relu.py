

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
# import kornia
def otf_clamp(otf, eps=1e-7):

    otf_mag = torch.abs(otf)
    otf_phase = torch.angle(otf)
    out = (torch.exp(otf_phase * 1j) * torch.clamp(otf_mag, eps))
    return out
def convert_otf2psf(otf, kernel_size=[31, 31], mode='rfft'):
    if mode == 'rfft':
        psf = torch.fft.irfft2(otf)
    else:
        psf = torch.fft.ifft2(otf).real
    psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    psf = torch.nn.functional.pad(psf, (0, 1, 0, 1))
    ker = kornia.geometry.center_crop(psf, kernel_size)
    # ker = kornia.geometry.center_crop(psf[..., 1:, 1:], kernel_size)
    # for axis, axis_size in enumerate(psf.shape[-2:]):
    # psf = torch.roll(psf, int(kernel_size[0] / 2), dims=-2)
    # psf = torch.roll(psf, int(kernel_size[1] / 2), dims=-1)
    # ker = psf[..., :kernel_size[0], :kernel_size[1]]
    return ker
def ker_topk(ker, k_len=61, kernel_size=65):
    ker = torch.mean(ker, dim=1, keepdim=True)

    ker = ker.view(-1, kernel_size * kernel_size)
    _, idx = torch.topk(torch.abs(ker), k=k_len, dim=-1)
    z = torch.zeros_like(ker)
    # print(idx)
    # print(kernels_reblur[:, idx])
    z[:, idx] = 1.
    ker = ker * z
    ker = ker.view(-1, 1, kernel_size, kernel_size)
    # ker = ker.view(-1, kernel_size, kernel_size)
    return ker


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

opt_path = '/data/mxt_code/AllinOne/options/all_in_one/test/test_uniDeblur.yml'
result_dir = '/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/uniDeblur/fftrelu'
# dataset_name = 'MotionBlur'
os.makedirs(result_dir,exist_ok=True)
opt = yaml_load(opt_path)
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
        color_fft = 'Greys_r'
        color_feature = 'Blues_r'
        color_ker = 'Greys_r'
        dataset_name = dataset_opt['name']
        os.makedirs(os.path.join(result_dir, dataset_name), exist_ok=True)
        crop_size = 256
        for path in tqdm(paths[:10]):
            file_ = os.path.basename(path["lq_path"])

            y_ = cv2.imread(path["gt_path"])
            h, w = y_.shape[:2]
            h_start = (h - crop_size) // 2
            h_end = (h + crop_size) // 2
            w_start = (w - crop_size) // 2
            w_end = (w + crop_size) // 2
            y_ = y_[h_start:h_end, w_start:w_end, :]
            y = y_ / 255.
            tar = torch.from_numpy(y).permute(2,0,1).unsqueeze(0)
            # tar = kornia.geometry.center_crop(tar, [crop_size, crop_size])
            tar = torch.mean(tar, dim=1, keepdim=True)
            y = torch.fft.fft2(tar)

            x_ = cv2.imread(path["lq_path"])
            h, w = x_.shape[:2]
            h_start = (h - crop_size) // 2
            h_end = (h + crop_size) // 2
            w_start = (w - crop_size) // 2
            w_end = (w + crop_size) // 2
            x_ = x_[h_start:h_end, w_start:w_end, :]
            x = x_ / 255.

            inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            # print(inp.shape)

            # inp = kornia.geometry.center_crop(inp, [crop_size, crop_size])
            inp = torch.mean(inp, dim=1, keepdim=True)
            x = torch.fft.fft2(inp)
            otf_gt_reblur_ = x / otf_clamp(y)
            ker_reblur_fft = torch.log(torch.fft.fftshift(torch.abs(otf_gt_reblur_) + 1e-7, dim=[-2, -1]))
            # ker_reblur_fft = torch.angle(torch.fft.ifftshift(otf_gt_reblur_, dim=[-2, -1])) / torch.pi
            x_ori = x
            x.real = torch.relu(x.real)
            x.imag = torch.relu(x.imag)
            ker_relu_fft = torch.log(torch.fft.fftshift(torch.abs(x-x_ori/2.) + 1e-7, dim=[-2, -1]))
            # ker_relu_fft = torch.angle(torch.fft.fftshift(x-x_ori/2., dim=[-2, -1])) / torch.pi
            x = torch.fft.ifft2(x).abs() - inp / 2.
            # x = torch.log(torch.clamp(x, 1e-6))
            psf = torch.fft.ifftshift(x, dim=[-2, -1])

            psf = torch.nn.functional.pad(psf, (0, 1, 0, 1))
            kernel = kornia.geometry.center_crop(psf, [65, 65])
            x_mean = kornia.enhance.normalize_min_max(kernel, 0., 1.)
            x_mean = x_mean.squeeze(1)
            x_mean = x_mean.cpu().detach().squeeze(0).numpy()

            out_way = os.path.join(result_dir, dataset_name,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_fftrelu' + '.png')
            sns_plot = plt.figure()

            sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False,
                        square=True)  # Reds_r .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()

            y.real = torch.relu(y.real)
            y.imag = torch.relu(y.imag)

            y = torch.fft.ifft2(y).abs() - inp / 2.
            # x = torch.log(torch.clamp(x, 1e-6))
            psfy = torch.fft.ifftshift(y, dim=[-2, -1])

            psfy = torch.nn.functional.pad(psfy, (0, 1, 0, 1))
            kernely = kornia.geometry.center_crop(psfy, [65, 65])
            y_mean = kornia.enhance.normalize_min_max(kernely, 0., 1.)
            y_mean = y_mean.squeeze(1)
            y_mean = y_mean.cpu().detach().squeeze(0).numpy()

            out_way = os.path.join(result_dir, dataset_name,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_fftrelu_sharp' + '.png')
            sns_plot = plt.figure()

            sns.heatmap(y_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False,
                        square=True)  # Reds_r .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()

            cv2.imwrite(os.path.join(result_dir, dataset_name, os.path.split(file_)[-1]), x_)
            cv2.imwrite(os.path.join(result_dir, dataset_name, os.path.splitext(os.path.split(file_)[-1])[0] + '_sharp' + '.png'), y_)

            x_mean = kornia.enhance.normalize_min_max(ker_reblur_fft, 0., 1.)
            x_mean = x_mean.squeeze(1)
            x_mean = x_mean.cpu().detach().squeeze(0).numpy()
            out_way = os.path.join(result_dir, dataset_name,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_reblur_fft.png')
            sns_plot = plt.figure()
            sns.heatmap(x_mean, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()

            x_mean = kornia.enhance.normalize_min_max(ker_relu_fft, 0., 1.)
            x_mean = x_mean.squeeze(1)
            x_mean = x_mean.cpu().detach().squeeze(0).numpy()
            out_way = os.path.join(result_dir, dataset_name,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_relu_fft.png')
            sns_plot = plt.figure()
            sns.heatmap(x_mean, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()

            ker_reblur = convert_otf2psf(otf_gt_reblur_, [65, 65])
            ker_reblur = ker_topk(ker_reblur, 61, 65)
            x_mean = kornia.enhance.normalize_min_max(ker_reblur, 0., 1.)
            x_mean = x_mean.squeeze(1)
            x_mean = x_mean.cpu().detach().squeeze(0).numpy()

            out_way = os.path.join(result_dir, dataset_name, os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_reblur' + '.png')
            sns_plot = plt.figure()

            sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False,
                        square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()