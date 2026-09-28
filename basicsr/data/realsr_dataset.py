import os.path

from torch.utils import data as data
import glob
import torch
import torch.nn.functional as F
import sys
import cv2

import numpy as np
import math
import random

sys.path.append('/data/mxt_code/AllinOne')
from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
    paths_from_folder,
    paths_from_lmdb,
    single_rgb_paths_from_medical_folder, paired_paths_from_lmdb_medical, prctile_norm
)
# from basicsr import USMSharp
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import augment, center_crop, paired_random_crop, random_crop
from basicsr.data.degradations import random_generate_poisson_noise, circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import USMSharp, DiffJPEG, FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.mosaic_util import mosaic_CFA_Bayer
from basicsr.utils.img_process_util import filter2D

# from .data_util import prctile_norm
def degradation(gt, kernel1, kernel2, sinc_kernel, jpeger, usm_sharpener, second_blur, opt):

    gt, kernel1, kernel2, sinc_kernel, jpeger, usm_sharpener = gt.cuda(), kernel1.cuda(), kernel2.cuda(), sinc_kernel.cuda(), jpeger.cuda(), usm_sharpener.cuda()
    if not second_blur:
        c, ori_h, ori_w = gt.shape[-3:]
        out = filter2D(gt, kernel1)
        # random resize
        updown_type = random.choices(['down', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'down':
            scale = np.random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(out, scale_factor=scale, mode=mode)
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        out = torch.clamp(out, 0., 1.)
        lq = torch.clamp((out.cpu() * 255.0).round(), 0, 255)
        return lq
    gt_usm = usm_sharpener(gt)
    c, ori_h, ori_w = gt_usm.shape[-3:]
    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, kernel1)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise

    if np.random.uniform() < 0.25:
        gray_noise_prob = opt['gray_noise_prob']
        if np.random.uniform() < opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
    # JPEG compression
    if np.random.uniform() < 0.5 and c == 3:
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)
    else:
        out = torch.clamp(out, 0, 1)
    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale), int(ori_w / scale)), mode=mode)
        # add noise
        if np.random.uniform() < 0.25:
            gray_noise_prob = opt['gray_noise_prob2']
            if np.random.uniform() < opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        # if np.random.uniform() < 0.5:
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            if c == 3:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            if c == 3:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
            out = filter2D(out, sinc_kernel)
    else:
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        out = filter2D(out, sinc_kernel)
    # clamp and round
    lq = torch.clamp((out.cpu() * 255.0).round(), 0, 255) # / 255.
    return lq

# @DATASET_REGISTRY.register()
class PairedImageGaussianBlurSRDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageGaussianBlurSRDataset, self).__init__()
        self.opt = opt
        self.jpeger = DiffJPEG(differentiable=False)
        self.usm_sharpener = USMSharp()
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.decode = opt.get("decode", True)
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None
        self.pad_size = 10
        self.gt_size = opt.get("gt_size", 128) + 2 * self.pad_size
        self.scale_range = opt.get("scale_range", [0.05, 3])
        self.center_crop = opt["center_crop"] + 2 * self.pad_size if "center_crop" in opt else None

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.gt_folder = opt["dataroot_gt"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = paths_from_folder(self.gt_folder)

        self.depth = self.opt.get("depth", 8)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        self.flag = "color"
        if "color" in self.opt and self.opt["color"] == "y":
            self.flag = "grayscale"

        self.float32 = not self.opt.get("prctile_norm", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.paths[index]
        if self.decode:
            img_bytes = self.file_client.get(gt_path, "gt")
            img_gt = imfrombytes(
                img_bytes,
                flag=self.flag,
                depth=self.depth,
                float32=self.float32,
            )
        else:
            img_bytes = self.file_client.get(gt_path, "gt")
            img = np.frombuffer(img_bytes, dtype=np.uint16)
            h, w, c = img[0:3]
            img_gt = img[3:].reshape(h, w, c)
            if self.float32:
                img_gt = img_gt.astype(np.float32) / 255.0

        # img_lq = img_gt.copy()
        # augmentation for training
        if self.opt["phase"] == "train":
            if img_gt.shape[-1] < self.gt_size or img_lq.shape[-2] < self.gt_size:
                pad_h = self.gt_size - img_gt.shape[-1] if self.gt_size > img_gt.shape[-1] else 0
                pad_w = self.gt_size - img_gt.shape[-2] if self.gt_size > img_gt.shape[-2] else 0
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)
                # img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT101)

            # random crop
            img_gt = random_crop(
                img_gt, self.gt_size, 1, gt_path
            )
            # flip, rotation
            img_gt = augment(
                [img_gt], self.opt["use_hflip"], self.opt["use_rot"]
            )
        else:
            if self.center_crop is not None:
                img_gt = center_crop(img_gt, self.center_crop)
                # img_lq = center_crop(img_lq, self.center_crop)

        # norm
        if self.opt.get("prctile_norm", False):
            img_gt = prctile_norm(img_gt)
            # img_lq = prctile_norm(img_lq)

        # BGR to RGB
        if img_gt.shape[-1] == 3:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            # img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        # HWC to CHW, numpy to tensor

        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        # img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()

        img_lq = degradation(img_gt.unsqueeze(0), kernel, kernel2, sinc_kernel, self.jpeger, self.usm_sharpener, self.opt).squeeze(0) / 255.
        # img_lq = img_lq.contiguous()
        img_gt = img_gt[:, self.pad_size: -self.pad_size, self.pad_size: -self.pad_size].contiguous()
        img_lq = img_lq[:, self.pad_size: -self.pad_size, self.pad_size: -self.pad_size].contiguous().detach()

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }

    def __len__(self):
        return len(self.paths)
# @DATASET_REGISTRY.register()
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __len__(self):
        return len(self.paths)


def image_degrade(img_gt, opt):
    # if len(img_gt.shape) == 2:
    #     c = 1
    #     img_gt = torch.from_numpy(img_gt).unsqueeze(0).float().contiguous() / 255.
    # else:
    #     c = 3
    #     img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous() / 255.
    second_blur = False
    # TODO: kernel range is now hard-coded, should be in the configure file
    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob']:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            opt['kernel_list'],
            opt['kernel_prob'],
            kernel_size,
            opt['blur_sigma'],
            opt['blur_sigma'], [-math.pi, math.pi],
            opt['betag_range'],
            opt['betap_range'],
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < opt['sinc_prob2']:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            opt['kernel_list2'],
            opt['kernel_prob2'],
            kernel_size,
            opt['blur_sigma2'],
            opt['blur_sigma2'], [-math.pi, math.pi],
            opt['betag_range2'],
            opt['betap_range2'],
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < opt['final_sinc_prob']:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    # BGR to RGB, HWC to CHW, numpy to tensor
    # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)
    img_lq = degradation(img_gt.unsqueeze(0), kernel, kernel2, sinc_kernel, jpeger, usm_sharpener, second_blur, opt).squeeze(0)

    return img_lq
if __name__ == '__main__':
    '''
                resize_prob: [0.2, 0.7, 0.1],  # up, down, keep
            resize_range: [0.15, 1.5],
            gaussian_noise_prob: 0.5,
            noise_range: [1, 30],
            poisson_scale_range: [0.05, 3],
            gray_noise_prob: 0.4,
            jpeg_range: [30, 95],
            # the second degradation process
            second_blur_prob: 0.8,
            resize_prob2: [0.3, 0.4, 0.3],  # up, down, keep
            resize_range2: [0.3, 1.2],
            gaussian_noise_prob2: 0.5,
            noise_range2: [1, 25],
            poisson_scale_range2: [0.05, 2.5],
            gray_noise_prob2: 0.4,
            jpeg_range2: [30, 95],
        blur_kernel_size: 21,
        kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        sinc_prob: 0.1,
        blur_sigma: [0.2, 3],
        betag_range: [0.5, 4],
        betap_range: [1, 2],

        blur_kernel_size2: 21,
        kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        sinc_prob2: 0.1,
        blur_sigma2: [0.2, 1.5],
        betag_range2: [0.5, 4],
        betap_range2: [1, 2],

        final_sinc_prob: 0.8,
        '''
    import cv2
    import torch
    import numpy as np
    import math
    import random
    import tqdm
    import time
    jpeger = DiffJPEG(differentiable=False)
    usm_sharpener = USMSharp()
    opt = {  # the first degradation process
        'resize_prob': [0.2, 0.7, 0.1],  # up, down, keep
        'resize_range': [0.15, 1.5],
        'gaussian_noise_prob': 0.5,
        'noise_range': [1, 30],
        'poisson_scale_range': [0.05, 3],
        'gray_noise_prob': 0.4,
        'jpeg_range': [30, 95],
        # the second degradation process
        'second_blur_prob': 0.8,
        'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
        'resize_range2': [0.3, 1.2],
        'gaussian_noise_prob2': 0.5,
        'noise_range2': [1, 25],
        'poisson_scale_range2': [0.05, 2.5],
        'gray_noise_prob2': 0.4,
        'jpeg_range2': [30, 95],
        'blur_kernel_size': 21,
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob': 0.1,
        'blur_sigma': [0.2, 3],
        'betag_range': [0.5, 4],
        'betap_range': [1, 2],

        'blur_kernel_size2': 21,
        'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob2': 0.1,
        'blur_sigma2': [0.2, 1.5],
        'betag_range2': [0.5, 4],
        'betap_range2': [1, 2],

        'final_sinc_prob': 0.8,
    }
    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    # img_dir = '/data/mxt_data/DOTA/train/sharp'
    # out_dir = '/data/mxt_data/DOTA/train/input_cropsX'
    # os.makedirs(out_dir, exist_ok=True)
    # out_dir2 = '/data/mxt_data/DOTA/train/target_cropsX'
    # os.makedirs(out_dir2, exist_ok=True)
    img_dir = '/data/mxt_data/SR/DF2K/HR'
    out_dir = '/data/mxt_data/SR/DF2K/input_cropsX'
    os.makedirs(out_dir, exist_ok=True)
    out_dir2 = '/data/mxt_data/SR/DF2K/target_cropsX'
    os.makedirs(out_dir2, exist_ok=True)
    img_file_list = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))

    cut_patches = True
    pad_size = 10
    patch_size = 512 + 2 * pad_size
    overlap = 256
    p_max = 0
    for img_file in tqdm.tqdm(img_file_list):
        # img_file = '/data/mxt_data/SR/SR/img012.png'
        img_file_name = os.path.basename(img_file)

        # if os.path.exists(out_img_file):
        #     continue
        img_gt = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img_gt_shape = img_gt.shape
        if len(img_gt.shape) == 2:
            c = 1
            img_gt = torch.from_numpy(img_gt).unsqueeze(0).float().contiguous()
        else:
            c = 3
            img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        if not cut_patches:
            if img_gt_shape[0] < 512 or img_gt_shape[1] < 512:
                os.system('rm {}'.format(img_file))
                print('rm {}'.format(img_file))
                time.sleep(0.001)
                continue
            out_img_file = os.path.join(out_dir, img_file_name)
            if os.path.exists(out_img_file):
                continue
            img_lq = image_degrade(img_gt / 255., opt)
            if c == 1:
                img_lq = img_lq.squeeze(0).cpu().detach().numpy()
            else:
                img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
            if img_lq.shape != img_gt_shape:
                print(img_lq.shape, img_gt_shape)

            cv2.imwrite(out_img_file, img_lq.astype('uint8'))
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
                for i in w1:
                    for j in h1:
                        num_patch += 1

                        # lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                        hr_patch = img_gt[:, i:i + patch_size, j:j + patch_size]
                        if hr_patch.shape[-2] != patch_size or hr_patch.shape[-1] != patch_size:
                            print(hr_patch.shape, filename)
                            continue
                        img_lq = image_degrade(hr_patch / 255., opt)[:, pad_size:-pad_size, pad_size:-pad_size]
                        gt_patch = hr_patch[:, pad_size:-pad_size, pad_size:-pad_size]
                        if c == 1:
                            img_lq = img_lq.squeeze(0).cpu().detach().numpy()
                            gt_patch = gt_patch.squeeze(0).cpu().detach().numpy()
                        else:
                            img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
                            gt_patch = gt_patch.cpu().detach().permute(1, 2, 0).numpy()
                        lr_savename = os.path.join(out_dir, filename + '-' + str(num_patch) + '.png')
                        hr_savename = os.path.join(out_dir2, filename + '-' + str(num_patch) + '.png')
                        # print(lr_savename)
                        cv2.imwrite(lr_savename, img_lq.astype('uint8'))
                        cv2.imwrite(hr_savename, gt_patch.astype('uint8'))

            else:
                img_lq = image_degrade(img_gt / 255., opt)
                if c == 1:
                    img_lq = img_lq.squeeze(0).cpu().detach().numpy()
                    img_gt = img_gt.squeeze(0).cpu().detach().numpy()
                else:
                    img_lq = img_lq.cpu().detach().permute(1, 2, 0).numpy()
                    img_gt = img_gt.cpu().detach().permute(1, 2, 0).numpy()
                lr_savename = os.path.join(out_dir, filename + '.png')
                hr_savename = os.path.join(out_dir2, filename + '.png')

                cv2.imwrite(lr_savename, img_lq.astype('uint8'))
                cv2.imwrite(hr_savename, img_gt.astype('uint8'))
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