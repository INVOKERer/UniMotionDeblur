## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2

import concurrent.futures
import sys
import pyiqa
import torch
import piq
from piq import fsim
device = 'cuda'
# iqa_metrics = {
#         'LPIPS': pyiqa.create_metric('lpips', device=device),
#         # 'NIMA': pyiqa.create_metric('nima', device=device),
#         # 'NIQE': pyiqa.create_metric('niqe', device=device)
#     }
from PIL import Image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def proc(filename):
    tar,prd = filename
    sr_img = Image.open(tar).convert("RGB")
    gt_img = Image.open(prd).convert("RGB")
    sr_img = np.array(sr_img)
    gt_img = np.array(gt_img)


    sr_tensor = img2tensor(sr_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
    gt_tensor = img2tensor(gt_img, bgr2rgb=False, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
    metric = fsim(sr_tensor, gt_tensor, data_range=1.0, chromatic=False)
    # print(metric)
    # with torch.no_grad():
    #     metrics = {}
    #     for name, metric in iqa_metrics.items():
    #         if name in ['NIMA', 'CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'PAQ2PIQ']:
    #             metrics[name] = metric(sr_tensor).item()
    #         else:
    #             metrics[name] = metric(sr_tensor, gt_tensor).item()
    # print(PSNR, SSIM)
    return np.array(metric.cpu()).mean()

# datasets = ['RealBlur_R', 'RealBlur_J']
# datasets = ['FoundIR']
# datasets = ['RealBlur_R']
# datasets = ['RealBlur_J']
datasets = ['BSD1', 'BSD2', 'MIORe', 'VAR-MIORe']
compare_methods = ['MIMOUNet', 'DeepRFT', 'NAFNet64', 'LoFormer', 'PromptIR', 'MoCEIR', 'BioIR', 'EVSSM', 'UniDeblur-B-k123','UniDeblur-L-k123']
# compare_methods = ['UniDeblur-B-k123', 'NAFNet64']
out_root = '/home/ubuntu/150t/personal_data/mxt/MXT/ckpts/uniDeblur'



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

        lpips, niqe, nima = [], [], []
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
                lpips.append(PSNR_SSIM)
                # niqe.append(PSNR_SSIM['NIQE'])
                # nima.append(PSNR_SSIM['NIMA'])
                # print(PSNR_SSIM)
        # except:
        #     print('x')
        #     psnr.append(0.)
        #     ssim.append(0.)

        avg_lpips = sum(lpips)/len(lpips)
        # avg_niqe = sum(niqe)/len(niqe)
        # avg_nima = sum(nima) / len(nima)
        print('For {:s} dataset FSIM: {:f}\n'.format(dataset, avg_lpips))
        meta_info_file = os.path.join(out_root, dataset + '_' + 'FSIM.txt')
        with open(meta_info_file, 'a') as f:
            info = 'For {:s} method FSIM: {:f} \n'.format(method, avg_lpips)
            # print(info)
            f.write(f'{info}\n')