import numpy as np
import glob
import cv2
import os
import kornia
import tqdm
import torch
# import spectral
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from natsort import natsorted
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
class GridX():
    def __init__(self, train_size=256, overlap_size=32, window_size=1, diff_max=2, diff_max_method=2, cal_method='fft_hsi'):
        '''

        :param train_size:
        :param overlap_size:
        :param window_size:
        :param diff_max:
        :param cal_method: fft_hsi, fft_rgb, laplace, brenner
        '''
        self.train_size = train_size

        self.grid = True
        self.overlap_size = (overlap_size, overlap_size)
        self.window_size = window_size
        self.diff_max = diff_max
        self.diff_max_method = diff_max_method
        # self.overlap_size = (16, 16)
        print(self.overlap_size)
        self.kernel_size = [train_size, train_size]
        # self.pix_max = pix_max
        ###########################
        self.h = None
        self.w = None
        self.cal_method = cal_method
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = self.overlap_size  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])
        self.stride = stride
        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col
        step_j = k2 if num_col == 1 else stride[1] # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0] # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        idxes_sharp = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                if self.cal_method == 'fft_rgb':
                    x_sharp, idx_sharp = self.cal_best_fft_rgb(x[:, :, i:i + k1, j:j + k2].cuda())
                elif self.cal_method == 'laplace_rgb':
                    x_sharp, idx_sharp = self.cal_best_laplace_rgb(x[:, :, i:i + k1, j:j + k2].cuda())
                elif self.cal_method == 'brenner_rgb':
                    x_sharp, idx_sharp = self.cal_best_brenner_rgb(x[:, :, i:i + k1, j:j + k2].cuda())
                elif self.cal_method == 'laplace_hsi':
                    x_sharp, idx_sharp = self.cal_best_laplace(x[:, :, i:i + k1, j:j + k2].cuda())
                elif self.cal_method == 'brenner_hsi':
                    x_sharp, idx_sharp = self.cal_best_brenner(x[:, :, i:i + k1, j:j + k2].cuda())
                elif self.cal_method == 'fft_hsi':
                    x_sharp, idx_sharp = self.cal_best_fft(x[:, :, i:i + k1, j:j + k2].cuda())
                # parts.append(x[3, :, i:i + k1, j:j + k2].unsqueeze(0))
                parts.append(x_sharp)
                idxes_sharp.append(idx_sharp)
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts, idxes_sharp
    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)
    def grids_inverse(self, outs):
        b, c, h, w = self.original_size
        preds = torch.zeros([1, c, h, w]).to(outs.device)
        # b, c, h, w = self.original_size
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # print(outs.shape)
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size[0], :] *= self.fuse_matrix_h2.to(outs.device)
            if i + k1*2 - self.ek1 < h:
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
                outs[cnt, :,  -self.overlap_size[0]:, :] *= self.fuse_matrix_h1.to(outs.device)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] *= self.fuse_matrix_eh2.to(outs.device)
            if i + k1*2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] *= self.fuse_matrix_eh1.to(outs.device)

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size[1]] *= self.fuse_matrix_w2.to(outs.device)
            if j + k2*2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size[1]:] *= self.fuse_matrix_w1.to(outs.device)
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] *= self.fuse_matrix_ew2.to(outs.device)
            if j + k2*2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] *= self.fuse_matrix_ew1.to(outs.device)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds # / count_mt

    def forward(self, inp_img, get_idx=False):
        B, C, H, W = inp_img.shape
        if get_idx:
            parts, idx = self.cal_best_fft_rgb(inp_img.cuda(), True)
            return idx
        else:
            x, idxes_sharp = self.grids(inp_img)
            # print(x)
            x = self.grids_inverse(x)
            return x[:,:,:H,:W].contiguous(), idxes_sharp

    def cal_best_fft_rgb(self, inp, get_idx=False):
        # b = inp.shape[0]
        # inpx = inp.clone()
        # mean_x = torch.mean(inp[0])
        # for i in range(b):
        #     # inp[i, ...] = kornia.enhance.normalize_min_max(inp[i, ...], min_val=k_min, max_val=k_max.cpu().numpy())
        #     inp[i, ...] = inp[i, ...] / torch.mean(inp[i, ...]) * mean_x
        x = torch.fft.rfft2(inp)
        x.real = torch.relu(x.real)
        x.imag = torch.relu(x.imag)
        x = torch.fft.irfft2(x) - inp / 2.
        # x = torch.fft.fftshift(x, dim=[-2, -1])
        # x = kornia.geometry.center_crop(x, [2, 2])
        x = x[:, :, :self.window_size, :self.window_size] + x[:, :, -self.window_size:, :self.window_size] + \
            x[:, :, :self.window_size, -self.window_size:] + x[:, :, -self.window_size:, -self.window_size:]
        x, idx = torch.max(torch.mean(x, dim=[1, 2, 3], keepdim=False), dim=0)

        if get_idx:
            return idx
        else:
            return inp[idx, ...].unsqueeze(0), idx

    def cal_best_laplace_rgb(self, inp, get_idx=False):
        x = kornia.filters.laplacian(inp, kernel_size=5)
        x, idx = torch.max(torch.mean(x**2, dim=[1, 2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return inp[idx, ...].unsqueeze(0)
    def cal_best_brenner_rgb(self, inp, get_idx=False):
        # kernel = torch.zeros([5, 5])
        # kernel[2, 2] = -1
        # kernel[2, -1] = 1
        # x = kornia.filters.filter2d(inp, kernel.unsqueeze(0))
        x = torch.nn.functional.pad(inp, (0, 2, 0, 0))
        x = x[:, :, :, 2:] - inp
        x, idx = torch.max(torch.mean(x**2, dim=[1, 2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return inp[idx, ...].unsqueeze(0)

    def check_idx(self, idx):

        idx_new1 = []
        idx_new2 = []
        c_range = list(range(idx.shape[0]))
        center_band = len(c_range) // 2
        # t_band_start = len(c_range) // 3
        # t_band_end = len(c_range)*2 // 3
        # center_idx = torch.mean(idx[t_band_start:t_band_end])
        c_range1 = c_range[center_band:]
        c_k_forward = None
        for c in c_range1:
            k = idx[c]
            # print(inp_c.shape) # b 1 h w
            if c_k_forward is None:
                c_k_forward = k
                # inp_z = torch.index_select(inp_c, dim=0, index=k)
            if abs(k.item() - c_k_forward.item()) >= self.diff_max:
                k = c_k_forward
            else:
                c_k_forward = k
            idx_new1.append(k.item())
        c_range2 = c_range[:center_band]
        c_k_forward = None
        for c in c_range2[::-1]:
            k = idx[c]
            # print(inp_c.shape) # b 1 h w
            if c_k_forward is None:
                c_k_forward = k
                # inp_z = torch.index_select(inp_c, dim=0, index=k)
            if abs(k.item() - c_k_forward.item()) >= self.diff_max:
                k = c_k_forward
            else:
                c_k_forward = k
            idx_new2.append(k.item())
        idx_new2 = idx_new2[::-1]
        idx_new = idx_new2 + idx_new1
        return idx_new
    def fuse_channel(self, inp, idx):
        results = []
        # c_k_forward = None
        c_range = list(range(idx.shape[0]))
        # idx = self.check_idx(idx)
        # center_band = len(c_range) // 2
        # c_range = c_range[center_band:] + c_range[:center_band]
        for c in c_range:
            k = idx[c]
            inp_c = torch.index_select(inp, dim=1, index=torch.tensor(c, device=inp.device))
            # print(inp_c.shape) # b 1 h w
            # if c_k_forward is None:
            #     c_k_forward = k
            #     # inp_z = torch.index_select(inp_c, dim=0, index=k)
            # if abs(k - c_k_forward) >= self.diff_max:
            #     k = c_k_forward
            inp_z = torch.index_select(inp_c, dim=0, index=torch.tensor(k, device=inp.device))

            # print(inp_z.shape)
            results.append(inp_z)  # inp[idx, i, :, :].unsqueeze(0))
        # results = results[center_band + 1:] + results[:center_band + 1]
        result = torch.cat(results, dim=1)
        return result
    def cal_best_fft(self, inp, get_idx=False):
        x = torch.fft.rfft2(inp)
        x.real = torch.relu(x.real)
        x.imag = torch.relu(x.imag)
        x = torch.fft.irfft2(x) - inp / 2.
        # x = torch.fft.fftshift(x, dim=[-2, -1])
        # x = kornia.geometry.center_crop(x, [2, 2])
        x = x[:, :, :self.window_size, :self.window_size] + x[:, :, -self.window_size:, :self.window_size] + \
            x[:, :, :self.window_size, -self.window_size:] + x[:, :, -self.window_size:, -self.window_size:]
        x, idx = torch.max(torch.mean(x, dim=[2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return self.fuse_channel(inp, idx)
    def cal_best_laplace(self, inp, get_idx=False):
        x = kornia.filters.laplacian(inp, kernel_size=5)
        x, idx = torch.max(torch.mean(x**2, dim=[2, 3], keepdim=False), dim=0, keepdim=False)

        if get_idx:
            return idx
        else:
            return self.fuse_channel(inp, idx)
    def cal_best_brenner(self, inp, get_idx=False):
        # kernel = torch.zeros([5, 5])
        # kernel[2, 2] = -1
        # kernel[2, -1] = 1
        # x = kornia.filters.filter2d(inp, kernel.unsqueeze(0))
        x = torch.nn.functional.pad(inp, (0, 2, 0, 0))
        x = x[:, :, :, 2:] - inp
        x, idx = torch.max(torch.mean(x ** 2, dim=[2, 3], keepdim=False), dim=0, keepdim=False)
        if get_idx:
            return idx
        else:
            return self.fuse_channel(inp, idx)
def judge_white_frelu_torch(img, thr=0.05):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.
    # img = kornia.image_to_tensor(img, keepdim=False)
    c = img.shape[1]
    if c > 1:
        img = kornia.color.rgb_to_grayscale(img)
    # print(img.shape)
    img_min, img_max = img.min(), img.max()
    img = (img-img_min)/(img_max-img_min)
    # img = kornia.enhance.normalize_min_max(img, 0., 1.)
    x = torch.fft.rfft2(img)
    # y_blur_f = np.fft.fftshift(y) 0.017

    x.real = torch.relu(x.real)
    x.imag = torch.relu(x.imag)
    x = torch.fft.irfft2(x)
    resx = x - img / 2.
    resx = resx.view(resx.shape[0], -1)
    resx = torch.quantile(resx, torch.tensor([0.05, 0.95], dtype=resx.dtype, device=resx.device), dim=1, keepdim=False)
    # print(resx.shape)
    res1, res2 = resx.chunk(2, dim=0)
    # print(res1.shape)
    res = res2 - res1 # .squeeze(1)
    # print(res)
    # _, idx = torch.where(res > thr)
    # print(idx)
    # print(res.item())
    return res.item() # > thr
def check_idx(idx, diff_max):

    idx_new1 = []
    idx_new2 = []
    c_range = list(range(idx.shape[0]))
    center_band = len(c_range) // 2
    # t_band_start = len(c_range) // 3
    # t_band_end = len(c_range)*2 // 3
    # center_idx = torch.mean(idx[t_band_start:t_band_end])
    c_range1 = c_range[center_band:]
    c_k_forward = None
    for c in c_range1:
        k = idx[c]
        # print(inp_c.shape) # b 1 h w
        if c_k_forward is None:
            c_k_forward = k
            # inp_z = torch.index_select(inp_c, dim=0, index=k)
        if abs(k.item() - c_k_forward.item()) >= diff_max:
            k = c_k_forward
        else:
            c_k_forward = k
        idx_new1.append(k.item())
    c_range2 = c_range[:center_band]
    c_k_forward = None
    for c in c_range2[::-1]:
        k = idx[c]
        # print(inp_c.shape) # b 1 h w
        if c_k_forward is None:
            c_k_forward = k
            # inp_z = torch.index_select(inp_c, dim=0, index=k)
        if abs(k.item() - c_k_forward.item()) >= diff_max:
            k = c_k_forward
        else:
            c_k_forward = k
        idx_new2.append(k.item())
    idx_new2 = idx_new2[::-1]
    idx_new = idx_new2 + idx_new1
    return idx_new
def fuse_channel(inp, idx):
    results = []
    # c_k_forward = None
    c_range = list(range(len(idx)))
    for c in c_range:
        k = idx[c]
        inp_c = torch.index_select(inp, dim=1, index=torch.tensor(c, device=inp.device))
        # print(inp_c.shape) # b 1 h w
        # if c_k_forward is None:
        #     c_k_forward = k
        #     # inp_z = torch.index_select(inp_c, dim=0, index=k)
        # if abs(k - c_k_forward) >= self.diff_max:
        #     k = c_k_forward
        inp_z = torch.index_select(inp_c, dim=0, index=torch.tensor(k, device=inp.device))

        # print(inp_z.shape)
        results.append(inp_z)  # inp[idx, i, :, :].unsqueeze(0))
    # results = results[center_band + 1:] + results[:center_band + 1]
    result = torch.cat(results, dim=1)
    return result


def sort_key(x):
    x = os.path.basename(x)
    x, ext = os.path.splitext(x)
    x = x.split('_')[-1]
    # print(x)
    return float(x)
def get_dataset(out_dir, img_blur_root, rois, preprocess=False):

    # out_dir_all = os.path.join(out_dir, 'data_all')
    # os.makedirs(out_dir_all, exist_ok=True)

    num_k = 0
    rgb_or_hsis = ['rgb']

    for rgb_or_hsi in rgb_or_hsis:
        ext = '.png' if rgb_or_hsi == 'rgb' else '.npz'
        pix_max = 255. # if rgb_or_hsi == 'rgb' else 65535.
        for roi in tqdm.tqdm(rois):
            # print(roi)
            data_name = os.path.basename(roi)
            slide_name = roi.split('/')[-2]
            dir_name = roi
            dir_name_out = os.path.dirname(os.path.dirname(dir_name))
            # print(dir_name_out)
            out_dir_datab = dir_name_out.replace('blur', 'input_crops')
            # os.makedirs(out_dir_datab, exist_ok=True)
            # if rgb_or_hsi == 'rgb':
            out_dir_data_b = out_dir_datab  # os.path.join(out_dir_datab, 'digi')

            out_dir_data = dir_name_out.replace('blur', 'target_crops')
            # os.makedirs(out_dir_data, exist_ok=True)
            # if rgb_or_hsi == 'rgb':
            out_dir_data_d = out_dir_data  # os.path.join(out_dir_data, 'digi')
            # else:
            #     out_dir_data_d = os.path.join(out_dir_data, 'homo')
            out_dir_datan = dir_name_out.replace('blur', 'ntarget_crops')
            # os.makedirs(out_dir_datan, exist_ok=True)
            # if rgb_or_hsi == 'rgb':
            out_dir_data_n = out_dir_datan  # os.path.join(out_dir_datan, 'digi')
            # else:
            #     out_dir_data_n = os.path.join(out_dir_datan, 'homo')
            # if os.path.exists(out_dir_data_b):
            #     os.system('rm {}'.format(meta_info_txt))
            #     os.system('rm -r {}'.format(out_dir_data_b))
            #     os.system('rm -r {}'.format(out_dir_data_d))
            #     os.system('rm -r {}'.format(out_dir_data_n))
            #     continue
            # else:
            #     continue
            # if os.path.exists(out_dir_data_n):
            #     continue
            os.makedirs(out_dir_data_b, exist_ok=True)
            os.makedirs(out_dir_data_d, exist_ok=True)
            os.makedirs(out_dir_data_n, exist_ok=True)

            # roi_o = roi_i
            # print(roi)
            # img_blur_list = []
            # img_blur_list_all = []
            if rgb_or_hsi == 'rgb':
                img_blur_list = glob.glob(os.path.join(roi, 'digi', '*' + ext))+glob.glob(os.path.join(roi.replace('blur','sharp'), 'digi', '*' + ext))+glob.glob(os.path.join(roi.replace('blur','nsharp'), 'digi', '*' + ext))
            # else:
            #     img_blur_list = glob.glob(os.path.join(roi, 'homo', '*' + ext))
            center = len(img_blur_list) // 2
            # print(img_blur_list)
            img_blur_list = natsorted(img_blur_list, key=sort_key)
            # print(len(img_blur_list))
            # print(img_blur_list)
                # if sel in select_list:

            x_blur_list = []
            x_sharp_list = []
            # print(img_blur_list)
            sel_start = center + select_list[0]  # - out_range_list[0]
            sel_end = center + select_list[-1]  # - out_range_list[0]
            # print(sel_start, sel_end)
            # blur_name = os.path.basename(img_blur_list[0])
            # blur_name, ext = os.path.splitext(blur_name)
            # blur_name = blur_name.split('_')[0]
            # blur_patch = kornia.tensor_to_image(blur_patch)
            # blur_patch = np.uint8(x_b * 255.)
            # out_dir_data = dir_name.replace('blur', 'sharp')
            # os.makedirs(out_dir_data, exist_ok=True)
            # if rgb_or_hsi == 'rgb':
            #     out_dir_data_d = os.path.join(out_dir_data, 'digi')
            # else:
            #     out_dir_data_d = os.path.join(out_dir_data, 'homo')
            # out_dir_datan = dir_name.replace('blur', 'nsharp')
            # os.makedirs(out_dir_datan, exist_ok=True)
            # if rgb_or_hsi == 'rgb':
            #     out_dir_data_n = os.path.join(out_dir_datan, 'digi')
            # else:
            #     out_dir_data_n = os.path.join(out_dir_datan, 'homo')
            # if os.path.exists(out_dir_data_d):
            #     continue
            # if os.path.exists(out_dir_data_n):
            #     continue
            # os.makedirs(out_dir_data_d, exist_ok=True)
            # os.makedirs(out_dir_data_n, exist_ok=True)
            for idx, c in enumerate(img_blur_list):
                # print(c)
                if idx < sel_start or idx > sel_end:
                    continue

                x_bx = cv2.imread(c)[320:-320, 320:-320, :]
                # print(x_bx.shape)
                # raise ValueError
                # x_sx = kornia.image_to_tensor(x_bx, keepdim=False)
                # x_sharp_list.append(x_sx)
                # if preprocess:
                #     # print(x_blank.shape, x_bx.shape)
                #     x_b = x_bx / x_blank # [512:-512, 512:-512, :]
                # else:
                #     x_b = x_bx / pix_max
                x_b = kornia.image_to_tensor(x_bx, keepdim=False)
                h, w = x_b.shape[-2:]

                # x_b = kornia.geometry.resize(x_b, [h//4, w//4])
                x_blur_list.append(x_b.cuda())

            x = torch.cat(x_blur_list, dim=0)
            h, w = x.shape[-2:]
            w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=np.int32))
            h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=np.int32))
            w1.append(w - patch_size)
            h1.append(h - patch_size)
            # print(h1, w1)
            num_patch = 0
            for i in w1:
                for j in h1:
                    num_patch += 1

                    lr_patch = x[:, :, j:j + patch_size, i:i + patch_size]
                    lr_patch_ = lr_patch / pix_max
                    # x_s = torch.cat(x_sharp_list, dim=0)

                    idx_fft = gridx.cal_best_fft_rgb(lr_patch_, True) + sel_start
                    idx_sharp = idx_fft.item()
                    # if idx_sharp + 1 > sel_end:
                    #     idx_sharp = idx_sharp - 1
                    # elif idx_sharp-1 < sel_start:
                    #     idx_sharp = idx_sharp + 1
                    if idx_sharp + 1 > sel_end or idx_sharp-1 < sel_start:
                        # print(out_dir_data_d)
                        # print(idx_sharp)
                        # raise ValueError
                        continue

                    # print(out_dir_data_d, blur_name)
                    # if rgb_or_hsi == 'rgb':
                    # sharp_file = img_blur_list[idx_sharp]
                    # out_way = os.path.join(out_dir_data_d, os.path.basename(sharp_file))
                    # print('mv {} {}'.format(sharp_file, out_way))
                    # os.system('mv {} {}'.format(sharp_file, out_way))

                    # x_all = kornia.tensor_to_image(lr_patch, keepdim=False)
                    file_name, ext = os.path.splitext(os.path.basename(img_blur_list[idx_sharp]))
                    file_name_sharp_patch = slide_name + '-' + file_name + '-' + str(num_patch) + ext

                    for b in range(len(img_blur_list)):
                    # for b in range(lr_patch.shape[0]):
                        file_name, ext = os.path.splitext(os.path.basename(img_blur_list[b]))
                        # print(slide_name)
                        file_name_patch = slide_name + '-' + file_name + '-' + str(num_patch) + ext
                        x_data = cv2.imread(img_blur_list[b])[320:-320, 320:-320, :] # lr_patch[b, :, :, :].cpu().permute(1, 2, 0).numpy().astype('uint8')
                        x_data = x_data[j:j + patch_size, i:i + patch_size, :]
                        if b == idx_sharp:
                            out_way = os.path.join(out_dir_data_d, file_name_patch)

                        elif b in [idx_sharp-1, idx_sharp+1]:
                            out_way = os.path.join(out_dir_data_n, file_name_patch)
                        else:
                            out_way = os.path.join(out_dir_data_b, file_name_patch)
                            # f = open(meta_info_txt, mode='a')
                            with open(meta_info_txt, 'a') as f:
                                info = f'{file_name_patch} {file_name_sharp_patch}'
                                # print(info)
                                f.write(f'{info}\n')
                            # f.close()
                        # print(out_way, x_data.max(), x_data.min(), idx_sharp)
                        cv2.imwrite(out_way, x_data)
                        # print('mv {} {}'.format(nsharp_file, out_way))
                        # os.system('mv {} {}'.format(nsharp_file, out_way))

                    time.sleep(0.001)
                    # print('--------------------------------------')
            # else:
            #     out_way = os.path.join(out_dir_data_d, blur_name+'_res0.npz')
            #     np.savez_compressed(out_way, res0)
            #     out_way = os.path.join(out_dir_data_d, blur_name + '_res1.npz')
            #     np.savez_compressed(out_way, res1)
            #     out_way = os.path.join(out_dir_data_d, blur_name + '_res1_bgr.png')
            #     res1_bgr = res1[:, :, [4, 21, 32]]
            #     res1_bgr = (res1_bgr - res1_bgr.min()) / (res1_bgr.max()-res1_bgr.min())
            #     cv2.imwrite(out_way, res1_bgr*255.)
            #     out_way = os.path.join(out_dir_data_d, blur_name + '_res2.npz')
            #     np.savez_compressed(out_way, res2)
    # pd.DataFrame(pd_fft).to_excel(os.path.join(out_dir, 'fft.xls'), index=False, header=False)
    # pd.DataFrame(pd_brenner).to_excel(os.path.join(out_dir, 'brenner.xls'), index=False, header=False)
    # pd.DataFrame(pd_laplace).to_excel(os.path.join(out_dir, 'laplace.xls'), index=False, header=False)
# dx = ['RGB', 'Gray']
# dx = ['RGB']
img_blur_root = '/data/mxt_data/MedicalDefocus' # /hsi/view1/view1
k_name = os.path.basename(img_blur_root)
# img_root = os.path.join(img_blur_root, 'view1_0')
out_root = '/data/mxt_data/MedicalDefocus'
os.makedirs(out_root, exist_ok=True)
# img_list = glob.glob(img_root+'/*.hdr')
# roi_i = os.path.basename(img_blur_root)

# select_list = ['1', '-1', '2', '-2'] CUDA_VISIBLE_DEVICES
select_list = [-5, 5]
# print(select_list)
blur_ext = '.png'
roi_name = 'ROI'
# patch_size = 256


preprocess = False
preprocess_rgb = False

# blank_dir = '/home/ubuntu/106-48t/personal_data/mxt/zjn-mxt/20230407-dataget/hsi/blank'

slide_clss = glob.glob(os.path.join(img_blur_root, '*'))
for slide_cls in slide_clss:
    slide_clsname = os.path.basename(slide_cls)
    # for d in dx:
    pix_max = 255.
    patch_size = 512
    overlap = 64
    gridx = GridX(patch_size, 0, window_size=1, diff_max=2, diff_max_method=2, cal_method='fft_rgb')
    # print(slide_cls)
    rois = glob.glob(os.path.join(slide_cls, 'train', 'blur/*/*'))
    meta_info_txt = os.path.join(slide_cls, 'train', 'meta_info_train.txt')
    # rois = ['/home/ubuntu/90t/dataset/medical_focus/RuXianAi_ER/U82-ER-auto-M19-5081/roi1/roi1-0-14-5/roi1-0-62-12-2']
    # slide_clsname = 'RuXianAi_ER'
    # out_dir_test = os.path.join(out_root, slide_clsname, 'test')
    # os.makedirs(out_dir_test, exist_ok=True)
    # get_dataset(out_dir_test, img_blur_root, rois, preprocess_rgb)
    # print(rois)
    # train_list, test_list = train_test_split(rois, test_size=0.3, random_state=42)
    # train_or_test =
    # out_dir_train = os.path.join(out_root, slide_clsname, 'train')
    # os.makedirs(out_dir_train, exist_ok=True)
    get_dataset(out_root, img_blur_root, rois, preprocess_rgb)

