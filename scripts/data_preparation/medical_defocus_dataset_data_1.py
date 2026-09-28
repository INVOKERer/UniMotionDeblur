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


def sort_key(x):
    x = os.path.basename(x)
    x, ext = os.path.splitext(x)
    x = x.split('_')[-1]
    # print(x)
    return float(x)
def get_dataset(out_dir, img_blur_root, rois, preprocess=False):

    out_dir_all = os.path.join(out_dir, 'blur')
    os.makedirs(out_dir_all, exist_ok=True)

    num_k = 0
    # rgb_or_hsis = ['rgb', 'hsi']
    rgb_or_hsis = ['rgb']
    for rgb_or_hsi in rgb_or_hsis:
        ext = '.png' if rgb_or_hsi == 'rgb' else '.hdr'
        pix_max = 255. if rgb_or_hsi == 'rgb' else 65535.
        for roi in tqdm.tqdm(rois):
            # print(roi)
            data_name = os.path.basename(roi)
            slide_name = roi.split('/')[-4]

            # roi_o = roi_i
            # print(roi)
            # img_blur_list = []
            # img_blur_list_all = []
            if rgb_or_hsi == 'rgb':
                img_blur_list = glob.glob(os.path.join(roi, 'digi', '*' + ext))
            else:
                img_blur_list = glob.glob(os.path.join(roi, 'homo', '*' + ext))
            center = len(img_blur_list) // 2
            # print(img_blur_list)
            img_blur_list = natsorted(img_blur_list, key=sort_key)
            # print(img_blur_list)
                # if sel in select_list:
                #     img_blur_list.append(os.path.join(roi, roi_i + '_' + str(sel) + '.hdr'))
            if preprocess:

                if rgb_or_hsi == 'rgb':
                    blank_i = 'blank.png'
                    roi_blank = '/home/ubuntu/106-48t/personal_data/mxt/Datasets/AutoFocus/company/company_blank'
                    print(roi_blank)
                    img_blank_file = os.path.join(roi_blank, blank_i)
                    x_blank = cv2.imread(img_blank_file)
                    x_blank = np.abs(x_blank) + 1e-7
                else:
                    blank_i = 'blankView'
                    img_blank_file = os.path.join(img_blur_root, blank_i, blank_i + '_0' + ext)
                    x_blank = spectral.open_image(img_blank_file)
                    x_blank = np.abs(x_blank[:, :, :]) + 1e-7
                # print(x_blank.type)
            # img_blur_list = glob.glob(img_root+'/' + roi_i[:-2] +'_*')
            # print(roi_i[:-2], len(img_blur_list))
            x_blur_list = []
            # print(img_blur_list)
            sel_start = center + select_list[0]  # - out_range_list[0]
            sel_end = center + select_list[-1]  # - out_range_list[0]

            out_dir_slide = os.path.join(out_dir_all, slide_name)
            os.makedirs(out_dir_slide, exist_ok=True)
            for idx, c in enumerate(img_blur_list):
                # print(c)
                # if idx < sel_start or idx > sel_end:
                #     continue
                # if rgb_or_hsi == 'rgb':
                #     x_bx = cv2.imread(c)
                # else:
                #     x_bx = spectral.open_image(c)
                #     x_bx = x_bx[:, :, :]

                # if preprocess:
                #     # print(x_blank.shape, x_bx.shape)
                #     x_b = x_bx / x_blank # [512:-512, 512:-512, :]
                # else:
                #     x_b = x_bx / pix_max

                # x = torch.cat(x_blur_list, dim=0)
                # x = torch.clamp(x, 0., 1.5)
                # x = kornia.enhance.normalize_min_max(x, 0., 1.)

                blur_name = os.path.basename(c)
                # blur_name, ext = os.path.splitext(blur_name)
                # blur_patch = kornia.tensor_to_image(blur_patch)
                # blur_patch = np.uint8(x_b * 255.)
                out_dir_data = os.path.join(out_dir_slide, data_name)
                os.makedirs(out_dir_data, exist_ok=True)
                if rgb_or_hsi == 'rgb':
                    out_dir_data_d = os.path.join(out_dir_data, 'digi')
                else:
                    out_dir_data_d = os.path.join(out_dir_data, 'homo')
                os.makedirs(out_dir_data_d, exist_ok=True)

                if rgb_or_hsi == 'rgb':
                    ori_blur_way = c

                    out_way = os.path.join(out_dir_data_d, blur_name)
                    img = cv2.imread(ori_blur_way)
                    h, w = img.shape[:2]
                    # print('mv {} {}'.format(ori_blur_way, out_way))
                    os.system('cp {} {}'.format(ori_blur_way, out_way))

                    time.sleep(0.001)
                    # cv2.imwrite(out_way, blur_patch)
                # else:
                #     out_way = os.path.join(out_dir_data_d, blur_name+'.npz')
                #     # np.save(out_way, blur_patch)
                #     np.savez_compressed(out_way, blur_patch)
                    # spectral.envi.save_image(out_way, blur_patch, force=False)
    # pd.DataFrame(pd_fft).to_excel(os.path.join(out_dir, 'fft.xls'), index=False, header=False)
    # pd.DataFrame(pd_brenner).to_excel(os.path.join(out_dir, 'brenner.xls'), index=False, header=False)
    # pd.DataFrame(pd_laplace).to_excel(os.path.join(out_dir, 'laplace.xls'), index=False, header=False)
# dx = ['RGB', 'Gray']
# dx = ['RGB']
img_blur_root = '/home/ubuntu/90t/dataset/medical_focus' # /hsi/view1/view1
k_name = os.path.basename(img_blur_root)
# img_root = os.path.join(img_blur_root, 'view1_0')
out_root = '/data/mxt_data/MedicalDefocus'
os.makedirs(out_root, exist_ok=True)
# img_list = glob.glob(img_root+'/*.hdr')
# roi_i = os.path.basename(img_blur_root)

# select_list = ['1', '-1', '2', '-2'] CUDA_VISIBLE_DEVICES
select_list = [-25, 25]
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
    # print(slide_cls)
    rois = glob.glob(os.path.join(slide_cls, '*/*/*/*'))
    # rois = ['/home/ubuntu/90t/dataset/medical_focus/RuXianAi_ER/U82-ER-auto-M19-5081/roi1/roi1-0-14-5/roi1-0-62-12-2']
    # slide_clsname = 'RuXianAi_ER'
    # out_dir_test = os.path.join(out_root, slide_clsname, 'test')
    # os.makedirs(out_dir_test, exist_ok=True)
    # get_dataset(out_dir_test, img_blur_root, rois, preprocess_rgb)
    # print(rois)
    train_list, test_list = train_test_split(rois, test_size=0.3, random_state=42)
    out_dir_train = os.path.join(out_root, slide_clsname, 'train')
    os.makedirs(out_dir_train, exist_ok=True)
    get_dataset(out_dir_train, img_blur_root, train_list, preprocess_rgb)

    out_dir_test = os.path.join(out_root, slide_clsname, 'test')
    os.makedirs(out_dir_test, exist_ok=True)
    get_dataset(out_dir_test, img_blur_root, test_list, preprocess_rgb)
