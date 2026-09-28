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

    # out_dir_all = os.path.join(out_dir, 'data_all')
    # os.makedirs(out_dir_all, exist_ok=True)

    num_k = 0
    rgb_or_hsis = ['rgb']
    for rgb_or_hsi in rgb_or_hsis:

        for roi in tqdm.tqdm(rois):
            # print(roi)
            # data_name = os.path.basename(roi)
            # slide_name = roi.split('/')[-4]
            dir_blur = roi
            # print(dir_blur)
            dir_sharp = dir_blur.replace('blur', 'sharp')
            if not os.path.exists(dir_sharp) or len(os.listdir(os.path.join(dir_sharp, 'digi')))==0:

                # print('rm -r {}'.format(dir_blur))
                # print('rm -r {}'.format(dir_sharp))
                os.system('rm -r {}'.format(dir_blur))
                os.system('rm -r {}'.format(dir_sharp))
                time.sleep(0.001)
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

    rois = glob.glob(os.path.join(slide_cls, '*/blur/*/*'))
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

