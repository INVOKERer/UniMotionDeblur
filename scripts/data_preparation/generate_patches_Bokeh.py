## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##### Data preparation file for training Restormer on the GoPro Dataset ########

import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing

def train_files(file_):

    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    index = lr_file.split('/')[-2]
    os.makedirs(os.path.join(tar, index), exist_ok=True)
    # print(tar, index)
    # print(os.path.join(tar, index))
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap[0], dtype=np.int32))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap[1], dtype=np.int32))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                
                lr_savename = os.path.join(tar, index, filename + '-' + str(num_patch) + '.png')
                
                cv2.imwrite(lr_savename, lr_patch)


############ Prepare Training data ####################
num_cores = 10
patch_size = 900
overlap = [300, 350]
p_max = 0

# patch_size = 384
# overlap = 0
# overlap = 48

src = '/data/mxt_data/RealBokeh/train/gt'
tar = '/data/mxt_data/RealBokeh/train/input_crops'

os.makedirs(tar, exist_ok=True)

lr_files = natsorted(glob(os.path.join(src, '*', '*.JPG')))

files = [(i, j) for i, j in zip(lr_files, lr_files)]

Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))

# src = '/data/mxt_data/timseizinger/RealBokeh_3MP/train/in'
# tar = '/data/mxt_data/timseizinger/RealBokeh_3MP/train/target_crops'
#
# os.makedirs(tar, exist_ok=True)
#
# lr_files = natsorted(glob(os.path.join(src, '*.JPG')))
#
# files = [(i, j) for i, j in zip(lr_files, lr_files)]
#
# Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))

############ Prepare validation data ####################
# val_patch_size = 256
# val_patch_size = 384
# src = '/home/ubuntu/Data1/mxt/GoPro/test'
# tar = '/home/ubuntu/Data1/mxt/GoPro/val'
#
# lr_tar = os.path.join(tar, 'input_crops_384')
# hr_tar = os.path.join(tar, 'target_crops_384')
#
# os.makedirs(lr_tar, exist_ok=True)
# os.makedirs(hr_tar, exist_ok=True)
#
# lr_files = natsorted(glob(os.path.join(src, 'blur', '*.png')) + glob(os.path.join(src, 'blur', '*.jpg')))
# hr_files = natsorted(glob(os.path.join(src, 'sharp', '*.png')) + glob(os.path.join(src, 'sharp', '*.jpg')))
#
# files = [(i, j) for i, j in zip(lr_files, hr_files)]
#
# Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
