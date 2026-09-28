## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

##### Data preparation file for training Restormer on the GoPro Dataset ########

import cv2
import numpy as np
from glob import glob
import random
# from PIL.ImagePalette import random
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx
from joblib import Parallel, delayed
import multiprocessing

def train_files(file_):
    hr_file = file_
    filename = os.path.splitext(os.path.split(hr_file)[-1])[0]
    # lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = hr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int32))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int32))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                # lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                
                # lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.png')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.png')
                
                # cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)

    else:
        # lr_savename = os.path.join(lr_tar, filename + '.png')
        hr_savename = os.path.join(hr_tar, filename + '.png')
        
        # cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)

def val_files(file_):
    hr_file = file_
    hr_path_dir = os.path.dirname(hr_file)
    # filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    img_file = os.path.basename(hr_file)
    file_dirs = hr_file.split('/')
    a, b = file_dirs[-7], file_dirs[-4]
    key_name = os.path.splitext(img_file)[0] # img_file.split('.{}'.format('png'))[0]
    # if data_name == 'sharp':
    file_name_ = key_name.split('_')
    key_name = file_name_[0]
    if len(file_name_) > 2:
        for x in file_name_[1:-1]:
            key_name = key_name + '_' + x
    # print(key_name)
    lq_path_ = os.path.join(hr_path_dir.replace('sharp', 'blur'), key_name + '_*.png')
    lr_files = glob(lq_path_)
    lr_file = random.choice(lr_files)
    key_name = a + '-' + b + '-' + key_name
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)

    lr_savename = os.path.join(lr_tar, key_name + '.png')
    hr_savename = os.path.join(hr_tar, key_name + '.png')

    w, h = lr_img.shape[:2]

    i = (w-val_patch_size)//2
    j = (h-val_patch_size)//2
                
    lr_patch = lr_img[i:i+val_patch_size, j:j+val_patch_size,:]
    hr_patch = hr_img[i:i+val_patch_size, j:j+val_patch_size,:]

    cv2.imwrite(lr_savename, lr_patch)
    cv2.imwrite(hr_savename, hr_patch)

# ############ Prepare Training data ####################
num_cores = 10
# patch_size = 512
# overlap = 256
# p_max = 0
#
# # patch_size = 384
# # overlap = 0
# # overlap = 48
# # src = '/data/mxt_data/DPDD/train'
# tar = '/data/mxt_data/DOTA/train'
#
# # lr_tar = os.path.join(tar, 'input_crops')
# hr_tar = os.path.join(tar, 'target_crops')
#
# # os.makedirs(lr_tar, exist_ok=True)
# os.makedirs(hr_tar, exist_ok=True)
#
# # lr_files = natsorted(glob(os.path.join(src, 'inputC', '*.png')) + glob(os.path.join(src, 'inputC', '*.jpg')))
# hr_files = natsorted(glob(os.path.join(tar, 'sharp', '*.png')) + glob(os.path.join(tar, 'sharp', '*.jpg')))
#
# # files = [(i, j) for i, j in zip(lr_files, hr_files)]
# files = hr_files
# Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))


############ Prepare validation data ####################
# val_patch_size = 256
val_patch_size = 512
src = '/data/mxt_data/MedicalDefocus'
tar = '/data/mxt_data/MedicalDefocus_val'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)
files = sorted(glob(os.path.join(src, '*', 'test', 'sharp', '*', '*', 'digi', '*.png')))
random.shuffle(files)
# lr_files = natsorted(glob(os.path.join(src, 'blur', '*.png')) + glob(os.path.join(src, 'blur', '*.jpg')))
# hr_files = natsorted(glob(os.path.join(src, 'sharp', '*.png')) + glob(os.path.join(src, 'sharp', '*.jpg')))

# files = [(i, j) for i, j in zip(lr_files, hr_files)]
end = 800
Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files[:end]))
