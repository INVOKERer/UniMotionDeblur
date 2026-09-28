import os
import yaml
from collections import OrderedDict
import time
from os import path as osp
from natsort import natsorted
import glob


out_root = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur'
dataset_name = 'BSD3'

meta_info_file = os.path.join(out_root, dataset_name + '_' + 'meta_info.txt')
with open(meta_info_file, 'a') as f:
    paths = natsorted(glob.glob(os.path.join(out_root, dataset_name, 'blur', '*')))
    for path in paths:

        blur_image_filename = os.path.basename(path)
        lq_meta_dst = os.path.join(dataset_name, 'blur', blur_image_filename)
        gt_meta_dst = os.path.join(dataset_name, 'sharp', blur_image_filename)
        info = f'{lq_meta_dst} {gt_meta_dst}'
        # print(info)
        f.write(f'{info}\n')
# meta_info_file = os.path.join(out_root, 'meta_info_MdeDefocus_2_test.txt')

# with open(meta_info_file, 'a') as f:
# for paths in paths_all:
#     blur_image_name = paths["lq_path"]
#     sharp_image_name = paths["gt_path"]
#     lq_path_dst = blur_image_name.replace('/data/mxt_data/', '')
#     gt_path_dst = sharp_image_name.replace('/data/mxt_data/', '')
#     info = f'{lq_path_dst} {gt_path_dst}'
#     # print(info)
#     f.write(f'{info}\n')
        # print(paths)
        # lq_paths = paths
        # # if not os.path.exists(dst):
        # lq_path_dst = os.path.dirname(lq_path).replace('/data/mxt_data', '/home/ubuntu/uniDeblurDataset')
        # gt_path_dst = os.path.dirname(gt_path).replace('/data/mxt_data', '/home/ubuntu/uniDeblurDataset')
        # # print(lq_path_dst, gt_path_dst)
        #
        # os.makedirs(lq_path_dst, exist_ok=True)
        # os.makedirs(gt_path_dst, exist_ok=True)
        # os.system('cp {} {}'.format(lq_path, lq_path_dst))
        # time.sleep(0.001)
        # os.system('cp {} {}'.format(gt_path, gt_path_dst))
        # time.sleep(0.001)
        # if meta_file is not None:
        #     os.system('cp {} {}'.format(meta_file, meta_file.replace('/data/mxt_data', '/home/ubuntu/uniDeblurDataset')))
        #     time.sleep(0.001)

# src_dir = '/data/mxt_data/GoPro/train/target_crops'
# dst_dir = '~/uniDeblurDataset/train/GoPro/train/target_crops'


# os.system('mv {} {}'.format(src_dir, dst_dir))