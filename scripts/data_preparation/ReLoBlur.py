# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
'''
for val set, extract the subset val-300

'''
import os
import time
from glob import glob
# from basicsr.utils.create_lmdb import create_lmdb_for_reds

def make_val_300(folder, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    templates = '*9.*'
    cp_command = 'cp {} {}'.format(os.path.join(folder, templates), dst)
    os.system(cp_command)


def flatten_folders(folder, n):
    vids = glob(os.path.join(folder, '*'))
    base_dir = os.path.dirname(folder)
    # base_name = base_name.split('_')[0]
    out_dir_sharp = os.path.join(base_dir, n, 'sharp')
    out_dir_blur = os.path.join(base_dir, n, 'blur')
    os.makedirs(out_dir_sharp, exist_ok=True)
    os.makedirs(out_dir_blur, exist_ok=True)
    for vid in vids:
        vidfolder_path = os.path.basename(vid)
        # print(vidfolder_path)
        # if not os.path.exists(os.path.join(folder, vidfolder_path)):
        #     continue

        kids = glob(os.path.join(vid, '*'))
        for kid in kids:
            # src_filename = '{:04}'.format(fid)
            krc_filename = os.path.basename(kid)
            ks_id, ext = os.path.splitext(krc_filename)
            # print(krc_filename)
            ks_id_, ids = ks_id.split('_')
            # print(ks_id_, ids, ext)
            if ids == 'sharp':
                out_dir = out_dir_sharp
            else:
                out_dir = out_dir_blur
            src_filepath = kid  # os.path.join(folder, vidfolder_path, src_filename+suffix)
            # print(src_filename, ks_id_, ids, ext)
            dst_filepath = os.path.join(out_dir, '{}_{}'.format(vidfolder_path, ks_id_)+ext)
            # print(src_filepath, dst_filepath)
            os.system('mv {} {}'.format(src_filepath, dst_filepath))
            time.sleep(0.001)
        os.system('rm -r {}'.format(os.path.join(folder, vidfolder_path)))


if __name__ == '__main__':
    flatten_folders('/data/mxt_data/ReLoBlur/train_folders', 'train')
    # flatten_folders('/data/mxt_data/LOLBlur/test/blur')
    # flatten_folders('/data/mxt_data/LOLBlur/test/sharp')
    # flatten_folders('./datasets/REDS/val/val_blur_jpeg')
    # flatten_folders('./datasets/REDS/val/val_sharp')
    # make_val_300('./datasets/REDS/val/val_blur_jpeg', './datasets/REDS/val/blur_300')
    # make_val_300('./datasets/REDS/val/val_sharp', './datasets/REDS/val/sharp_300')

    # create_lmdb_for_reds()


