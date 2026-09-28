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


def flatten_folders(folder):
    vids = glob(os.path.join(folder, '*'))
    for vid in vids:
        vidfolder_path = os.path.basename(vid)
        # print(vidfolder_path)
        # if not os.path.exists(os.path.join(folder, vidfolder_path)):
        #     continue
        fids  = glob(os.path.join(vid, '*'))
        print('working on .. {} .. {}'.format(folder, vid))
        for fid in fids:
            # src_filename = '{:04}'.format(fid)
            src_filename = os.path.basename(fid)
            # suffixes = ['.jpg', '.png']
            # suffix = None
            #
            # for suf in suffixes:
            #     # print(os.path.join(folder, vidfolder_path, src_filename+suf))
            #     if os.path.exists(os.path.join(folder, vidfolder_path, src_filename+suf)):
            #         suffix = suf
            #         break
            # assert suffix is not None


            src_filepath = fid # os.path.join(folder, vidfolder_path, src_filename+suffix)
            dst_filepath = os.path.join(folder, '{}_{}'.format(vidfolder_path, src_filename))
            os.system('mv {} {}'.format(src_filepath, dst_filepath))
            time.sleep(0.001)
        os.system('rm -r {}'.format(os.path.join(folder, vidfolder_path)))


if __name__ == '__main__':
    flatten_folders('/data/mxt_data/LOLBlur/train/sharp')
    flatten_folders('/data/mxt_data/LOLBlur/train/blur')
    flatten_folders('/data/mxt_data/LOLBlur/test/blur')
    flatten_folders('/data/mxt_data/LOLBlur/test/sharp')
    # flatten_folders('./datasets/REDS/val/val_blur_jpeg')
    # flatten_folders('./datasets/REDS/val/val_sharp')
    # make_val_300('./datasets/REDS/val/val_blur_jpeg', './datasets/REDS/val/blur_300')
    # make_val_300('./datasets/REDS/val/val_sharp', './datasets/REDS/val/sharp_300')

    # create_lmdb_for_reds()


