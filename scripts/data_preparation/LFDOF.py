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
import tqdm

def flatten_folders(folder):
    vids = glob(os.path.join(folder, '*'))
    for vid in vids:
        vidfolder_path = os.path.basename(vid)
        # print(vidfolder_path)
        # if not os.path.exists(os.path.join(folder, vidfolder_path)):
        #     continue
        fids  = glob(os.path.join(vid, '*'))
        print('working on .. {} .. {}'.format(folder, vid))
        for fid in tqdm.tqdm(fids):
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
            dst_filepath = os.path.join(folder, src_filename) # '{}_{}'.format(vidfolder_path, src_filename))
            # print('mv {} {}'.format(src_filepath, dst_filepath))
            os.system('mv {} {}'.format(src_filepath, dst_filepath))
            time.sleep(0.001)
        os.system('rm -r {}'.format(os.path.join(folder, vidfolder_path)))


if __name__ == '__main__':
    flatten_folders('/data/mxt_data/LFDOF/test_data/input')
    flatten_folders('/data/mxt_data/LFDOF/train_data/input')


