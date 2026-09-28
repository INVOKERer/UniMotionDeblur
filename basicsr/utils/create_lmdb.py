# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
import os
from os import path as osp
import time
import glob
# from misc import scandir
from lmdb_util import make_lmdb_from_imgs, make_lmdb_from_medical_imgs
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_reds():
    folder_path = '/data/mxt_data/REDS/train/blur'
    lmdb_path = '/data/mxt_data/REDS/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/REDS/train/sharp'
    lmdb_path = '/data/mxt_data/REDS/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_reloblur():
    folder_path = '/data/mxt_data/ReLoBlur/train/blur'
    lmdb_path = '/data/mxt_data/ReLoBlur/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/ReLoBlur/train/sharp'
    lmdb_path = '/data/mxt_data/ReLoBlur/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_lolblur():
    folder_path = '/data/mxt_data/LOLBlur/train/blur'
    lmdb_path = '/data/mxt_data/LOLBlur/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/LOLBlur/train/sharp'
    lmdb_path = '/data/mxt_data/LOLBlur/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_realblur():
    folder_path = '/data/mxt_data/RealBlur/RealBlur_R/train/blur'
    lmdb_path = '/data/mxt_data/RealBlur/RealBlur_R/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/RealBlur/RealBlur_R/train/sharp'
    lmdb_path = '/data/mxt_data/RealBlur/RealBlur_R/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/RealBlur/RealBlur_J/train/blur'
    lmdb_path = '/data/mxt_data/RealBlur/RealBlur_J/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/RealBlur/RealBlur_J/train/sharp'
    lmdb_path = '/data/mxt_data/RealBlur/RealBlur_J/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_gopro():
    folder_path = '/data/mxt_data/GoPro/train/blur'
    lmdb_path = '/data/mxt_data/GoPro/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/GoPro/train/sharp'
    lmdb_path = '/data/mxt_data/GoPro/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = '/data/mxt_data/GoPro/test/sharp'
    # lmdb_path = '/data/mxt_data/GoPro/test/sharp.lmdb'
    #
    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    #
    # folder_path = '/data/mxt_data/GoPro/test/blur'
    # lmdb_path = '/data/mxt_data/GoPro/test/blur.lmdb'
    #
    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_defocus():
    folder_path = '/data/mxt_data/DPDD/train/input_crops'
    lmdb_path = '/data/mxt_data/DPDD/train/input_crops.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/DPDD/train/target_crops'
    lmdb_path = '/data/mxt_data/DPDD/train/target_crops.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/SDD/train/input_crops'
    lmdb_path = '/data/mxt_data/SDD/train/input_crops.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/SDD/train/target_crops'
    lmdb_path = '/data/mxt_data/SDD/train/target_crops.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_NCIRD():
    folder_path = '/data/mxt_data/NCIRD/train/blur'
    lmdb_path = '/data/mxt_data/NCIRD/train/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = '/data/mxt_data/NCIRD/train/sharp'
    lmdb_path = '/data/mxt_data/NCIRD/train/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def prepare_keys_for_medical(folder_path, suffix='png', train_val='train', data_name='blur'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list_all = sorted(glob.glob(os.path.join(folder_path, '*', train_val, data_name, '*', '*', 'digi', '*.'+suffix)))
    # img_path_list = sorted(
    #     list(scandir(folder_path, suffix=suffix, recursive=False)))
    # keys = [os.path.basename(img_path).split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]
    filters = ['RuXianAi_HE', 'weiai_HE', 'xiaoXiBaoAi_HE', 'yiXianCHuanCi_DiffQuick']
    keys = []
    img_path_list = []
    for img_path in sorted(img_path_list_all):
        img_file = os.path.basename(img_path)
        file_dirs = img_path.split('/')
        a, b = file_dirs[-7], file_dirs[-4]
        if a in filters:
            continue

        key_name = img_file.split('.{}'.format(suffix))[0]
        if data_name == 'sharp':
            file_name_ = key_name.split('_')
            key_name = file_name_[0]
            if len(file_name_) > 2:
                for x in file_name_[1:-1]:
                    key_name = key_name + '_' + x
        key_name = a+'-'+b+'-'+key_name
        keys.append(key_name)
        img_path_list.append(img_path)
    return img_path_list, keys
def create_lmdb_for_medical_defocus():
    folder_path = '/data/mxt_data/MedicalDefocus'
    lmdb_path = '/data/mxt_data/MedicalDefocus_lmdb/blur.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys_for_medical(folder_path, 'png', train_val='train', data_name='blur')
    # print(img_path_list[0], keys[0])
    make_lmdb_from_medical_imgs(folder_path, lmdb_path, img_path_list, keys)
    #
    folder_path = '/data/mxt_data/MedicalDefocus'
    lmdb_path = '/data/mxt_data/MedicalDefocus_lmdb/sharp.lmdb'
    if os.path.exists(lmdb_path):
        os.system('rm -r {}'.format(lmdb_path))
        time.sleep(0.001)
    img_path_list, keys = prepare_keys_for_medical(folder_path, 'png', train_val='train', data_name='sharp')
    # print(img_path_list[0], keys[0])
    make_lmdb_from_medical_imgs(folder_path, lmdb_path, img_path_list, keys)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
def create_lmdb_for_rain13k():
    folder_path = './datasets/Rain13k/train/input'
    lmdb_path = './datasets/Rain13k/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/Rain13k/train/target'
    lmdb_path = './datasets/Rain13k/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_SIDD():
    folder_path = './datasets/SIDD/train/input_crops'
    lmdb_path = './datasets/SIDD/train/input_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/train/gt_crops'
    lmdb_path = './datasets/SIDD/train/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    #for val
    folder_path = './datasets/SIDD/val/input_crops'
    lmdb_path = './datasets/SIDD/val/input_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationNoisyBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationNoisyBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/val/gt_crops'
    lmdb_path = './datasets/SIDD/val/gt_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationGtBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationGtBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

if __name__ == '__main__':
    # create_lmdb_for_NCIRD()
    # create_lmdb_for_reloblur()
    # create_lmdb_for_lolblur()
    # create_lmdb_for_realblur()
    # create_lmdb_for_defocus()
    create_lmdb_for_medical_defocus()
