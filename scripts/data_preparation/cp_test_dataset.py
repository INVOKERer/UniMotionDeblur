import os
import yaml
from collections import OrderedDict
import time
from os import path as osp
from natsort import natsorted
def paired_paths_from_meta_info_file2(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
        len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    # lq_names = []
    # gt_names = []

    paths = []
    with open(meta_info_file, "r") as fin:
        for line in fin:
            elements = line.strip().split()
            if len(elements) == 2:
                # lq_names.append(elements[0])
                # gt_names.append(elements[1])
                input_path = osp.join(input_folder, elements[0])
                gt_path = osp.join(gt_folder, elements[1])
                paths.append(
                    dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
                )
        # lq_names = [line.strip().split(" ")[0] for line in fin]
        # gt_names = [line.strip().split(" ")[1] for line in fin]

    # paths = []
    # for lq_name, gt_name in zip(lq_names, gt_names):
    #     # basename, ext = osp.splitext(osp.basename(gt_name))
    #     # input_name = f"{filename_tmpl.format(basename)}{ext}"
    #     input_path = osp.join(input_folder, lq_name)
    #     gt_path = osp.join(gt_folder, gt_name)
    #     paths.append(
    #         dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
    #     )
    return paths
def paired_paths_from_folder(folders, keys, filename_tmpl, multi=False):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert (
            len(keys) == 2
    ), f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    # if multi:
    input_paths = natsorted(os.listdir(input_folder))
    # else:
    #     input_paths = list(scandir(input_folder))
    #     input_paths = list(scandir(input_folder))
    # gt_paths = list(scandir(gt_folder))
    gt_paths = natsorted(os.listdir(gt_folder))

    assert len(input_paths) == len(gt_paths), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_paths)}, {len(gt_paths)}."
    )
    paths = []
    input_paths_set = set(input_paths)
    for gt_path in gt_paths:

        basename, ext = osp.splitext(osp.basename(gt_path))

        # print(gt_path, filename_tmpl, f"{filename_tmpl.format(basename)}{ext}")
        if multi:
            input_name = f"{filename_tmpl.format(basename)}"
        else:
            input_name = f"{filename_tmpl.format(basename)}{ext}"
        if "_HR_" in input_name:
            input_name = input_name.replace("_HR_", "_LRBI_")
        input_path = osp.join(input_folder, input_name)
        assert (
                len(set([input_name]) & input_paths_set) == 1
        ), f"{input_name} is not in {input_key}_paths."
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
        )
    return paths
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper
def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, "r") as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])

opt_path = '/data/mxt_code/AllinOne/options/all_in_one/test/test_MotionDeblur.yml'
out_root = '/data/mxt_data/uniDeblur_dataset/test'
dataset_name = 'MotionBlur'
opt = yaml_load(opt_path)
paths_all = []
for phase, dataset_opt in opt["datasets"].items():
    # if phase.split("_")[0] == "train":
    if phase.split("_")[0] == "test":
        lq_folder = dataset_opt['dataroot_lq']
        gt_folder = dataset_opt['dataroot_gt']
        meta_file = dataset_opt.get("meta_info_file", None)
        # print(lq_path, gt_path)
        # input_paths = os.listdir(lq_path)
        if meta_file is not None:
            paths = paired_paths_from_meta_info_file2(
                [lq_folder, gt_folder],
                ["lq", "gt"],
                 meta_file,
                "{}",
            )
        else:
            multi = opt["multi"] if "multi" in opt else False
            paths = paired_paths_from_folder(
                [lq_folder, gt_folder],
                ["lq", "gt"],
                "{}",
                multi=False,
            )
        step = dataset_opt.get("step", None)
        if step is not None:
            paths = paths[::step]
        # paths_all.extend(paths)
        os.makedirs(os.path.join(out_root, dataset_name, dataset_opt['name'], 'blur'), exist_ok=True)
        os.makedirs(os.path.join(out_root, dataset_name, dataset_opt['name'], 'sharp'), exist_ok=True)
        meta_info_file = os.path.join(out_root, dataset_name, dataset_opt['name'] + '_' + 'meta_info.txt')
        with open(meta_info_file, 'a') as f:
            for path in paths:
                blur_image_name = path["lq_path"]
                sharp_image_name = path["gt_path"]
                blur_image_filename = os.path.basename(path["lq_path"])
                sharp_image_filename = os.path.basename(path["gt_path"])


                lq_path_dst = os.path.join(out_root, dataset_name, dataset_opt['name'], 'blur', blur_image_filename)
                gt_path_dst = os.path.join(out_root, dataset_name, dataset_opt['name'], 'sharp', blur_image_filename)
                os.system('cp {} {}'.format(blur_image_name, lq_path_dst))
                os.system('cp {} {}'.format(sharp_image_name, gt_path_dst))
                time.sleep(0.001)

                lq_meta_dst = os.path.join(dataset_opt['name'], 'blur', blur_image_filename)
                gt_meta_dst = os.path.join(dataset_opt['name'], 'sharp', blur_image_filename)
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