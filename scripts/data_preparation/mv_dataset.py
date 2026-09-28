import os
import yaml
from collections import OrderedDict
import time
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

opt_path = '/data/mxt_code/AllinOne/options/all_in_one/train/train_NAFNet_AIO.yml'

opt = yaml_load(opt_path)

for phase, dataset_opt in opt["datasets"].items():
    # if phase.split("_")[0] == "train":
    if phase.split("_")[0] == "train":
        lq_path = dataset_opt['dataroot_lq']
        gt_path = dataset_opt['dataroot_gt']
        meta_file = dataset_opt.get("meta_info_file", None)
        print(lq_path, gt_path, meta_file)
        # if not os.path.exists(dst):
        lq_path_dst = os.path.dirname(lq_path).replace('/data/mxt_data', '/home/ubuntu/uniDeblurDataset')
        gt_path_dst = os.path.dirname(gt_path).replace('/data/mxt_data', '/home/ubuntu/uniDeblurDataset')
        # print(lq_path_dst, gt_path_dst)
        os.makedirs(lq_path_dst, exist_ok=True)
        os.makedirs(gt_path_dst, exist_ok=True)
        os.system('mv {} {}'.format(lq_path, lq_path_dst))
        time.sleep(0.001)
        os.system('mv {} {}'.format(gt_path, gt_path_dst))
        time.sleep(0.001)
        if meta_file is not None:
            os.system('cp {} {}'.format(meta_file, meta_file.replace('/data/mxt_data', '/home/ubuntu/uniDeblurDataset')))
            time.sleep(0.001)
    elif phase.split("_")[0] == "val":
        continue
# src_dir = '/data/mxt_data/GoPro/train/target_crops'
# dst_dir = '~/uniDeblurDataset/train/GoPro/train/target_crops'


# os.system('mv {} {}'.format(src_dir, dst_dir))