import os
import yaml
from collections import OrderedDict
import time
import glob

# opt_path = '/data/mxt_data/DPDD/train/target_error'
# lq_path_dir = '/data/mxt_data/DPDD/train/inputC'
# lq_path_dst = '/data/mxt_data/DPDD/train/inputC_error'
opt_path = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur/BSD3/sharp_error'
lq_path_dir = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur/BSD3/blur'
lq_path_dst = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur/BSD3/blur_error'
os.makedirs(lq_path_dst, exist_ok=True)
for file in glob.glob(os.path.join(opt_path, '*')):
    lq_path = os.path.join(lq_path_dir, os.path.basename(file))
    os.system('mv {} {}'.format(lq_path, lq_path_dst))
    time.sleep(0.001)

# opt_path = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur/MIORe/sharp_error'
# lq_path_dir = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur/MIORe/blur_error'
# lq_path_dst = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur/VAR-MIORe/sharp_error'
# os.makedirs(lq_path_dst, exist_ok=True)
# for file in glob.glob(os.path.join(opt_path, '*')):
#     lq_path = os.path.join(lq_path_dir, os.path.basename(file))
#     if os.path.exists(lq_path):
#         continue
#     else:
#         os.system('mv {} {}'.format(file, lq_path_dst))
#         time.sleep(0.001)
# src_dir = '/data/mxt_data/GoPro/train/target_crops'
# dst_dir = '~/uniDeblurDataset/train/GoPro/train/target_crops'


# os.system('mv {} {}'.format(src_dir, dst_dir))