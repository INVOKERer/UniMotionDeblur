import os
import glob
import time
import tqdm
root_pth = '/home/ubuntu/uniDeblurDataset'
dir_pths = glob.glob(os.path.join(root_pth, '*'))
out_pth = '/data/mxt_data/uniDeblur_dataset/train'
os.makedirs(out_pth, exist_ok=True)
for dir_pth in tqdm.tqdm(dir_pths):
    # if phase.split("_")[0] == "train":
    dir_name = os.path.basename(dir_pth)
    print('7z a {} {}'.format(os.path.join(out_pth, dir_name+'.7z'), dir_pth))
    os.system('7z a {} {}'.format(os.path.join(out_pth, dir_name+'.7z'), dir_pth))
    time.sleep(0.001)
# src_dir = '/data/mxt_data/GoPro/train/target_crops'
# dst_dir = '~/uniDeblurDataset/train/GoPro/train/target_crops'


# os.system('mv {} {}'.format(src_dir, dst_dir))