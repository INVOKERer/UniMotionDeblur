import glob
import os
import random



img_blur_root = '/data/mxt_data/LFDOF/train_data' # /hsi/view1/view1
out_root = '/data/mxt_data/LFDOF/train_data'
os.makedirs(out_root, exist_ok=True)

# blank_dir = '/home/ubuntu/106-48t/personal_data/mxt/zjn-mxt/20230407-dataget/hsi/blank'

blur_images = glob.glob(os.path.join(img_blur_root, 'input', '*', '*.png'))


meta_info_file = os.path.join(out_root, 'meta_info_train.txt')


with open(meta_info_file, 'a') as f:
    for paths in blur_images:
        blur_image_name = os.path.basename(paths)
        sharp_image_name = paths.split('/')[-2] + '.png'
        info = f'{blur_image_name} {sharp_image_name}'
        # print(info)
        f.write(f'{info}\n')
