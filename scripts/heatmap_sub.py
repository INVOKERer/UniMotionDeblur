import glob
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# img_root = '/home/ubuntu/150t/personal_data/mxt/TPAMI/images/compare_select/RealBlur_R/109_208_413'

dir_root = '/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/uniDeblur/compare_select_supp/GoPro2'
img_roots = glob.glob(os.path.join(dir_root, '*'))
    # img_root = '/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/uniDeblur/compare_select_supp/GoPro_selected/GOPR0384_11_00-000029_256_128'
    # /home/ubuntu/150t/personal_data/mxt/TPMAI/images/compare_select/RealBlur_J/109_0_0
    # img_root = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/GoPro_image_patches/GoPro/416-1'
    # img_root = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/RealBlur_J_image_patches/RealBlur_J/198-3'
    # img_root = '/home/ubuntu/90t/personal_data/mxt/paper/CVPR2024/GoPro_images/img_diff_decoder/1026/360-960'
for img_root in img_roots:
    save_dir = os.path.join(img_root, 'img_sub')
    os.makedirs(save_dir, exist_ok=True)
    images = glob.glob(os.path.join(img_root, '*.png'))
    img_blur = cv2.imread(os.path.join(img_root, 'blur.png'), 0) / 255.
    img_sharp = cv2.imread(os.path.join(img_root, 'sharp.png'), 0) / 255.

    x_sub = np.abs(img_blur - img_sharp) #  # / 255.
    print(x_sub.max(), x_sub.min(), x_sub.shape)
    # x_sub_max = x_sub.max()
    # x_sub_min = x_sub.min()
    # x_sub_max = 0.3
    x_sub_max = 0.5
    x_sub_min = 0.
    out_way = os.path.join(save_dir, 'sub_sharp' + '_blur.png' )
    # cv2.imwrite(out_way, x_sub * 255.)

    sns_plot = plt.figure()
    sns.heatmap(x_sub, cmap='Reds', linewidths=0.0, vmin=x_sub_min, vmax=x_sub_max,
                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds_r .invert_yaxis()
    sns_plot.savefig(out_way, dpi=1000, pad_inches=0, bbox_inches='tight')
    plt.close()

    for i in images:
        img1 = cv2.imread(i, 0) / 255.
        file_name, ext = os.path.splitext(os.path.basename(i))
        # print(img1.max(), img_sharp.min())

        x_sub = np.abs(img1 - img_sharp) #  # / 255.
        print(x_sub.max(), x_sub.min())

        out_way = os.path.join(save_dir, 'sub_sharp' + '_' + file_name + ext)
        # cv2.imwrite(out_way, x_sub * 255.)

        sns_plot = plt.figure()
        sns.heatmap(x_sub, cmap='Reds', linewidths=0.0, vmin=x_sub_min, vmax=x_sub_max,
                    xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds_r .invert_yaxis()
        sns_plot.savefig(out_way, dpi=100, pad_inches=0, bbox_inches='tight')
        plt.close()