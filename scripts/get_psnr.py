import os.path

import numpy as np
import math
from skimage.metrics import structural_similarity as SSIM
import cv2
import glob
from tqdm import tqdm

def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_metrics(fake, real) -> (float, float):
    psnr = PSNR(fake, real)
    ssim = 0.
    return psnr, ssim


if __name__ == '__main__':
    real_way = '/data/mxt_data/uniDeblur_dataset/test/DefocusBlur/DPDD/val_sharp'
    fake_Way = '/data/mxt_code/AllinOne/results/AblationDefocus_NAFNet48_ema-5e-4_decay1e-2_val_10/visualization/DPDD'
    out_dir = '/data/mxt_code/AllinOne/results/AblationDefocus_NAFNet48_ema-5e-4_decay1e-2_val_10/visualization/DPDD_PSNR'
    os.makedirs(out_dir, exist_ok=True)
    # real_way = '/media/mxt/9a254338-2f19-427f-8f68-6d041ef327d9/deblur/dataset/test/CAM_sharp'
    # fake_Way = '/media/mxt/9a254338-2f19-427f-8f68-6d041ef327d9/deblur/MPRNet-main/Deblurring/results/MPRNet_HE/CAM'
    img_list_A = glob.glob(real_way + '/*')
    # img_list_B = glob.glob(fake_Way + '/*')
    img_list_A.sort()
    # img_list_B.sort()
    psnr_list = []
    ssim_list = []
    for i in tqdm(range(len(img_list_A))):
        # print(img_list_A[i].split('/')[-1], img_list_B[i].split('/')[-1])
        img_list_B_way = fake_Way + '/' + img_list_A[i].split('/')[-1]
        file_name = os.path.basename(img_list_A[i])
        file_name, ext = os.path.splitext(file_name)
        realim = cv2.imread(img_list_A[i])
        # realim = cv2.resize(realim, [640, 640])
        fakeim = cv2.imread(img_list_B_way)
        # iou_s = diceCoeff_score(outim, gtim)
        psnr, ssim = get_metrics(realim, fakeim)
        cv2.imwrite(os.path.join(out_dir, file_name+'_'+str(psnr)[:4]+ext), fakeim)

