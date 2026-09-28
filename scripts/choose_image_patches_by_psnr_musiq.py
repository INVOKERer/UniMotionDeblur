import numpy as np
import math
from skimage.metrics import structural_similarity as SSIM
import cv2
import glob
from tqdm import tqdm
import shutil
import os
import shutil
import random
import json
import torch
import pyiqa

# device = 'cuda'
# iqa_metrics = {
#         'MUSIQ': pyiqa.create_metric('musiq', device=device)
#     }
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_metrics(fake, real) -> (float, float):
    psnr = PSNR(fake, real)
    # ssim = SSIM(fake, real, multichannel=True)
    # for name, metric in iqa_metrics.items():
    #     sr_tensor = img2tensor(fake, bgr2rgb=False, float32=True).unsqueeze(0).to(device).contiguous() / 255.0
    #     if name in ['NIMA', 'CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA', 'PAQ2PIQ']:
    #         musiq = metric(sr_tensor).item()
    return psnr # , ssim

def get_psnr_x(j_path, im2, force_to_compute_PSNR):
    file_name_ext = os.path.basename(j_path)
    file_name_c, ext = os.path.splitext(file_name_ext)
    file_l = file_name_c.split('_')
    if len(file_l) > 1 and not force_to_compute_PSNR:
        psnr_c = float(file_l[-1])
    else:
        # img_sharp_way = sharp_Way + '/' + file_l[0]+'.png'
        # if not os.path.exists(img_sharp_way):
        #     raise FileExistsError
        im1 = cv2.imread(j_path)
        # im2 = cv2.imread(img_sharp_way)
        psnr_c = get_metrics(im1, im2)
        psnr_c = float(str(psnr_c)[:5])
    return psnr_c
if __name__ == '__main__':

    out_root = '/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/uniDeblur/compare_select_supp'
    ddataset_root = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur'
    # ddataset_root = '/home/ubuntu/106-48t/personal_data/mxt/Datasets/Defocus'
    compared_root = '/home/ubuntu/150t/personal_data/mxt/MXT/ckpts/uniDeblur'
    ncompared_root = '/home/ubuntu/150t/personal_data/mxt/MXT/ckpts/uniDeblur'
    # compared_root = '/home/ubuntu/106-48t/personal_data/mxt/exp_results/images_defocus'
    Image_root = '/home/ubuntu/150t/personal_data/mxt/MXT/ckpts/uniDeblur'
    methods = 'uniDeblur-L-k123'
    not_compared_list = ['UniDeblur-B-k123']# ['RevID-B_GoPro']
    # compared_list = ['NAFNet64', 'Restormer', 'FMIMOUNetPLUS_gopro', 'MIMO-UNetPLUS_ori_gopro', 'MPRNet_gopro']
    compared_list = ['NAFNet64', 'LoFormer', 'PromptIR', 'MoCEIR', 'BioIR', 'EVSSM'] # GoPro-HIDE
    # compared_list = ['results_lama', 'results_ddtrans', 'results_fcn'] # RealBlur
    # compared_list = ['HINet', 'NAFNet64']
    # compared_type = 'REDS'
    # compared_type = 'GoPro'
    compared_types = ['BSD1', 'BSD2']
    # compared_types = ['GoPro', 'HIDE', 'RealBlur_R', 'RealBlur_J', 'REDS', 'RSBlur', 'LOLBlur', 'BSD1', 'BSD2', 'MIORe', 'VAR-MIORe']
    # compared_type = 'HIDE' # 'mayo128' #8
    out_cx = methods


    # img_list_B = glob.glob(fake_Way + '/*')
    # img_list_A.sort()
    # img_list_B.sort()
    # '/home/mxt/106-48t/personal_data/mxt/Datasets/Defocus/DPDD/test/sharp/1P0A0917.png'
    for compared_type in compared_types:
        # Image_Way = os.path.join(Image_root, methods)
        # print(Image_Way)
        img_list_A = glob.glob(os.path.join(Image_root, methods, 'visualization', compared_type, '*.png'))
        force_to_compute_PSNR = True

        out_dict = {}
        thresh_hold = 0.5  # 0.2# 0.45
        end = 20
        psnr_list = []
        musiq_list = []
        random.shuffle(img_list_A)
        for i, file in tqdm(enumerate(img_list_A[:end])):
            # print(img_list_A[i].split('/')[-1])
            out_dir = os.path.join(out_root, compared_type)
            # if compared_type == 'REDS':
            #     sharp_Way = os.path.join(ddataset_root, compared_type, 'sharp_300')
            #     blur_Way = os.path.join(ddataset_root, compared_type, 'blur_300')
            # if 'RealBlur' in compared_type:
            #     sharp_Way = os.path.join(ddataset_root, 'RealBlur', compared_type, 'test', 'sharp')
            #     blur_Way = os.path.join(ddataset_root, 'RealBlur', compared_type, 'test', 'blur')
            # else:
            sharp_Way = os.path.join(ddataset_root, compared_type, 'sharp')
            blur_Way = os.path.join(ddataset_root, compared_type, 'blur')
            file_name_ext = os.path.basename(file)
            file_name, ext = os.path.splitext(file_name_ext)
            # print(file_name_ext)
            file_l = file_name.split('_')
            # file_name_x = file_l[0]
            # for x in file_l[1:-2]:
            #     file_name_x = file_name_x + '_' + x
            dir_name_out = file_name # if compared_type == 'REDS' else file_name_x
            # print(dir_name_out)
            # print(os.path.join(sharp_Way, '*'+dir_name_out + '*'))
            img_sharp_way = os.path.join(sharp_Way, file_name_ext) # glob.glob(os.path.join(sharp_Way, file_name_ext))+glob.glob(os.path.join(sharp_Way, dir_name_out + '*'))
            # img_sharp_way = img_sharp_ways[0]
            # img_sharp_way = os.path.join(sharp_Way, file_name_ext)
            # print(img_sharp_way)
            if not os.path.exists(img_sharp_way):
                print(img_sharp_way)
                raise FileExistsError
            hr_img = cv2.imread(img_sharp_way)
            h, w = hr_img.shape[:2]
            patch_size = 320
            overlap = 64
            p_max = 0
            if w > p_max and h > p_max:
                w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=np.int32))
                h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=np.int32))
                w1.append(w - patch_size)
                h1.append(h - patch_size)
                random.shuffle(h1)
                random.shuffle(w1)
                for i_w in w1[:4]:
                    for j_h in h1[:4]:
                        # if len(file_l) > 1 and not force_to_compute_PSNR:
                        #     psnr_c = float(file_l[-1])
                        # else:

                        im1 = cv2.imread(file)
                        im2 = cv2.imread(img_sharp_way)
                        # print(img_sharp_way, img_list_A[i], im1.shape, im2.shape)
                        im1 = im1[j_h:j_h + patch_size, i_w:i_w + patch_size, :]
                        im2 = im2[j_h:j_h + patch_size, i_w:i_w + patch_size, :]
                        # print(img_list_A[i], img_sharp_way)
                        psnr_c = get_metrics(im1, im2)
                        psnr_c = float(str(psnr_c)[:5])
                        out_dict[out_cx] = psnr_c
                        # out_c_name = out_cx + str(psnr_c)
                        out_way = os.path.join(out_dir, dir_name_out+'_'+str(j_h)+'_'+str(i_w))
                        out_c_path = os.path.join(out_way, out_cx + ext) #  + '_' + str(psnr_c)
                        j_list = []
                        out_j_list = []
                        for j in compared_list:
                            # print(os.path.join(compared_root, j, compared_type, dir_name_out))
                            # print()
                            j_path = os.path.join(compared_root, j, 'visualization', compared_type, dir_name_out+'.png')
                            # j_path = glob.glob(os.path.join(compared_root, j, compared_type, dir_name_out+'.png'))+\
                            #          glob.glob(os.path.join(compared_root, j, compared_type, dir_name_out+'*'))
                            # print(compared_root, j, compared_type, dir_name_out+'.png')
                            # j_path = j_path[0]
                            im_j = cv2.imread(j_path)
                            im_j = im_j[j_h:j_h + patch_size, i_w:i_w + patch_size, :]
                            # print(j_path, sharp_Way)
                            psnr_j = get_metrics(im_j, im2)
                            # print(psnr_j-psnr_c)
                            if psnr_c-psnr_j > thresh_hold:
                                j_list.append(j_path)
                                _, j_ext = os.path.splitext(j_path)
                                # print(psnr_j)
                                out_j_list.append(os.path.join(out_way, j + j_ext)) #  + '_' + str(psnr_j)
                            else:
                                break
                            out_dict[j] = psnr_j
                        # print(j_list, dir_name_out, out_j_list)
                        if len(j_list) == len(compared_list):
                            # print(dir_name_out)
                            os.makedirs(os.path.join(out_way), exist_ok=True)
                            # img_blur_ways = glob.glob(os.path.join(blur_Way, dir_name_out + '.png')) + glob.glob(os.path.join(compared_root, dir_name_out + '*'))
                            # img_blur_way = img_blur_ways[0]
                            img_blur_way = os.path.join(blur_Way, dir_name_out + '.png')
                            # img_blur_way = glob.glob(os.path.join(blur_Way, dir_name_out + '*'))[0]
                            # print(img_blur_way)
                            _, blur_ext = os.path.splitext(img_blur_way)
                            _, sharp_ext = os.path.splitext(img_sharp_way)
                            im1 = cv2.imread(img_blur_way)[j_h:j_h + patch_size, i_w:i_w + patch_size, :]
                            im2 = cv2.imread(img_sharp_way)[j_h:j_h + patch_size, i_w:i_w + patch_size, :]
                            psnr_b = get_metrics(im1, im2)
                            psnr_b = str(psnr_b)[:5]
                            out_dict['blur'] = psnr_b
                            nj_list = []
                            out_nj_list = []
                            for nj in not_compared_list:
                                # nj_path = glob.glob(os.path.join(ncompared_root, nj, compared_type, dir_name_out + '.png')) + \
                                #          glob.glob(os.path.join(ncompared_root, nj, compared_type, dir_name_out + '_*.png'))
                                # # print(os.path.join(ncompared_root, nj, compared_type, dir_name_out + '.png'))
                                # nj_path = nj_path[0]
                                nj_path = os.path.join(ncompared_root, nj, 'visualization', compared_type, dir_name_out + '.png')
                                im_nj = cv2.imread(nj_path)
                                im_nj = im_nj[j_h:j_h + patch_size, i_w:i_w + patch_size, :]
                                # print(j_path, sharp_Way)
                                psnr_nj = get_metrics(im_nj, im2)
                                # psnr_nj = get_psnr_x(nj_path, im2, force_to_compute_PSNR)
                                nj_list.append(nj_path)
                                _, nj_ext = os.path.splitext(nj_path)
                                # print(psnr_j)
                                out_nj_list.append(os.path.join(out_way, nj + nj_ext))
                                out_dict[nj] = psnr_nj

                            out_blur_path = os.path.join(out_way, 'blur'+blur_ext) # +'_'+str(psnr_b)
                            out_sharp_path = os.path.join(out_way, 'sharp'+sharp_ext)

                            cv2.imwrite(out_sharp_path, im2)
                            cv2.imwrite(out_blur_path, cv2.imread(img_blur_way)[j_h:j_h + patch_size, i_w:i_w + patch_size, :])
                            cv2.imwrite(out_c_path, cv2.imread(img_list_A[i])[j_h:j_h + patch_size, i_w:i_w + patch_size, :])

                            for z in range(len(j_list)):
                                # mlz = 'sudo cp -r ' + j_list[z] + ' ' + out_j_list[z]
                                cv2.imwrite(out_j_list[z],
                                            cv2.imread(j_list[z])[j_h:j_h + patch_size, i_w:i_w + patch_size, :])
                                # os.system(mlz)
                            for z in range(len(nj_list)):
                                # mlz = 'sudo cp -r ' + nj_list[z] + ' ' + out_nj_list[z]
                                # os.system(mlz)
                                cv2.imwrite(out_nj_list[z],
                                            cv2.imread(nj_list[z])[j_h:j_h + patch_size, i_w:i_w + patch_size, :])
                                # shutil.copy(j_list[z], out_j_list[z])
                                # print(j_list[z], out_j_list[z])
                            tf = open(os.path.join(out_way, "psnr.json"), "w")
                            json.dump(out_dict, tf)
                            tf.close()