import glob
import os
import random



root = '/data/mxt_data/uniDeblur_dataset/test/MotionBlur'

meta_info_file = os.path.join(root, 'MotionBlur_meta_info.txt')

# meta_info_file_c = ['Weiai_HE_meta_info', 'RuXianAi_HE_meta_info', 'taipan_HE_meta_info', 'xiaoXiBaoAi_HE_meta_info']
# meta_info_file_c = ['RuXianAi_ER_meta_info', 'RuXianAi_PR_meta_info', 'RuXianAi_ERPR_meta_info', 'weiai_CAM52_meta_info', 'weiai_CAM52_HE_meta_info', 'weiai_Ki67_meta_info']
# meta_info_file_c = ['GuJuLiuAi_H3F3A_meta_info', 'GuJuLiuAi_SATB2_meta_info', 'GuJuLiuAi_H3F3ASATB2_meta_info']
# meta_info_file_c = ['RealBlur_R_meta_info', 'RealBlur_J_meta_info']
meta_info_file_c = ['HIDE_meta_info', 'REDS_meta_info', 'RSBlur_meta_info']
with open(meta_info_file, 'a') as f:
    for meta_info_file_ in meta_info_file_c:
        with open(os.path.join(root, meta_info_file_+'.txt'), "r") as fin:
            for line in fin:
                elements = line.strip().split()
                if len(elements) == 2:
                    # lq_names.append(elements[0])
                    # gt_names.append(elements[1])
                    blur_image_name = elements[0]  # osp.join(input_folder, elements[0])
                    sharp_image_name = elements[1]  # osp.join(gt_folder, elements[1])
                    info = f'{blur_image_name} {sharp_image_name}'
                    # print(info)
                    f.write(f'{info}\n')
