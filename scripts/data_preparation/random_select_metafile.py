import glob
import os
import random

img_blur_root = '/data/mxt_data/MedicalDefocus' # /hsi/view1/view1
out_root = '/data/mxt_data/MedicalDefocus'
os.makedirs(out_root, exist_ok=True)

# blank_dir = '/home/ubuntu/106-48t/personal_data/mxt/zjn-mxt/20230407-dataget/hsi/blank'

slide_clss = glob.glob(os.path.join(img_blur_root, '*'))
for slide_cls in slide_clss:
    meta_info_file = os.path.join(slide_cls, 'test', 'meta_info_test.txt')
    meta_info_file_out = os.path.join(slide_cls, 'test', 'meta_info_test2.txt')
    paths_all = []
    start = 0
    end = 48
    with open(meta_info_file, "r") as fin:
        for line in fin:
            if start == 0:
                paths = []
            start += 1
            elements = line.strip().split()
            if len(elements) == 2:
                # lq_names.append(elements[0])
                # gt_names.append(elements[1])
                input_path = elements[0] # osp.join(input_folder, elements[0])
                gt_path = elements[1] # osp.join(gt_folder, elements[1])
                paths.append([input_path, gt_path])
            if start == end:
                start = 0
                paths_all.append(paths)
                # print(len(paths))
                # print(paths)
    # for paths in paths_all:
    #     path_selects = random.choices(paths, k=5)
    #     print(path_selects)
    #     for path_select in path_selects:
    #         info = f'{path_select[0]} {path_select[1]}'
    with open(meta_info_file_out, 'a') as f:
        for paths in paths_all:
            path_selects = random.choices(paths, k=1)
            # print(path_selects)
            for path_select in path_selects:
                info = f'{path_select[0]} {path_select[1]}'
                # print(info)
                f.write(f'{info}\n')
