import os
import tqdm
import time

def separate_elements(file_path):
    list1 = []
    list2 = []

    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 2:
                list1.append(elements[0])
                list2.append(elements[1])

    return list1, list2
def flatten_folders(fids, folder):
    os.makedirs(folder, exist_ok=True)
    # for vid in vids:
    #     vidfolder_path = os.path.basename(vid)
    #     # print(vidfolder_path)
    #     # if not os.path.exists(os.path.join(folder, vidfolder_path)):
    #     #     continue
    #     fids  = glob(os.path.join(vid, '*'))
    #     print('working on .. {} .. {}'.format(folder, vid))
    for fid in tqdm.tqdm(fids):
        # src_filename = '{:04}'.format(fid)
        path_splits = fid.split('/')
        a, b = path_splits[-4], path_splits[-3]
        src_filename = os.path.basename(fid)
        filename, ext = os.path.splitext(src_filename)
        # suffixes = ['.jpg', '.png']
        # suffix = None
        #
        # for suf in suffixes:
        #     # print(os.path.join(folder, vidfolder_path, src_filename+suf))
        #     if os.path.exists(os.path.join(folder, vidfolder_path, src_filename+suf)):
        #         suffix = suf
        #         break
        # assert suffix is not None


        src_filepath = fid # os.path.join(folder, vidfolder_path, src_filename+suffix)
        dst_filepath = os.path.join(folder, '{}_{}'.format(a, b)+ext)
        # print('mv {} {}'.format(src_filepath, dst_filepath))
        os.system('mv {} {}'.format(src_filepath, dst_filepath))
        time.sleep(0.001)
    # os.system('rm -r {}'.format(os.path.join(folder, vidfolder_path)))

def main_dataset_rsblur(data_path, train_path, test_path):

    PATH_TRAIN = os.path.join(data_path, 'RSBlur_real_train.txt')

    PATH_VALID = os.path.join(data_path, 'RSBlur_real_test.txt')

    paths_sharp_train, paths_blur_train = separate_elements(PATH_TRAIN)
    paths_sharp_valid, paths_blur_valid = separate_elements(PATH_VALID)

    
    list_blur_train = [os.path.join(train_path, path) for path in paths_blur_train]
    list_sharp_train = [os.path.join(train_path, path) for path in paths_sharp_train]

    list_blur_valid = [os.path.join(test_path, path) for path in paths_blur_valid]
    list_sharp_valid = [os.path.join(test_path, path) for path in paths_sharp_valid]     
    print(len(list_blur_train), len(list_blur_valid))
    flatten_folders(list_blur_train, os.path.join(train_path, 'train', 'blur'))
    flatten_folders(list_sharp_train, os.path.join(train_path, 'train', 'sharp'))
    flatten_folders(list_blur_valid, os.path.join(train_path, 'test', 'blur'))
    flatten_folders(list_sharp_valid, os.path.join(train_path, 'test', 'sharp'))

if __name__ == '__main__':
    
    PATH_TEST = '/data/mxt_data/RSBlur'

    main_dataset_rsblur(data_path=PATH_TEST, train_path=PATH_TEST,test_path=PATH_TEST)

