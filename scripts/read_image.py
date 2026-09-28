import cv2
import glob
import os

image_root = '/data/mxt_data/NCIRD/train/sharp'
files = glob.glob(os.path.join(image_root, '*'))
for file in files:
    img = cv2.imread(file)
    print(img.max(), img.dtype, img.shape)