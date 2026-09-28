import glob
import os
import random
from natsort import natsorted


img_blur_root = '/data/mxt_data/MCBLur/Defocus/test/blurry' # /hsi/view1/view1

# blank_dir = '/home/ubuntu/106-48t/personal_data/mxt/zjn-mxt/20230407-dataget/hsi/blank'

blur_images = natsorted(glob.glob(os.path.join(img_blur_root, '*.png')))

img_ahsrp_root = '/data/mxt_data/MCBLur/Defocus/test/sharp'

sharp_images = natsorted(glob.glob(os.path.join(img_ahsrp_root, '*.png')))

for sharp, blur in zip(sharp_images, blur_images):
    name1 = os.path.basename(sharp)
    name2 = os.path.basename(blur)
    # print(name1, name1)
    if name1 == name2:
        continue
    else:
        cp_command = 'mv {} {}'.format(blur, os.path.join(img_blur_root, name1))
        # print(cp_command)
        os.system(cp_command)
