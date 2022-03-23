import os

import numpy as np
import cv2
import natsort

raw_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/raw/'
mask_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/mask/'
dst_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/result_1/'

file_list = natsort.natsorted(os.listdir(raw_path))

comp_img = cv2.imread("/home/bh/Downloads/3_o.png", cv2.IMREAD_COLOR)

for file_idx in file_list:
    mask = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)
    if len(np.unique(mask)) == 2:
        img = cv2.imread(os.path.join(raw_path, file_idx), cv2.IMREAD_COLOR)
        mask_rgb = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_COLOR)
        add_raw_img = 0.5*comp_img + 0.5*img
        add_mask_img = 0.5*comp_img + 0.5*mask_rgb
        img_all = np.concatenate([add_raw_img, add_mask_img, mask_rgb], axis=1)
        cv2.imwrite(os.path.join(dst_path, file_idx), img_all)
        print(file_idx)

