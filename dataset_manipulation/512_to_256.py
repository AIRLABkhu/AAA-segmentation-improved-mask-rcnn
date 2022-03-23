import os
import numpy as np
from shutil import copyfile, move
import cv2
import natsort

mask_dir = '/home/bh/Downloads/aaa_segmentation/data/1220_window/mask/'
raw_dir = '/home/bh/Downloads/aaa_segmentation/data/1220_window/raw/'

mask_dst_dir = '/home/bh/Downloads/aaa_segmentation/data/1220_window_256/mask/'
raw_dst_dir = '/home/bh/Downloads/aaa_segmentation/data/1220_window_256/raw/'


list_file = natsort.natsorted(os.listdir(mask_dir))
print(len(list_file))

for idx in list_file:

    mask = cv2.imread(os.path.join(mask_dir, idx), cv2.IMREAD_GRAYSCALE)
    raw = cv2.imread(os.path.join(raw_dir, idx), cv2.IMREAD_GRAYSCALE)

    # print(len(np.unique(mask)))
    # if len(np.unique(mask))>1:
    #     copyfile(os.path.join(mask_dir, idx), os.path.join(mask_dst_dir, idx))
    #     copyfile(os.path.join(raw_dir, idx), os.path.join(raw_dst_dir, idx))

    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)
    raw_256 = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_LINEAR)

    # print(np.unique(mask_256))
    mask_256[mask_256 > 127] = 255
    mask_256[mask_256 <= 127] = 0
    # print(np.unique(mask_256))

    # os.makedirs(mask_dst_dir, exist_ok=True)
    # os.makedirs(raw_dst_dir, exist_ok=True)

    cv2.imwrite(os.path.join(mask_dst_dir, idx), mask_256)
    cv2.imwrite(os.path.join(raw_dst_dir, idx), raw_256)

