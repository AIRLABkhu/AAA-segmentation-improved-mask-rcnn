import os
import numpy as np
from shutil import copyfile, move
import cv2
import natsort

raw_dir = '/home/bh/Downloads/0906_modify_window_contrast/RAW_PNG/35139583_1/'

file_list = natsort.natsorted(os.listdir(raw_dir))

for idx in file_list:
    if idx=='.DS_Store': continue

    idx_split = idx.split('_')
    rename = idx_split[0]+'_'+idx_split[1][0]+'_'+idx_split[1][1:]
    os.rename(os.path.join(raw_dir, idx), os.path.join(raw_dir, rename))