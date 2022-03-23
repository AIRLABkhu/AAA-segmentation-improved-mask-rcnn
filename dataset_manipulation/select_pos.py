import numpy as np
import os
import cv2
import natsort
import shutil

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Downloads/1220/mask/'
    raw_path = '/home/bh/Downloads/1220/window/'

    mask_dst_path = '/home/bh/Downloads/1220/mask_pos/'
    raw_dst_path = '/home/bh/Downloads/1220/window_pos/'

    sub_list = natsort.natsorted(os.listdir(mask_path))


    for sub_idx in sub_list:
        print(sub_idx)
        file_list = list(natsort.natsorted(os.listdir(os.path.join(mask_path, sub_idx))))
        if not os.path.exists(os.path.join(mask_dst_path, sub_idx)):
            os.mkdir(os.path.join(mask_dst_path, sub_idx))
            os.mkdir(os.path.join(raw_dst_path, sub_idx))

        for file_idx in file_list:
            if file_idx == '.DS_Store': continue
            mask = cv2.imread(os.path.join(mask_path, sub_idx, file_idx), cv2.IMREAD_GRAYSCALE)
            mask_uniq = np.unique(mask)

            if len(mask_uniq) == 2:
               shutil.copyfile(os.path.join(raw_path,sub_idx,file_idx), os.path.join(raw_dst_path, sub_idx, file_idx))
               shutil.copyfile(os.path.join(mask_path,sub_idx,file_idx), os.path.join(mask_dst_path, sub_idx, file_idx))
