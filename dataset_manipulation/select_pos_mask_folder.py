import numpy as np
import os
import cv2
import natsort
from shutil import copyfile

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Downloads/0906_modify_window_contrast/ROI/'
    raw_path = '/home/bh/Downloads/0906_modify_window_contrast/RAW_PNG/'
    blood_path = '/home/bh/Downloads/0906_modify_window_contrast/Blood_PNG_RAW/'

    mask_dst_path = '/home/bh/Downloads/0906_modify_window_contrast/mask_deepaaa/'
    raw_dst_path = '/home/bh/Downloads/0906_modify_window_contrast/raw_deepaaa/'
    blood_dst_path = '/home/bh/Downloads/0906_modify_window_contrast/Blood_deepaaa/'

    file_list = natsort.natsorted(os.listdir(blood_path))

    count = 1
    for sub_idx in file_list:
        print(sub_idx)

        if sub_idx=='.DS_Store': continue

        if not os.path.exists(os.path.join(mask_dst_path, str(count))):
            os.makedirs(os.path.join(mask_dst_path, str(count)))
            os.makedirs(os.path.join(raw_dst_path, str(count)))
            os.makedirs(os.path.join(blood_dst_path, str(count)))

        sub_file_list = natsort.natsorted(os.listdir(os.path.join(mask_path, sub_idx)))
        count_file = 1
        for file_idx in sub_file_list:
            if file_idx=='.DS_Store': continue

            idx_split = file_idx.split('_')
            raw_file_idx = idx_split[0]+'_'+idx_split[1]+'_'+idx_split[3]
            blood_file_idx = idx_split[0]+'_'+idx_split[1]+'_blood_'+idx_split[3]

            file_name = '%d.png'%count_file

            mask = cv2.imread(os.path.join(mask_path, sub_idx, file_idx), cv2.IMREAD_GRAYSCALE)
            raw = cv2.imread(os.path.join(raw_path, sub_idx, raw_file_idx), cv2.IMREAD_GRAYSCALE)
            blood = cv2.imread(os.path.join(blood_path, sub_idx, blood_file_idx), cv2.IMREAD_GRAYSCALE)


            mask_256 = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_LINEAR)
            raw_256 = cv2.resize(raw, (128, 128), interpolation=cv2.INTER_LINEAR)
            blood_256 = cv2.resize(blood,(128, 128), interpolation=cv2.INTER_LINEAR)

            # if len(np.unique(mask)) == 2:
            blood_256[blood_256 > 127.5] = 255
            blood_256[blood_256 <= 127.5] = 0
            mask_256[mask_256 > 127.5] = 255
            mask_256[mask_256 <= 127.5] = 0


            cv2.imwrite(os.path.join(mask_dst_path, str(count), file_name), mask_256)
            cv2.imwrite(os.path.join(raw_dst_path, str(count), file_name), raw_256)
            cv2.imwrite(os.path.join(blood_dst_path, str(count), file_name), blood_256)


            count_file = count_file + 1


        count = count + 1

