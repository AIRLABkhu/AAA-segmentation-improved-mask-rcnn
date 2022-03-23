import os
import numpy as np
from shutil import copyfile, move
import cv2
import natsort

mask_dir = '/home/bh/Downloads/1220/mask/'
raw_dir = '/home/bh/Downloads/1220/full/'
window_dir = '/home/bh/Downloads/1220/window/'

mask_dst_dir = '/home/bh/Downloads/1220/rename_mask/'
raw_dst_dir = '/home/bh/Downloads/1220/rename_full/'
window_dst_dir = '/home/bh/Downloads/1220/rename_window/'

list_folder = natsort.natsorted(os.listdir(mask_dir))

count = 1


for folder_idx in list_folder:
    if folder_idx == '.DS_Store': continue

    sub_mask_path = os.path.join(mask_dir, folder_idx)
    sub_raw_path = os.path.join(raw_dir, folder_idx)
    sub_window_path = os.path.join(window_dir, folder_idx)

    list_file = natsort.natsorted(os.listdir(sub_mask_path))

    for list_idx in list_file:
        if list_idx == '.DS_Store': continue

        split_idx = list_idx.split('_')
        rename = str(count) +"_"+ split_idx[2]

        copyfile(os.path.join(sub_mask_path, list_idx), os.path.join(mask_dst_dir, rename))
        copyfile(os.path.join(sub_raw_path, list_idx), os.path.join(raw_dst_dir, rename))
        copyfile(os.path.join(sub_window_path, list_idx), os.path.join(window_dst_dir, rename))

    count = count + 1


    # # print(idx)
    # # pre = idx[:16]
    # # post = idx[-8:]
    # # name = pre + '_' + post
    # # print(name)
    # # os.rename(os.path.join(mask_dir,"35882023_1",idx), os.path.join(mask_dir,"35882023_1",name))
    #
    # if pre == int(idx_spilt[0]):
    #     rename = str(count)+"_"+str(idx_spilt[1])
    #
    # else:
    #     pre = int(idx_spilt[0])
    #     count = count + 1
    #     rename = str(count)+"_"+str(idx_spilt[1])
    #
    # copyfile(os.path.join(mask_dir, idx), os.path.join(mask_dst_dir, rename))
    # copyfile(os.path.join(raw_dir, idx), os.path.join(raw_dst_dir, rename))

    # if int(idx_spilt[0]) not in [1,13,27,43,49]:
    #     copyfile(os.path.join(mask_dir, idx), os.path.join(mask_dst_dir, idx))
    #     copyfile(os.path.join(raw_dir, idx), os.path.join(raw_dst_dir, idx))

#     mask = cv2.imread(os.path.join(mask_dir, idx), cv2.IMREAD_GRAYSCALE)
#     raw = cv2.imread(os.path.join(raw_dir, idx), cv2.IMREAD_GRAYSCALE)
#
#     # print(len(np.unique(mask)))
#     # if len(np.unique(mask))>1:
#     #     copyfile(os.path.join(mask_dir, idx), os.path.join(mask_dst_dir, idx))
#     #     copyfile(os.path.join(raw_dir, idx), os.path.join(raw_dst_dir, idx))
#
#
#     mask_256 = cv2.resize(mask, (256,256), interpolation=cv2.INTER_LINEAR)
#     raw_256 = cv2.resize(raw, (256,256), interpolation=cv2.INTER_LINEAR)
#
#     # print(np.unique(mask_256))
#     mask_256[mask_256 > 127] = 255
#     mask_256[mask_256 <= 127] = 0
#     # print(np.unique(mask_256))
#     #
#     #
#     cv2.imwrite(os.path.join(mask_dst_dir, idx), mask_256)
#     cv2.imwrite(os.path.join(raw_dst_dir, idx), raw_256)
#     #
#     # # for file in file_mask:
#     # #     file_name = "%d_"%(int(idx+1))+file
#     # #     copyfile(os.path.join(mask_dir, subject, str(file)), os.path.join(mask_dst_dir,file_name))
#     # #     copyfile(os.path.join(raw_dir, subject, str(file)), os.path.join(raw_dst_dir,file_name))
# print("=== DONE ===")
# num_mask = natsort.natsorted(os.listdir(mask_dst_dir))
# print(len(num_mask))
