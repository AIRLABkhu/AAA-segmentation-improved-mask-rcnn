import os
import natsort
import numpy as np
import cv2
from shutil import copyfile

if __name__ =="__main__":
    depth_list = []
    all_length = []

    pos_path = "/home/bh/PycharmProjects/3d_pytorch_bh/data/raw_256_pos/"

    all_path = "/home/bh/PycharmProjects/3d_pytorch_bh/data/mask_256/"
    all_raw_path = "/home/bh/PycharmProjects/3d_pytorch_bh/data/raw_256/"
    subj_list = list(natsort.natsorted(os.listdir(pos_path)))

    dst_mask_path = "/home/bh/PycharmProjects/3d_pytorch_bh/data/mask_256_70/"
    dst_raw_path = "/home/bh/PycharmProjects/3d_pytorch_bh/data/raw_256_70/"

    percentage = 0.7

    for subj_idx in subj_list:
        file_list = list(natsort.natsorted(os.listdir(os.path.join(pos_path, subj_idx))))
        all_list = list(natsort.natsorted(os.listdir(os.path.join(all_path, str(subj_idx)))))
        depth_list.append(len(file_list))
        all_length.append(len(all_list))

    #check percentage
    percent = (np.array(depth_list) / np.array(all_length)) * 100
    print(percent)
    print(np.mean(percent))

    print(all_length)
    depth_list = np.array(depth_list).astype(np.float)
    subj_percent_list = depth_list * (1/percentage)
    subj_percent_list = np.array(subj_percent_list).astype(np.uint32)

    # percent length over the num of input that subject_id using all
    subj_check = all_length - subj_percent_list
    print(subj_check)
    for idx, index in enumerate(subj_check):
        if index < 0:
            subj_check[idx] = 1
        else: subj_check[idx] = 0

    side_list = (subj_percent_list - depth_list) / 2
    side_list = np.array(side_list).astype(np.uint32)

    start_finish_list = np.zeros((60,2))

    for subj_idx in subj_list:

        file_list = list(natsort.natsorted(os.listdir(os.path.join(all_path, str(subj_idx)))))
        flag = 0
        for file_idx in file_list:
            raw = cv2.imread(os.path.join(all_raw_path, str(subj_idx), file_idx))
            mask = cv2.imread(os.path.join(all_path, str(subj_idx), file_idx))
            mask_uniq = np.unique(mask)

            if len(mask_uniq) == 2 and flag==0:
                start_finish_list[int(subj_idx)-1, 0] = int(file_idx.split(".")[0])
                flag = flag + 1
            elif len(mask_uniq) != 2 and flag==1:
                start_finish_list[int(subj_idx)-1, 1] = int(file_idx.split(".")[0])-1
                flag = flag=0

        subj_len = all_length[int(subj_idx)-1]

        front_flag = 0
        front_num = 0

        back_flag = 0
        back_num = 0
        print("Subject = %d, pos_start = %d, pos_finish = %d, side_list = %d, front_plus = %d, back_plus = %d, total = %d"
              %(int(subj_idx), start_finish_list[int(subj_idx)-1, 0], start_finish_list[int(subj_idx)-1, 1], side_list[int(subj_idx)-1],
                (start_finish_list[int(subj_idx) - 1, 0]-side_list[int(subj_idx)-1]), (start_finish_list[int(subj_idx) - 1, 1]+side_list[int(subj_idx)-1]), subj_len
                ))

        if subj_check[int(subj_idx)-1] == 1:
            start_finish_list[int(subj_idx)-1, 0] = 1
            start_finish_list[int(subj_idx)-1, 1] = subj_len

        else:
            if start_finish_list[int(subj_idx)-1,0] - side_list[int(subj_idx)-1] < 0:
                front_flag = 1
                front_num = abs(start_finish_list[int(subj_idx)-1,0] - side_list[int(subj_idx)-1])
                start_finish_list[int(subj_idx)-1, 0] = 1
            else:
                start_finish_list[int(subj_idx)-1, 0] = start_finish_list[int(subj_idx)-1, 0] - side_list[int(subj_idx)-1]

            if (start_finish_list[int(subj_idx)-1,1] + side_list[int(subj_idx)-1]) > subj_len:
                back_flag = 1
                back_num = abs(start_finish_list[int(subj_idx)-1,1] + side_list[int(subj_idx)-1] - subj_len)
                start_finish_list[int(subj_idx)-1, 1] = subj_len
            else: start_finish_list[int(subj_idx)-1,1] = start_finish_list[int(subj_idx)-1,1] + side_list[int(subj_idx)-1]

    print(all_length)
    print(start_finish_list)


    for subj_idx in subj_list:
        file_list = list(natsort.natsorted(os.listdir(os.path.join(all_path, str(subj_idx)))))
        count = 1
        if not os.path.exists(os.path.join(dst_raw_path, str(subj_idx))):
            os.mkdir(os.path.join(dst_raw_path, str(subj_idx)))
            os.mkdir(os.path.join(dst_mask_path, str(subj_idx)))



        for file_idx in file_list:
            if int(subj_idx) == 50:
                if int(file_idx.split(".")[0]) >= start_finish_list[int(subj_idx)-1,0]:

                    copyfile(os.path.join(all_raw_path, str(subj_idx), file_idx),
                             os.path.join(dst_raw_path, str(subj_idx), "%d.png" % count))
                    copyfile(os.path.join(all_path, str(subj_idx), file_idx),
                             os.path.join(dst_mask_path, str(subj_idx), "%d.png" % count))
                    count = count + 1

            elif int(file_idx.split(".")[0]) >= start_finish_list[int(subj_idx)-1,0] and int(file_idx.split(".")[0]) <= start_finish_list[int(subj_idx)-1,1]:

                copyfile(os.path.join(all_raw_path, str(subj_idx), file_idx), os.path.join(dst_raw_path, str(subj_idx), "%d.png"%count))
                copyfile(os.path.join(all_path, str(subj_idx), file_idx), os.path.join(dst_mask_path, str(subj_idx), "%d.png"%count))
                count = count+1
