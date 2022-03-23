import numpy as np
import os
import cv2
import natsort

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/mask/'
    file_list = natsort.natsorted(os.listdir(mask_path))

    data_arr = np.zeros((60,3)) # 0: number of mask images about subject, 1: mask start point, 2. mask finish point

    subject_idx = np.arange(1,61)

    count = 0 # mask count
    total_count = 0 # subject count

    file_pos_list = []
    total_pos_num_list = []

    for sub_idx in subject_idx:
        print(sub_idx)
        for file_idx in file_list:
            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:
                total_count = total_count + 1
                mask = cv2.imread(os.path.join(mask_path, file_idx))
                # Check positive sample
                pos_check = int(len(np.unique(mask)))
                if pos_check == 2:
                    count = count+1
                    file_pos_list.append(num)

        if (file_pos_list[-1]-file_pos_list[0]+1) != len(file_pos_list):
            print("=== ERROR ===")
            print(sub_idx)
            print("=== ERROR ===")
            break

        data_arr[(sub_idx-1),0] = total_count
        data_arr[(sub_idx-1),1] = file_pos_list[0]
        data_arr[(sub_idx-1),2] = file_pos_list[-1]
        total_pos_num_list.append(count)
        count = 0
        total_count = 0
        file_pos_list = []

np.save("./GT_512.npy",data_arr)
print("*"*50)
print(data_arr)
print(total_pos_num_list)
print(sum(total_pos_num_list))
print("==== Done ====")
