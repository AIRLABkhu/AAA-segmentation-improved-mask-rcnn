import os
import natsort
import cv2
import numpy as np
import csv

if __name__ == "__main__":
    raw_path = '/home/bh/Downloads/AAA-HOSEO-PNG-Update_8_12/AAA-HOSEO-PNG'
    blood_path = os.path.join(raw_path, 'Blood_PNG_RAW')
    stent_path = os.path.join(raw_path, 'Stent_PNG_RAW')
    roi_path = os.path.join(raw_path, 'ROI')

    list_file = natsort.natsorted(os.listdir(blood_path))
    list_file.pop() # Delete Ds.Store file

    blood_start = []
    blood_middle = []
    blood_finish = []
    stent_start = []
    stent_middle = []
    stent_finish = []
    roi_start = []
    roi_middle = []
    roi_finish =[]


    for file_idx in list_file:
        print(file_idx)
        blood_flag = 0
        stent_flag = 0
        roi_flag = 0



        list_png = natsort.natsorted(os.listdir(os.path.join(blood_path, file_idx)))
        if file_idx == "00302843_1" or file_idx == "14185273_1":
            list_png.pop()
        # print("%s %d" %(file_idx,len(list_png)))
        for png_idx in list_png:
            png_split = png_idx.split('_')
            blood_img = cv2.imread(os.path.join(blood_path, file_idx, png_idx), cv2.IMREAD_GRAYSCALE)

            png_split[2] = "stent"
            stent_img = cv2.imread(os.path.join(stent_path, file_idx, '_'.join(png_split)), cv2.IMREAD_GRAYSCALE)

            png_split[2] = "ROI"
            roi_img = cv2.imread(os.path.join(roi_path, file_idx, '_'.join(png_split)), cv2.IMREAD_GRAYSCALE)

            blood_check = len(np.unique(blood_img))
            stent_check = len(np.unique(stent_img))
            roi_check = len(np.unique(roi_img))

            if blood_check == 2 and blood_flag == 0:
                file_name = int(png_split[3].split('.')[0])
                blood_start.append(file_name)
                blood_flag = 1
            elif blood_check == 2 and blood_flag == 1:
                file_name = int(png_split[3].split('.')[0])
                blood_middle.append(file_name)

            if stent_check == 2 and stent_flag == 0:
                file_name = int(png_split[3].split('.')[0])
                stent_start.append(file_name)
                stent_flag = 1
            elif stent_check == 2 and stent_flag == 1:
                file_name = int(png_split[3].split('.')[0])
                stent_middle.append(file_name)

            if roi_check == 2 and roi_flag == 0:
                file_name = int(png_split[3].split('.')[0])
                roi_start.append(file_name)
                roi_flag = 1
            elif roi_check == 2 and roi_flag == 1:
                file_name = int(png_split[3].split('.')[0])
                roi_middle.append(file_name)

        blood_finish.append(blood_middle.pop())
        stent_finish.append(stent_middle.pop())
        roi_finish.append(roi_middle.pop())

    with open('./result.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(blood_start)
        write.writerow(blood_finish)
        write.writerow(stent_start)
        write.writerow(stent_finish)
        write.writerow(roi_start)
        write.writerow(roi_finish)


