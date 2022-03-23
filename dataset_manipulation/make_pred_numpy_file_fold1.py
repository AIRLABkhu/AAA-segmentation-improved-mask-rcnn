import numpy as np
import os
import cv2
import natsort
import math


# Draw elipsis on image
def fit_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # find biggest contour
    # contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    # biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    biggest_contour = max(contours, key=cv2.contourArea)

    cnt = biggest_contour
    ellipse = cv2.fitEllipse(cnt)

    # Calculate ellipse diameter
    (xc, yc), (d1, d2), angle = ellipse
    rmajor = max(d1, d2) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    xtop = xc + math.cos(math.radians(angle)) * rmajor
    ytop = yc + math.sin(math.radians(angle)) * rmajor
    xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
    ybot = yc + math.sin(math.radians(angle + 180)) * rmajor

    x = int(xtop) - int(xbot)
    y = int(ytop) - int(ybot)
    ellipse_diameter = math.sqrt(math.pow(x,2) + math.pow(y,2))
    bigger_pixel_ellip = ellipse_diameter

    x,y,w,h = cv2.boundingRect(cnt)

    bigger_pixel_rect = w
    if bigger_pixel_rect < h:
         bigger_pixel_rect = h


    return bigger_pixel_rect, int(bigger_pixel_ellip)

if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################

    mask_path = '/home/bh/Downloads/aaa_segmentation/data_visualization/result_220105_pos_0.5/'
    print(mask_path)

    # GT
    # mask_path = '/home/bh/Desktop/AAA_DATA_NEW/256/mask_all'

    file_list = natsort.natsorted(os.listdir(mask_path))
    file_list.pop()

    data_arr = np.zeros((15,300))

    subject_idx = np.arange(1,16)

    count = 0

    for sub_idx in subject_idx:
        ellipse = []
        print(sub_idx)
        for file_idx in file_list:

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:
                mask = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)
                # Check positive sample
                pos_check = int(len(np.unique(mask)))

                if pos_check == 2:

                    data_arr[(sub_idx - 1), count] = 1

                    # ### ellipse check
                    # rect, ellip_diameter = fit_ellipse(mask)
                    # ellipse.append(ellip_diameter)
                    #
                    # if ellip_diameter >=18:
                    #     data_arr[(sub_idx-1),count] = 1

                count = count + 1
        # print(data_arr[(sub_idx-1)])
        # print(ellipse)
        count = 0


    np.save("./predict_default",data_arr)
    print("*"*50)
    print("==== Done ====")

