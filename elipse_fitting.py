import os
import numpy as np
import natsort
import cv2
import math



def fit_ellipse(mask, file_idx):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # find biggest contour
    # contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    # biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    biggest_contour = max(contours, key=cv2.contourArea)

    cnt = biggest_contour
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(m3, ellipse, (0, 255, 0), 1)
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
    cv2.line(m3, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (255, 0, 0), 1)

    x = int(xtop) - int(xbot)
    y = int(ytop) - int(ybot)
    ellipse_diameter = math.sqrt(math.pow(x,2) + math.pow(y,2))

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(m3, (x,y), (x+w, y+h), (0,0,255), 1)

    bigger_pixel_rect = w
    if bigger_pixel_rect < h:
         bigger_pixel_rect = h

    bigger_pixel_ellip = ellipse_diameter

    cv2.imwrite(os.path.join("/home/bh/Desktop/AAA_DATA/ellipse",file_idx), m3)

    return m3, bigger_pixel_rect, int(bigger_pixel_ellip)



if __name__ =="__main__":
    mask_path = "/home/bh/Downloads/aaa_segmentation/data/update/all/mask"
    mask_list = natsort.natsorted(os.listdir(mask_path))

    sub_list = np.arange(1,52)

    start = 0
    finish = 0

    whole = []

    for sub_idx in sub_list:
        pixel_list = []
        pixel_start = []
        pixel_finish = []
        for file_idx in mask_list:
            idx_split = int(file_idx.split("_")[0])
            if idx_split == sub_idx:
                mask_img = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)
                mask_check = int(len(np.unique(mask_img)))

                # only positive sample
                if mask_check == 2:
                    # elipse fitting with python
                    # https://www.kaggle.com/bguberfain/ellipse-fit
                    mask_with_ellipse, big_pixel_rec, big_pixel_ell = fit_ellipse(mask_img, file_idx)
                    pixel_list.append(big_pixel_ell)
                    whole.append(big_pixel_ell)

        start = start + int(pixel_list[0])
        finish = finish + int(pixel_list[-1])
        print(sub_idx)
        print(pixel_list)

    print(start/52)
    print(finish/52)

    print(whole)
    print("=== Done ===")
