import cv2
import numpy as np
import os
import natsort

def ImageBlending(img_path, gt_path, dst_path):

    file_list = natsort.natsorted(os.listdir(img_path))
    for file_idx in file_list:
        print(file_idx)
        img = cv2.imread(os.path.join(img_path, file_idx))
        mask = cv2.imread(os.path.join(gt_path, file_idx))

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Define sufficient enough colors for blobs
        colors = [(0, 102, 255), (0, 102, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]

        # Draw all contours, and their children, with different colors
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        k = -1
        # print(hier)
        for i, cnt in enumerate(contours):
            if (hier[0, i, 3] == -1):
                k += 1
            cv2.drawContours(img, [cnt], -1, colors[k], 2)
            # cv2.drawContours(im1, [cnt], -1, colors[k], thickness=cv2.FILLED)


        # img = cv2.addWeighted(im1, 1, boundary , 0.2, 0)

        cv2.imwrite(os.path.join(dst_path, file_idx), img)


img_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/raw/'
mask_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/mask/'
dst_path = '/home/bh/Downloads/aaa_segmentation/data/1220_window/boundary/'

ImageBlending(img_path, mask_path, dst_path)