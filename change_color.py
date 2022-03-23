import cv2
import numpy as np
import os

if __name__ =="__main__":
    '''
    change image color 
    step1: image read
    step2: detect image color
    step3: change image color 
    '''

    file_list = list(os.path.join("/home/bh/Desktop/result/", ext)
                     for ext in os.listdir("/home/bh/Desktop/result/") if ext.endswith(".png"))
    for img_path in file_list:
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        for w in np.arange(0,512):
            for h in np.arange(0,512):
                # intersection
                if img[w,h,0]==0 and img[w,h,1]==255 and img[w,h,2]==255:
                    img[w, h, 0] = 255
                    img[w, h, 1] = 255
                    img[w, h, 2] = 255

                # false positive
                elif img[w,h,0]==0 and img[w,h,1]==255 and img[w,h,2]==0:
                    img[w, h, 0] = 0
                    img[w, h, 1] = 255
                    img[w, h, 2] = 0

                # false negative
                elif img[w,h,0]==0 and img[w,h,1]==0 and img[w,h,2]==255:
                    img[w, h, 0] = 0
                    img[w, h, 1] = 165
                    img[w, h, 2] = 255
        x = 177
        y = 116
        h = 202
        w = 202
        crop_img = img[y:y+h, x:x+w]

        cv2.imwrite(os.path.join("/home/bh/Desktop/result/ogw/crop/", img_path.split("/")[5]), crop_img)

        cv2.imwrite(os.path.join("/home/bh/Desktop/result/ogw/", img_path.split("/")[5]), img)

