import cv2
import numpy as np
import os


def ImageBlending(BG, GT, PRED, save_path):
    bg = cv2.imread(BG, cv2.IMREAD_COLOR)
    gt = cv2.imread(GT, cv2.IMREAD_COLOR)
    pred = cv2.imread(PRED, cv2.IMREAD_COLOR)
    pred[pred > 127] = 255
    pred[pred <= 127] = 0

    blended = cv2.addWeighted(gt, 1, pred, 1, 0)

    bg = cv2.bitwise_and(bg, cv2.bitwise_not(blended))

    # color

    gt[(gt == 255).all(-1)] = [0, 0, 255]
    pred[(pred == 255).all(-1)] = [0, 255, 0]

    # cv2.imshow("gt",gt)
    # cv2.imshow("pred",pred)
    blended = cv2.addWeighted(gt, 1, pred, 1, 0)
    # blended = cv2.addWeighted(gt, 1, gt, 1, 0)

    # blended = cv2.addWeighted(blended, 1, bg, 1, 0)

    # cv2.imshow("blend",blended)
    # print(save_path)
    # print(PRED.split('/')[-1])
    print(save_path + '/' + PRED.split('/')[-1])
    cv2.imwrite(save_path + '/' + PRED.split('/')[-1], blended)

    # cv2.waitKey(0)


GT = "/home/bh/Downloads/1220/mask"
CT = "/home/bh/Downloads/1220/window"
PRED = "/home/bh/digits/detectnet_1220/1223/1_512_pred"
result = "/home/bh/digits/detectnet_1220/1223/1_result_gt"

subject_list = os.listdir(PRED)
# print(subject_list)


for subject in subject_list:
    slice_list = os.listdir(os.path.join(PRED, subject))
    if not os.path.isdir(os.path.join(result, subject)):
        os.mkdir(os.path.join(result, subject))

    for slice_num in slice_list:
        pred = os.path.join(PRED, subject, slice_num)

        # print(slice_num[5:])
        # slice_num = slice_num[5:]
        gt = os.path.join(GT, subject, slice_num)
        ct = os.path.join(CT, subject, slice_num)

        ImageBlending(ct, gt, pred, os.path.join(result, subject))
