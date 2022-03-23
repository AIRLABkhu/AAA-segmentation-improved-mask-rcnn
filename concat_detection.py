import os
import natsort
import cv2
import numpy as np

file_list = natsort.natsorted(os.listdir("/home/bh/Downloads/aaa_segmentation/data/1220_window/mask/"))


# smooth_list = os.listdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_smooth/result_analysis")
# diou_list = os.listdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_diou/result_analysis")
# giou_list = os.listdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_giou/result_analysis")
# ciou_list = os.listdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_ciou/result_analysis")
# ours_list = os.listdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_0.2/result_analysis")

for file_idx in file_list:
        img = cv2.imread(os.path.join("/home/bh/Downloads/aaa_segmentation/data/1220_window/mask/", file_idx), cv2.IMREAD_GRAYSCALE)

        if len(np.unique(img))==2:
            smooth_img = cv2.imread(os.path.join("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_smooth/result_analysis",file_idx.split(".")[0]+"_bbox.png"))
            diou_img = cv2.imread(os.path.join("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_diou/result_analysis",file_idx.split(".")[0]+"_bbox.png"))
            giou_img = cv2.imread(os.path.join("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_giou/result_analysis",file_idx.split(".")[0]+"_bbox.png"))
            ciou_img = cv2.imread(os.path.join("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_ciou/result_analysis",file_idx.split(".")[0]+"_bbox.png"))
            ours_img = cv2.imread(os.path.join("/home/bh/Downloads/aaa_segmentation/data_visualization/result_220214_0.2/result_analysis",file_idx.split(".")[0]+"_bbox.png"))

            # cv2.putText(smooth_img, "\"smooth\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            # cv2.putText(diou_img, "\"diou\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            # cv2.putText(giou_img, "\"giou\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            # cv2.putText(ciou_img, "\"ciou\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            # cv2.putText(ours_img, "\"ours\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)

            img_all = np.concatenate([smooth_img, diou_img, giou_img, ciou_img, ours_img], axis=1)

            cv2.imwrite(os.path.join("/home/bh/Downloads/aaa_segmentation/data_visualization/detection_result/",file_idx), img_all)
