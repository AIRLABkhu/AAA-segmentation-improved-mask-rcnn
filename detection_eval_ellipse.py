import numpy as np
import os
import cv2
import math
import natsort

# Draw elipsis on image
def fit_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    has_ellipse = len(contours) > 0
    if has_ellipse:
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
    else:
        bigger_pixel_rect = 0
        bigger_pixel_ellip = 0

    return bigger_pixel_rect, int(bigger_pixel_ellip)


if __name__ =="__main__":

    # real world We dont know GT value
    # So we make function and check
    gt_data = np.load("./dataset_manipulation/GT.npy")
    pred_data = np.load("./dataset_manipulation/predict_rpn_1.npy")
    # pred_data = np.load("./dataset_manipulation/predict_fd.npy")


    mask_path = '/home/bh/Downloads/aaa_segmentation/0803/result_0819_update_all_rpn_1/'
    pred_file_list = natsort.natsorted(os.listdir(mask_path))
    pred_file_list.pop()

    subject = np.arange(1, 52)

    total_rate = []
    total_fn = []
    total_fp = []

    mini_slice_num = 9

    for sub_idx in subject:
        flag = 0

        sub_total_len = int(gt_data[(sub_idx-1), 0])
        sub_start = int(gt_data[(sub_idx-1),1])
        sub_finish = int(gt_data[(sub_idx-1),2])
        num_gt = sub_finish-sub_start+1

        gt_np = np.zeros(sub_total_len)
        gt_np[sub_start:sub_finish+1] = 1

        pred_np = pred_data[(sub_idx-1), :sub_total_len]
        num_pred = pred_np.sum()

        ###############################
        #   preprocessing using pred  #
        ###############################
        pred_true_idx = np.array(np.where(pred_np>0)[0])
        pred_seg_idx = []
        # print(pred_true_idx)

        start_point = pred_true_idx[0]
        pred_seg_idx.append(start_point)
        temp_list = []

        for idx in pred_true_idx[1:]:
            if idx-start_point == 1:
                start_point = idx
                pred_seg_idx.append(idx)

            else:
                start_point = idx
                if len(pred_seg_idx) >= mini_slice_num:
                    if len(temp_list) == 0:
                        temp_list = pred_seg_idx
                        flag=1
                    else:
                        if flag == 1:
                            temp_list.extend(pred_seg_idx)
                            sorted(temp_list)
                            flag = 0

                        elif len(temp_list) < len(pred_seg_idx):
                            temp_list = pred_seg_idx
                    pred_seg_idx = []
                else:
                    pred_seg_idx = []

                pred_seg_idx.append(start_point)

        # condition with "1" subject
        if len(temp_list) > mini_slice_num and len(pred_seg_idx) > mini_slice_num:
            temp_list.extend(pred_seg_idx)
            sorted(temp_list)

        # No more than 1 difference between start and finish
        if len(temp_list) < len(pred_seg_idx):
            temp_list = pred_seg_idx
        # print(temp_list)

        # Fill in the empty value between start and finish value
        np_predict_list = np.arange(temp_list[0], temp_list[-1]+1)
        seg_idx = list(np_predict_list)

        not_seg_idx = [x for x in range(len(pred_np)) if x not in seg_idx]
        # print(not_seg_idx)
        pred_np[not_seg_idx] = 0
        pred_np[seg_idx] = 1

        ellipse_list = []
        ### Ellipse fitting post processing
        for i in seg_idx:
            for file_idx in pred_file_list:
                idx_split = file_idx.split("_")
                if int(idx_split[0]) == sub_idx:
                    if i == int(idx_split[1].split('.')[0]):
                        mask_img = cv2.imread(os.path.join(mask_path,file_idx), cv2.IMREAD_GRAYSCALE)
                        rect, ellipse = fit_ellipse(mask_img)
                        ellipse_list.append(ellipse)



        # predict rate in GT
        pred_rate = pred_np[sub_start:sub_finish+1].sum() / num_gt * 100

        fn = 100 - pred_rate

        pred_np[sub_start:sub_finish+1] = 0
        fp = pred_np.sum() / num_pred * 100

        total_rate.append(pred_rate)
        total_fn.append(fn)
        total_fp.append(fp)

        print("\n=== Subject\"%d\" ==="%(sub_idx))
        print("GT_total_num = %d, GT_number of mask = %d, GT_start_point = %d, GT_finish_point = %d"%(sub_total_len, num_gt, sub_start, sub_finish))
        print("pre_total_num = %d, pre_number of mask = %d, pre_start_point = %d, pre_finish_point = %d, pre_num_of_mask_no_filtering = %d"%(sub_total_len, (temp_list[-1]-temp_list[0]+1), temp_list[0], temp_list[-1], len(pred_true_idx)))
        print("False negative rate = %.2f%% , False positive rate = %.2f%%"%(fn, fp))
        print("[Predict(in GT) / GT range] rate = %.2f%%\n\n"%(pred_rate))
        print(ellipse_list)
        # print("%.2f"%fn)

    print("total_rate = %.2f%%, total_fn = %.2f%%, total_fp = %.2f%%"%((sum(total_rate)/len(total_rate)), (sum(total_fn)/len(total_fn)), (sum(total_fp)/len(total_fp))))
