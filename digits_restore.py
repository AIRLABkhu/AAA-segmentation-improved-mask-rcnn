import os
import cv2
import numpy as np
import natsort

def check_detection_rate(slice_num=6, jump=False):
    # real world We dont know GT value
    # So we make function and check
    gt_data = np.load("./dataset_manipulation/GT_512.npy")
    pred_data = np.load("./dataset_manipulation/predict_digits_default.npy")

    subject = np.arange(1, 61)
    pred_final_start_finish_point = np.zeros((60, 2))

    total_rate = []
    total_fn = []
    total_fp = []

    mini_slice_num = slice_num

    for sub_idx in subject:

        flag = 0

        sub_total_len = int(gt_data[(sub_idx - 1), 0])
        sub_start = int(gt_data[(sub_idx - 1), 1])
        sub_finish = int(gt_data[(sub_idx - 1), 2])
        num_gt = sub_finish - sub_start + 1

        gt_np = np.zeros(sub_total_len)
        gt_np[sub_start:sub_finish + 1] = 1

        pred_np = pred_data[(sub_idx - 1), :sub_total_len]
        num_pred = int(pred_np.sum())
        if num_pred == 0:
            print("=== Model Do not detected anything %d ===" % (sub_idx))
            total_rate.append(0)
            total_fn.append(0)
            total_fp.append(0)
            continue
        ###############################
        #   preprocessing using pred  #
        ###############################
        pred_true_idx = np.array(np.where(pred_np > 0)[0])
        pred_seg_idx = []
        # print(pred_true_idx)

        start_point = pred_true_idx[0]
        pred_seg_idx.append(start_point)
        temp_list = []

        for idx in pred_true_idx[1:]:
            if idx - start_point == 1:
                start_point = idx
                pred_seg_idx.append(idx)

            else:
                start_point = idx
                if len(pred_seg_idx) >= mini_slice_num:
                    if len(temp_list) == 0:
                        temp_list = pred_seg_idx
                        flag = 1
                    else:
                        if (jump):
                            if flag == 1:
                                temp_list.extend(pred_seg_idx)
                                sorted(temp_list)
                                flag = 0

                            elif len(temp_list) < len(pred_seg_idx):
                                temp_list = pred_seg_idx
                        else:
                            if len(temp_list) < len(pred_seg_idx):
                                temp_list = pred_seg_idx

                    pred_seg_idx = []
                else:
                    pred_seg_idx = []

                pred_seg_idx.append(start_point)

        # # condition with "1" subject
        # if len(temp_list) > mini_slice_num and len(pred_seg_idx) > mini_slice_num:
        #     temp_list.extend(pred_seg_idx)
        #     sorted(temp_list)

        # No more than 1 difference between start and finish
        if len(temp_list) < len(pred_seg_idx):
            temp_list = pred_seg_idx
        # print(temp_list)

        # Fill in the empty value between start and finish value
        np_predict_list = np.arange(temp_list[0], temp_list[-1] + 1)
        seg_idx = list(np_predict_list)

        not_seg_idx = [x for x in range(len(pred_np)) if x not in seg_idx]
        # print(not_seg_idx)
        pred_np[not_seg_idx] = 0
        pred_np[seg_idx] = 1

        # predict rate in GT
        pred_rate = pred_np[sub_start:sub_finish + 1].sum() / num_gt * 100

        fn = (1 - pred_np[sub_start:sub_finish + 1]).sum() / num_gt * 100

        pred_np[sub_start:sub_finish + 1] = 0
        fp = pred_np.sum() / num_pred * 100

        total_rate.append(pred_rate)
        total_fn.append(fn)
        total_fp.append(fp)

        # print("\n=== Subject\"%d\" ===" % (sub_idx))
        # print("GT_total_num = %d, GT_number of mask = %d, GT_start_point = %d, GT_finish_point = %d" % (
        # sub_total_len, num_gt, sub_start, sub_finish))
        # print(
        #     "pre_total_num = %d, pre_number of mask = %d, pre_start_point = %d, pre_finish_point = %d, pre_num_of_mask_no_filtering = %d" % (
        #     sub_total_len, (temp_list[-1] - temp_list[0] + 1), temp_list[0], temp_list[-1], len(pred_true_idx)))
        # # print("[Predict(in GT) / GT range] rate = %.2f%%\n\n" % (pred_rate))
        # print("False negative rate = %.2f%% , False positive rate = %.2f%%\n\n\n" % (fn, fp))
        # # print("%.2f"%fn)

        pred_final_start_finish_point[(sub_idx - 1), 0] = temp_list[0]
        pred_final_start_finish_point[(sub_idx - 1), 1] = temp_list[-1]

    fn_t_percent = sum(total_fn) / len(total_fn)
    fp_t_percent = sum(total_fp) / len(total_fp)

    fn_t = (sum(total_fn) / len(total_fn)) / 100
    fp_t = (sum(total_fp) / len(total_fp)) / 100
    recall = (1 - fn_t) / ((1 - fn_t) + fn_t)
    precision = (1 - fn_t) / ((1 - fn_t) + fp_t)
    f1 = 2 * ((recall * precision) / (recall + precision))
    print("Total_fn = %.2f%%, Total_fp = %.2f%%" % (fn_t_percent, fp_t_percent))
    print("recall = %.2f%% , Precision = %.2f%%" % (recall * 100, precision * 100))
    print("F1 Score = %.2f%%" % (f1 * 100))

    return pred_final_start_finish_point


if __name__=="__main__":

    np_start_finish = check_detection_rate(slice_num=8, jump=False)
    # print(np_start_finish)
    gt_path = '/home/bh/Downloads/1220/window/'
    sub_list = list(natsort.natsorted(os.listdir(gt_path)))

    mask_path = '/home/bh/Downloads/1220/mask/'

    bbox_path = '/home/bh/digits/detectnet_1220/1223/1227_all/'
    bbox_list = list(natsort.natsorted(os.listdir(bbox_path)))

    for sub_id in range(0,60):
        # print(sub_id)
        global_bbox = {'x_min': 10000, 'y_min': 10000, 'x_max': 0, 'y_max': 0}

        bbox_subj_list = [index for index in bbox_list if int(index.split('_')[0])==int(sub_id + 1)]

        sub_folder_name = sub_list[sub_id]
        start_finish = np_start_finish[sub_id]

        if start_finish[0] == 0 or start_finish[1] ==0:
            continue

        if len(start_finish) != 0:
            np_predict_list = np.arange(start_finish[0], start_finish[1] + 1)

        file_list = list(natsort.natsorted(os.listdir(os.path.join(gt_path,sub_folder_name))))
        pred_file_list = list(natsort.natsorted(os.listdir(os.path.join('/home/bh/digits/detectnet_1220/1223/1_crop_gt',sub_folder_name))))

        # check file_length
        if len(bbox_subj_list) != len(file_list):
            print("Error")
            break

        for file_idx, file_name in enumerate(bbox_subj_list):
            if file_idx in np_predict_list:
                bbox = np.load(os.path.join(bbox_path, file_name))

                if global_bbox['x_min'] > bbox[0] and bbox[0]>0:
                    global_bbox['x_min'] = bbox[0]
                if global_bbox['y_min'] > bbox[1] and bbox[0]>0:
                    global_bbox['y_min'] = bbox[1]
                if global_bbox['x_max'] < bbox[2] and bbox[0]>0:
                    global_bbox['x_max'] = bbox[2]
                if global_bbox['y_max'] < bbox[3] and bbox[0]>0:
                    global_bbox['y_max'] = bbox[3]

        global_bbox['x_min'] -= 42
        global_bbox['y_min'] -= 41
        global_bbox['x_max'] += 42
        global_bbox['y_max'] += 41

        print(global_bbox)

        path = '/home/bh/digits/detectnet_1220/1223/1_crop_gt/'
        dst_path ='/home/bh/digits/detectnet_1220/1223/1_512_gt/'
        if not os.path.exists(os.path.join(dst_path,sub_folder_name)):
            os.mkdir(os.path.join(dst_path,sub_folder_name))

        for file_idx, file_name in enumerate(pred_file_list):
            # name = file_name.split("_")
            # name = name[1] +"_"+ name[2] +"_"+ name[3]

            pred_img = cv2.imread(os.path.join(path,sub_folder_name,file_name), cv2.IMREAD_GRAYSCALE)

            img = np.zeros((512,512), dtype=np.uint8)
            img[global_bbox['y_min']:global_bbox['y_max'], global_bbox['x_min']:global_bbox['x_max']] = pred_img

            cv2.imwrite(os.path.join(dst_path, sub_folder_name, file_name), img)


