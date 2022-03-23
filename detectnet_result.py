import numpy as np
import os
import natsort
import cv2


def check_detection_rate(slice_num=6, jump=False):
    # real world We dont know GT value
    # So we make function and check
    gt_data = np.load("./dataset_manipulation/GT_512.npy")
    pred_data = np.load("./fold1.npy")
    print(slice_num)
    print("="*20)
    subject = np.arange(1, 16)
    pred_final_start_finish_point = np.zeros((15, 2))

    total_rate = []
    total_fn = []
    total_fp = []

    mini_slice_num = slice_num

    for sub_idx in subject:

        flag = 0

        sub_total_len = int(gt_data[(sub_idx+45 - 1), 0])
        sub_start = int(gt_data[(sub_idx+45 - 1), 1])
        sub_finish = int(gt_data[(sub_idx+45 - 1), 2])
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



if __name__ =="__main__":
    fold_name = 'fold4'
    file_path = '/home/bh/digits/detectnet_1220/1223'
    file_name = '%s_result_234'%fold_name
    file_img_name = '%s_test'%fold_name
    result_folder_name = 'result_%s'%fold_name

    file_list = list(natsort.natsorted(os.listdir(os.path.join(file_path, file_img_name))))
    print(os.path.join(file_path, file_name))
    f = open(os.path.join(file_path, file_name), 'r')

    while True:
        line = f.readline()
        if not line: break
        line_split = line.split('\t')
        file_idx = int(line_split[0])
        file_bbox = line_split[4].split('.')
        x_min = int(file_bbox[0][2:])
        y_min = int(file_bbox[1])
        x_max = int(file_bbox[2])
        y_max = int(file_bbox[3])
        bbox_list = []
        bbox_list.append(x_min)
        bbox_list.append(y_min)
        bbox_list.append(x_max)
        bbox_list.append(y_max)
        bbox_list = np.array(bbox_list)
        # print(bbox_list)

        np_file_name = file_list[file_idx-1].split('.')[0] + '.npy'
        np.save(os.path.join(file_path, result_folder_name, np_file_name), bbox_list)

    mask_path = '/home/bh/digits/detectnet_1220/1223/result_%s/'%fold_name

    # GT
    # mask_path = '/home/bh/Desktop/AAA_DATA_NEW/256/mask_all'

    file_list = natsort.natsorted(os.listdir(mask_path))
    # file_list.pop()

    data_arr = np.zeros((15, 300))

    if fold_name == "fold1":
        subject_idx = np.arange(1, 16)
    if fold_name == "fold2":
        subject_idx = np.arange(16, 31)
    if fold_name == "fold3":
        subject_idx = np.arange(31, 46)
    if fold_name == "fold4":
        subject_idx = np.arange(46, 61)

    count = 0
    numpy_idx = 0
    for sub_idx in subject_idx:
        ellipse = []
        # print(sub_idx)
        for file_idx in file_list:

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:
                bbox_array = np.load(os.path.join(mask_path, file_idx))

                if np.sum(bbox_array) != 0:
                    # print(bbox_array)
                    data_arr[numpy_idx, count] = 1

                count = count + 1
        numpy_idx +=1
        count = 0

    np.save("./fold1", data_arr)
    print("*" * 50)
    print("==== Done ====")

    np_start_finish = check_detection_rate(slice_num=8, jump=True)
    print(np_start_finish)
