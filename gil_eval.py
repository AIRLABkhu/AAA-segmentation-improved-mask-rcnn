import numpy as np
import cv2
import os
from PIL import Image
import natsort

def eval_segmentation(src, tgt):
    s = (src / 255.0).astype(np.uint32)
    t = (tgt / 255.0).astype(np.uint32)

    s_abs = s.sum()
    t_abs = t.sum()

    if (t_abs == 0 and s_abs > 0):
        # print("target = background & src = predict")
        s = np.where(s>0.5,0,1)
        t = np.where(t>0.5,0,1)

    s_int_t = np.bitwise_and(s, t)
    s_uni_t = np.bitwise_or(s, t)

    s_diff_t = s - s_int_t
    t_diff_s = t - s_int_t

    if (t_abs ==0 and s_abs == 0):
        # print("target & src == background")
        overlab = 1.0
        jaccard = 1.0
        dice = 1.0
        fn = 0.0
        fp = 0.0
    else:
        overlab = s_int_t.sum() / t.sum()
        jaccard = s_int_t.sum() / s_uni_t.sum()
        dice = 2.0 * s_int_t.sum() / (s.sum() + t.sum())
        fn = t_diff_s.sum() / t.sum()
        fp = s_diff_t.sum() / s.sum()


    # print("s sum: %f, t sum: %f, s_int_t: %f, s_uni_t: %f, s_diff_t:%f, t_diff_t:%f"%(s_abs,t_abs,s_int_t.sum(),s_uni_t.sum(),s_diff_t.sum(), t_diff_s.sum()))

    # cv2.imshow("s_int_t", s_int_t.astype(np.uint8) * 255)
    # cv2.imshow("s_uni_t", s_uni_t.astype(np.uint8) * 255)
    # cv2.imshow("s_diff_t", s_diff_t.astype(np.uint8) * 255)
    # cv2.imshow("t_diff_s", t_diff_s.astype(np.uint8) * 255)
    # cv2.waitKey(1000)
    return overlab, jaccard, dice, fn, fp

def eval_segmentation_volume(save_dir, subject, data_path, start_finish=[]):

    if len(start_finish)!=0:
        np_predict_list = np.arange(start_finish[0], start_finish[1]+1)

    # mask file load
    pred_path = save_dir
    gt_path = os.path.join(data_path, "mask")

    gt_mask_list = natsort.natsorted([name for name in os.listdir(gt_path) if subject == name.split("_")[0]])
    pred_mask_list = natsort.natsorted([name for name in os.listdir(pred_path) if subject == name.split("_")[0]])

    # print("[Subject = \"%d\"] & number of pred image \"%d\" & num of GT \"%d\" "%(int(subject), len(pred_mask_list), len(gt_mask_list)))

    # calculation
    s_sum, t_sum = 0, 0
    intersection, union = 0, 0
    s_diff_t, t_diff_s = 0, 0

    if(len(gt_mask_list) == len(pred_mask_list)):
        for i in range(len(gt_mask_list)):
            gt_slice = Image.open(os.path.join(gt_path, gt_mask_list[i]))
            gt_slice = (np.array(gt_slice)/255.0).astype(np.uint32)
            pred_slice = Image.open(os.path.join(pred_path, pred_mask_list[i]))
            pred_slice = (np.array(pred_slice) / 255.0).astype(np.uint32)

            #### 21.12.01
            if len(start_finish) != 0:
                if int(i) not in np_predict_list:
                    pred_slice = np.zeros((np.shape(pred_slice)[0],np.shape(pred_slice)[1]), dtype=type(pred_slice))

            # print(list(set(np.ravel(pred_slice))))
            # print(list(set(np.ravel(gt_slice))))

            # print(pred_slice.sum())
            # print(gt_slice.sum())
            s_sum += pred_slice.sum()
            t_sum += gt_slice.sum()

            # print(np.bitwise_and(pred_slice, gt_slice).sum())
            # print(np.bitwise_or(pred_slice, gt_slice).sum())
            intersection += np.bitwise_and(pred_slice, gt_slice).sum()
            union += np.bitwise_or(pred_slice, gt_slice).sum()

            # print((pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum())
            # print((gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum())
            s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
            t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

        overlab = intersection / t_sum
        jaccard = intersection / union
        dice = 2.0*intersection / (s_sum + t_sum)
        fn = t_diff_s / t_sum
        fp = s_diff_t / s_sum
        return overlab, jaccard, dice, fn, fp

    else:
        print("GT : %d, Pred : %d" %(len(gt_mask_list), len(pred_mask_list)))
        print("subject = "+subject + " ERROR")


def eval_total_overlab(src, tgt):

    return


def eval_Jaccard(src, tgt):
    return


def eval_Dice(src, tgt):
    return


def eval_false_negative(src, tgt):
    return


def eval_false_positive(src, tgt):
    return
