import numpy as np
import cv2
import os
import natsort
from PIL import Image

def eval_segmentation_volume(gt_path, pred_path, file_list):

    # calculation
    s_sum, t_sum = 0, 0
    intersection, union = 0, 0
    s_diff_t, t_diff_s = 0, 0

    for i in range(len(file_list)):
        gt_slice = Image.open(os.path.join(gt_path, file_list[i]))
        gt_slice = (np.array(gt_slice)/255.0).astype(np.uint32)
        pred_slice = Image.open(os.path.join(pred_path, file_list[i]))
        pred_slice = (np.array(pred_slice) / 255.0).astype(np.uint32)

        if (len(np.unique(gt_slice)) != len(np.unique(pred_slice))):
            black_img = np.zeros((np.shape(pred_slice)[0], np.shape(pred_slice)[1]), dtype=np.uint8)
            print(os.path.join(pred_path, file_list[i]))
            print("Finish")
            cv2.imwrite(os.path.join(pred_path, file_list[i]), black_img)
            pred_slice = black_img.astype(np.uint32)

        s_sum += pred_slice.sum()
        t_sum += gt_slice.sum()
        intersection += np.bitwise_and(pred_slice, gt_slice).sum()
        union += np.bitwise_or(pred_slice, gt_slice).sum()
        s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
        t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

    overlab = intersection / t_sum
    jaccard = intersection / union
    dice = 2.0*intersection / (s_sum + t_sum)
    fn = t_diff_s / t_sum
    fp = s_diff_t / s_sum
    return overlab, jaccard, dice, fn, fp


def change_pixel_value():
    root = "/home/bh/Downloads/image_result/pred_mia_rgb/"
    dst = "/home/bh/Downloads/image_result/pred_mia_rgb_re/"
    subj_list = natsort.natsorted(os.listdir(root))

    for subj_idx in subj_list:

        if subj_idx == '09161643_1' or subj_idx =='32280863_1':

            if not os.path.exists(os.path.join(dst, subj_idx)):
                os.mkdir(os.path.join(dst, subj_idx))

            file_list = natsort.natsorted(os.listdir(os.path.join(root, subj_idx)))
            for file_idx in file_list:
                print(file_idx)
                img = cv2.imread(os.path.join(root,subj_idx, file_idx), cv2.IMREAD_COLOR)
                img_gray = cv2.imread(os.path.join(root,subj_idx, file_idx), cv2.IMREAD_GRAYSCALE)
                black_img = np.zeros((np.shape(img)[0], np.shape(img)[1]), dtype=np.uint8)
                if len(np.unique(img_gray))!=1:
                    rows, cols, _ = img.shape
                    for i in range(rows):
                        for j in range (cols):
                            if np.array_equal(img[i,j], [0, 0, 255]):
                                black_img[i,j] = 100
                            elif np.array_equal(img[i,j], [0, 255, 0]):
                                black_img[i,j] = 155
                            elif np.array_equal(img[i,j], [0, 255, 255]):
                                black_img[i,j] = 255
                    cv2.imwrite(os.path.join(dst, subj_idx, file_idx), black_img)
                else:
                    cv2.imwrite(os.path.join(dst, subj_idx, file_idx), black_img)



if __name__ == "__main__":
    change_pixel_value()

    # test_path = "/home/bh/Downloads/image_result/pred_mia_rgb/"
    # gt_path = "/home/bh/Downloads/image_result/gt/"
    # pred_path = "/home/bh/Downloads/image_result/pred_MIA/"
    # # pred_default_path = "/home/bh/Downloads/image_result/pred_default/"
    #
    # subj_list = natsort.natsorted(os.listdir(gt_path))
    #
    # total_ol = []
    # total_ja = []
    # total_di = []
    # total_fp = []
    # total_fn = []
    #
    # for subj_idx in subj_list:
    #     print(subj_idx)
    #     file_list = natsort.natsorted(os.listdir(os.path.join(gt_path, subj_idx)))
    #
    #     for file_idx in file_list:
    #         if not os.path.exists(os.path.join(test_path, subj_idx)):
    #             os.mkdir(os.path.join(test_path, subj_idx))
    #
    #         gt_img = cv2.imread(os.path.join(gt_path, subj_idx, file_idx), cv2.IMREAD_COLOR)
    #         pred_img = cv2.imread(os.path.join(pred_path, subj_idx, file_idx), cv2.IMREAD_GRAYSCALE)
    #         # pred_img_default = cv2.imread(os.path.join(pred_default_path, subj_idx, file_idx), cv2.IMREAD_GRAYSCALE)
    #
    #         img_overlap = gt_img.copy()
    #         img_overlap[:, :, 0] = 0
    #         img_overlap[:, :, 1] = pred_img
    #
    #         cv2.imwrite(os.path.join(test_path,subj_idx, file_idx), img_overlap)
    # #     overlap, jaccard, dice, fn, fp = eval_segmentation_volume(os.path.join(gt_path, subj_idx),
    # #                                                               os.path.join(pred_path, subj_idx), file_list)
    # #
    # #     total_ol.append(overlap)
    # #     total_ja.append(jaccard)
    # #     total_di.append(dice)
    # #     total_fn.append(fn)
    # #     total_fp.append(fp)
    # #
    # #     print(str(subj_idx) + ' %.4f %.4f %.4f %.4f %.4f' % (overlap, jaccard, dice, fn, fp))
    # #
    # # print('[Average volume evaluation] overlap:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
    # #     np.mean(total_ol), np.mean(total_ja), np.mean(total_di), np.mean(total_fn), np.mean(total_fp)))
    #
