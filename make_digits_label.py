import os
import natsort
import cv2
import numpy as np
from sklearn.model_selection import KFold
import shutil

if __name__=="__main__":

    work_dir = '/home/bh/Downloads/1220'
    mask_dir = '/home/bh/Downloads/1220/rename_mask'

    file_idx = natsort.natsorted(os.listdir(mask_dir))

    if not os.path.exists(os.path.join(work_dir, 'label')):
        os.makedirs(os.path.join(work_dir, 'label'))

    for idx in file_idx:

        idx_split = idx.split('.')
        mask = cv2.imread(os.path.join(mask_dir, idx), cv2.IMREAD_GRAYSCALE)


        # check B.B coordinate
        if len(np.unique(mask)) == 2:
            pos = np.where(mask)
            l = np.min(pos[1])
            t = np.min(pos[0])
            r = np.max(pos[1])
            b = np.max(pos[0])

        else:
            l = 10
            t = 10
            r = 502
            b = 502

        # img = cv2.imread(os.path.join(work_dir, 'raw_all', idx), cv2.IMREAD_COLOR)
        # cv2.rectangle(img, (l,t), (r,b), (0,255,0), 3)
        # cv2.imwrite(os.path.join(work_dir,'raw_bbox_check', idx), img)
        # cv2.imshow("img", img)
        # cv2.waitKey(1000000)

        with open(work_dir + '/label/' + idx_split[0] + '.txt', 'w') as fp:
                if len(np.unique(mask)) == 2:
                    type = 'Car'
                    truncated = 0
                    occluded = 0
                    alpha = 0
                    tail = '0 0 0 0 0 0 0 0'

                    label = type + ' ' + \
                            str(truncated) + ' ' + \
                            str(occluded) + ' ' + \
                            str(alpha) + ' ' + \
                            str(l) + ' ' + str(t) + ' ' + str(r) + ' ' + str(b) + ' ' + tail

                    fp.write(label + '\n')

                else:
                    type = 'DontCare'
                    truncated = -1
                    occluded = -1
                    alpha = -10
                    tail = '-1 -1 -1 -1000 -1000 -1000 -10'

                    label = type + ' ' + \
                            str(truncated) + ' ' + \
                            str(occluded) + ' ' + \
                            str(alpha) + ' ' + \
                            str(l) + ' ' + str(t) + ' ' + str(r) + ' ' + str(b) + ' ' + tail

                    fp.write(label + '\n')

    path = '/home/bh/Downloads/1220/label/'
    file_list = list(natsort.natsorted(os.listdir(path)))

    total_subject = list(range(1, 61))
    kfold = KFold(n_splits=4, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):
        print(fold)
        train_ids = train_ids + 1
        test_ids = test_ids + 1
        fold_idx = fold + 1
        # move train folder
        for file_idx in file_list:
            if int(file_idx.split('_')[0]) in train_ids:
                shutil.copyfile(os.path.join(path, file_idx), os.path.join('/home/bh/digits/detectnet_1220/train/fold%d_label/'%fold_idx, file_idx))
            else:
                continue

