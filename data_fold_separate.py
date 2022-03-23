import os
import natsort
import random
import numpy as np

from shutil import copyfile
from sklearn.model_selection import KFold

if __name__=="__main__":
    dst_dir = '/home/bh/Downloads/0906_modify_full_contrast/0906_rename_for_bh/Fold/'
    label_dir = '/home/bh/Downloads/0906_modify_full_contrast/0906_rename_for_bh/label/'
    raw_dir = '/home/bh/Downloads/0906_modify_full_contrast/0906_rename_for_bh/raw_all/'

    val_ratio = 0.2

    fold_num = 4
    subject_idx = np.arange(1,61)
    kfold = KFold(n_splits=fold_num, shuffle=False)

    file_list = natsort.natsorted(os.listdir(raw_dir))

    for fold, (train_ids, test_ids) in enumerate(kfold.split(subject_idx)):
        fold_id = fold + 1
        train_ids = train_ids + 1
        test_ids = test_ids + 1

        if not os.path.exists(os.path.join(dst_dir, 'fold'+str(fold_id))):
            os.makedirs(os.path.join(dst_dir, 'fold' + str(fold_id)))
            os.makedirs(os.path.join(dst_dir, 'fold' + str(fold_id)+'_label'))
            os.makedirs(os.path.join(dst_dir, 'fold' + str(fold_id)+'_val'))
            os.makedirs(os.path.join(dst_dir, 'fold' + str(fold_id)+'_val_label'))
            os.makedirs(os.path.join(dst_dir, 'fold' + str(fold_id)+'_test'))
            os.makedirs(os.path.join(dst_dir, 'fold' + str(fold_id)+'_test_label'))

        for idx in file_list:
            idx_split = idx.split('_')
            label_split = idx.split('.')
            label_name = label_split[0]+'.txt'

            if int(idx_split[0]) in train_ids:
                if random.random() < val_ratio:
                    copyfile(os.path.join(raw_dir, idx), os.path.join(dst_dir, 'fold' + str(fold_id) + '_val', idx))
                    copyfile(os.path.join(label_dir, label_name), os.path.join(dst_dir, 'fold' + str(fold_id) + '_val_label', label_name))

                else:
                    copyfile(os.path.join(raw_dir, idx), os.path.join(dst_dir, 'fold' + str(fold_id), idx))
                    copyfile(os.path.join(label_dir, label_name), os.path.join(dst_dir, 'fold' + str(fold_id)+'_label', label_name))

            elif int(idx_split[0]) in test_ids:
                copyfile(os.path.join(raw_dir, idx), os.path.join(dst_dir, 'fold' + str(fold_id)+'_test', idx))
                copyfile(os.path.join(label_dir, label_name), os.path.join(dst_dir, 'fold' + str(fold_id)+'_test_label', label_name))


