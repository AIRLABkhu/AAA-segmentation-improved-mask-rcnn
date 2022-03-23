import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from GilAAADataset import *
from engine import train_one_epoch, evaluate
import utils
import pickle
import cv2
import transforms as T
from torchvision.transforms import functional as F
from sklearn.model_selection import KFold
from torchvision.models.detection import roi_heads
from torchsummary import summary as summary_
import natsort
import shutil

from gil_eval import *
import random
from unet import mask_unet


if __name__ =="__main__":
    print("=")
    # Mask Preprocessing

    # path = "/home/bh/Downloads/AAA_sampling/dicom_img/"
    # gt_path = "/home/bh/Downloads/AAA_sampling/GT_nii/"
    # mask_path = "/home/bh/Downloads/AAA_sampling/mask_gt/"
    # sub_list = list(natsort.natsorted(os.listdir(os.path.join(path))))
    #
    # for subj in sub_list:
    #     mask_nii_file = str(subj.split("_")[0])+"_substract.nii"
    #     mask = nib.load(os.path.join(gt_path, mask_nii_file))
    #     mask = mask.get_fdata().transpose((1,0,2,3)).astype(np.uint8)
    #     mask = mask.squeeze()
    #     mask[mask>0.5] =255
    #     mask_depth = mask.shape[2]
    #
    #     if not os.path.exists(os.path.join(mask_path, subj)):
    #         os.mkdir(os.path.join(mask_path, subj))
    #
    #     for depth in range(mask_depth):
    #         mask_file = mask[:,:, depth]
    #         cv2.imwrite(os.path.join(mask_path, subj, subj+"_%d.png"%depth), mask_file)


    # Image Preprocessing
    # path = "/home/bh/Downloads/aa/AAA_Hoseo/"
    # img_path = "/home/bh/Downloads/aa/image/"
    # sub_list = list(natsort.natsorted(os.listdir(os.path.join(path))))
    #
    # for subj in sub_list:
    #     file_list = list(natsort.natsorted(os.listdir(os.path.join(path, subj))))
    #
    #     if not os.path.exists(os.path.join(img_path, subj)):
    #         os.mkdir(os.path.join(img_path, subj))
    #
    #     for idx, file_idx in enumerate(file_list):
    #         img = pydicom.dcmread(os.path.join(path,subj,file_idx))
    #         np_img = ((img.pixel_array / 4095) * 255).astype(np.uint8)
    #         # np_img = (img.pixel_array).astype(np.uint8)
    #         cv2.imwrite(os.path.join(img_path, subj, "%d.png" % idx), np_img)

    ###################################################################################

    # dicom_path = "/home/bh/Downloads/AAA_sampling/dicom_img/"
    # bit_path = "/home/bh/Downloads/AAA_sampling/8bit_PNG/"
    # rebit_path = "/home/bh/Downloads/AAA_sampling/8bit_from_dicom/"
    # mask_path = "/home/bh/Downloads/AAA_sampling/mask_gt/"
    #
    #
    # dicom_path_re = "/home/bh/Downloads/AAA_sampling/re_dicom/"
    # bit_path_re = "/home/bh/Downloads/AAA_sampling/re_8bit/"
    # rebit_path_re = "/home/bh/Downloads/AAA_sampling/re_8bit_from_dicom/"
    # mask_path_re = "/home/bh/Downloads/AAA_sampling/re_mask/"
    #
    #
    # dicom_list = list(natsort.natsorted(os.listdir(dicom_path)))
    # bit_list = list(natsort.natsorted(os.listdir(bit_path)))
    #
    # sub_idx = [index for index in dicom_list if index in bit_list]
    #
    # for subj in sub_idx:
    #     if os.path.exists(os.path.join(dicom_path,subj)):
    #         shutil.copytree(dicom_path+subj, dicom_path_re+subj)
    #
    #     if os.path.exists(os.path.join(bit_path,subj)):
    #         shutil.copytree(bit_path+subj, bit_path_re+subj)
    #
    #     if os.path.exists(os.path.join(rebit_path, subj)):
    #         shutil.copytree(rebit_path + subj, rebit_path_re + subj)
    #
    #     if os.path.exists(os.path.join(mask_path, subj)):
    #         shutil.copytree(mask_path + subj, mask_path_re + subj)

    #
    # bit_path = "/home/bh/Downloads/AAA_sampling/re_8bit/"
    # rebit_path = "/home/bh/Downloads/AAA_sampling/re_8bit_from_dicom/"
    #
    # sub_list = list(natsort.natsorted(os.listdir(bit_path)))
    #
    # for sub_idx in sub_list:
    #     bit_len = len(list(natsort.natsorted(os.listdir(os.path.join(bit_path,sub_idx)))))
    #     rebit_len = len(list(natsort.natsorted(os.listdir(os.path.join(rebit_path,sub_idx)))))
    #
    #     if bit_len != rebit_len:
    #         print(sub_idx)



    # move
    # path = '/home/bh/Downloads/1214/KU/DCM/'
    # dst_path = '/home/bh/Downloads/1214/KU/dicom/'
    #
    # gt = "/home/bh/Downloads/1214/KU/mask/"
    # gt_dir = "/home/bh/Downloads/1214/KU/re_mask/"
    #
    # gt_list = list(natsort.natsorted(os.listdir(gt)))
    #
    # list_folder = list(natsort.natsorted(os.listdir(path)))
    #
    # count = 1
    #
    # for folder_idx in list_folder:
    #
    #     sub_mask_path = os.path.join(path, folder_idx)
    #     list_file = list(natsort.natsorted(os.listdir(sub_mask_path)))
    #
    #     sub_idx = [index for index in gt_list if int(index.split('_')[0]) == int(folder_idx.split("_")[0])]
    #     sub_idx =list(natsort.natsorted(sub_idx))
    #     if len(list_file) != len(sub_idx):
    #         print("="*30)
    #
    #     for idx, list_idx in enumerate(list_file):
    #         # split_idx = list_idx.split('_')
    #         rename = str(count) + "_%.4d.png" %(int(list_idx.split(".")[0][-4:]))
    #
    #         shutil.copyfile(os.path.join(gt, sub_idx[idx]), os.path.join(gt_dir, rename))
    #
    #         # img = cv2.imread(os.path.join(sub_mask_path, list_idx))
    #         # cv2.imwrite(os.path.join(dst_path, rename), img)
    #
    #     count = count + 1



    # dicom_path = '/home/bh/Downloads/1214/KU/DCM/'
    # img_path = '/home/bh/Downloads/0906_modify_window_contrast/RAW_PNG/'
    # gt_path = '/home/bh/Downloads/0906_modify_window_contrast/ROI/'
    # blood_path = '/home/bh/Downloads/0906_modify_full_contrast/Blood_PNG_RAW/'
    #
    # dicom_path_dst = '/home/bh/Downloads/1214/1217_final/dicom/'
    # img_path_dst = '/home/bh/Downloads/1214/1217_final/raw/'
    # gt_path_dst = '/home/bh/Downloads/1214/1217_final/mask/'
    # blood_path_dst = '/home/bh/Downloads/1214/1217_final/blood/'
    #
    #
    #
    # dicom_list = list(natsort.natsorted(os.listdir(dicom_path)))
    # img_list = list(natsort.natsorted(os.listdir(img_path)))
    #
    # for subj in dicom_list:
    #     if subj == "05860383_1" or subj == '13420963_1' or subj == '16470713_1' or subj == '17018233_1' or subj == '33675193_1':
    #         continue
    #     if subj in img_list:
    #         shutil.copytree(dicom_path+subj, dicom_path_dst+subj)
    #         shutil.copytree(img_path+subj, img_path_dst+subj)
    #         shutil.copytree(gt_path+subj, gt_path_dst+subj)
    #         shutil.copytree(blood_path+subj, blood_path_dst+subj)

    # dataset check .DS_Store
    # path_full = "/home/bh/Downloads/1220/full/"
    # path_window = "/home/bh/Downloads/1220/window/"
    # path_mask = "/home/bh/Downloads/1220/mask/"
    #
    # path_mask_start = "/home/bh/Downloads/0906_modify_window_contrast/ROI/"
    # path_window_start = "/home/bh/Downloads/0906_modify_window_contrast/RAW_PNG/"
    #
    # list_full = list(natsort.natsorted(os.listdir(path_full)))
    #
    # for subj in list_full:
    #     list_file = list(natsort.natsorted(os.listdir(os.path.join(path_full, subj))))
    #     if not os.path.exists(os.path.join(path_mask, subj)):
    #         os.mkdir(os.path.join(path_mask, subj))
    #         os.mkdir(os.path.join(path_window, subj))
    #     for file_idx in list_file:
    #         mask_name = file_idx.split('_')
    #         mask_name = mask_name[0]+'_'+mask_name[1]+'_'+'ROI_'+mask_name[2]
    #         shutil.copyfile(os.path.join(path_window_start,subj,file_idx), os.path.join(path_window, subj, file_idx))
    #         shutil.copyfile(os.path.join(path_mask_start,subj,mask_name), os.path.join(path_mask, subj, file_idx))


    # move test folder

    #
    # path = '/home/bh/digits/detectnet_1220/test/fold4_test'
    # file_list = list(natsort.natsorted(os.listdir(path)))
    # with open('/home/bh/digits/detectnet_1220/test/' + 'fold4.txt', 'w') as fp:
    #
    #     for file_idx in file_list:
    #         label = os.path.join(path,file_idx)
    #         fp.write(label + '\n')
    #
    # fp.close()