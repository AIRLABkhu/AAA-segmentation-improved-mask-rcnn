import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from GilAAADataset import *
from DicomDataset import *

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

from gil_eval import *
import random
from unet import mask_unet
from dataset_manipulation.make_pred_numpy_file import make_prediction_file
from detection_eval import check_detection_rate

def count_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from dataset_manipulation.make_pred_numpy_file import make_prediction_file
from detection_eval import check_detection_rate

def count_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    model.roi_heads.mask_unet = mask_unet()
    return model


def main(mode, model_path_name, gpu_idx=0, train_batch_size=1, raw_path=""):


    GPU_NUM = gpu_idx
    fold_num = 4
    device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

    raw_path = raw_path

    if 'dicom' in raw_path:
        total_dataset = AAA_dicom(raw_path, get_transform(train=True))
        total_dataset_test = AAA_dicom(raw_path, get_transform(train=False))
    else:
        total_dataset = GilAAADataset(raw_path, get_transform(train=True))
        total_dataset_test = GilAAADataset(raw_path, get_transform(train=False))

    ################################
    # Modify subject range
    total_subject = list(range(1, 61))
    kfold = KFold(n_splits=fold_num, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):
        if fold!=2:
            continue

        for index, value in enumerate(test_ids):
            test_ids[index] = value + 1
        test_idx = []

        index = 0
        for path in total_dataset.imgs:
            split_path = path.split("_")
            if int(split_path[0]) in test_ids:
                test_idx.append(index)

            index = index + 1

        total_index = list(range(0, len(total_dataset.imgs)))
        train_idx = [index for index in total_index if index not in test_idx]

        indices1 = train_idx
        indices2 = test_idx

        np.random.shuffle(indices1)
        np.random.shuffle(indices2)

        # valid_idx = random.sample(indices1, int(len(indices1)*0.1))
        # indices1 = [index for index in indices1 if index not in valid_idx]

        dataset = torch.utils.data.Subset(total_dataset, indices1)
        # dataset_valid = torch.utils.data.Subset(total_dataset_test, valid_idx)
        dataset_test = torch.utils.data.Subset(total_dataset_test, indices2)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, num_workers=0,
            collate_fn=utils.collate_fn)

        # data_loader_valid = torch.utils.data.DataLoader(
        #     dataset_valid, batch_size=1, shuffle=False, num_workers=0,
        #     collate_fn=utils.collate_fn)

        # our dataset has two classes only - background and ...
        num_classes = 2

        # get the model using our helper function
        model = get_instance_segmentation_model(num_classes)
        # move model to the right device
        model.to(device)



    if 'test' in mode:
        print("*"*25)
        print("\n")
        print("\t\tTesting ...")
        print("\n")
        print("*" * 25)

        if not os.path.exists("/home/bh/Downloads/aaa_segmentation/data_visualization/result_%s/"%(model_path_name)):
            os.mkdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_%s/"%(model_path_name))
        if not os.path.exists('/home/bh/Downloads/aaa_segmentation/data_visualization/result_%s/result_analysis/'%(model_path_name)):
            os.mkdir("/home/bh/Downloads/aaa_segmentation/data_visualization/result_%s/result_analysis/"%(model_path_name))
        save_dir = '/home/bh/Downloads/aaa_segmentation/data_visualization/result_%s/'%(model_path_name)
        save_dir_analysis = '/home/bh/Downloads/aaa_segmentation/data_visualization/result_%s/result_analysis/'%(model_path_name)

        total_ol = []
        total_ja = []
        total_di = []
        total_fp = []
        total_fn = []

        for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):
            # if fold!= 0:
            #     continue

            for index, value in enumerate(test_ids):
                test_ids[index] = value + 1

            fold_ol = []
            fold_ja = []
            fold_di = []
            fold_fp = []
            fold_fn = []

            # model.load_state_dict(torch.load('./pretrained/256_s_fold_%d_%s.pth'%(fold,model_path_name)))
            # # test_path = "0815_update_all"
            # dataset_test = torch.load('./pretrained/256_s_test_fold_%d_%s.pth'%(fold,model_path_name))
            #
            # num_test = len(dataset_test.indices)
            #
            # for i in range(num_test):
            #     img_name = raw_path + '/raw/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
            #     mask_name = raw_path + '/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
            #
            #     if 'dicom' in raw_path:
            #         mask_name = raw_path + '/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]].split('.')[0]+'.png'
            #         img = pydicom.dcmread(img_name)
            #         img = (img.pixel_array / 4095).astype(np.float32)
            #         img = torch.from_numpy(img).contiguous().type(torch.FloatTensor)
            #         img_rgb = (np.array(img) * 255).astype(np.uint8)
            #         img = img.unsqueeze(0)
            #
            #     else:
            #         img = Image.open(img_name)
            #         # img = Image.open(img_name).convert("RGB")
            #         img_rgb = np.array(img)
            #         img = F.to_tensor(img)
            #
            #     mask_gt = Image.open(mask_name).convert("RGB")
            #
            #     model.eval()
            #     with torch.no_grad():
            #         prediction = model([img.to(device)])
            #
            #     if (list(prediction[0]['boxes'].shape)[0] == 0):
            #         mask = np.zeros((512, 512), dtype=np.uint8)
            #     else:
            #         mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
            #
            #     img_mask = np.array(mask)
            #     img_mask_gt = np.array(mask_gt)
            #     img_mask[img_mask > 127] = 255
            #     img_mask[img_mask <= 127] = 0
            #
            #     # img_gray = img_rgb
            #     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            #
            #     img_gray_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            #
            #     img_overlap = img_mask_gt.copy()
            #     img_overlap[:, :, 0] = 0
            #     img_overlap[:, :, 1] = img_mask
            #
            #     img_pred_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
            #
            #     add_img = cv2.addWeighted(img_gray_color, 0.7, img_overlap, 0.3, 0)
            #
            #     red_gt = img_mask_gt.copy()
            #     green_pred = img_pred_color.copy()
            #
            #     idx_gt = np.where(red_gt > 0)
            #     idx_sr = np.where(green_pred > 0)
            #     red_gt[idx_gt[0], idx_gt[1], :] = [0, 0, 255]
            #     green_pred[idx_sr[0], idx_sr[1], :] = [0, 255, 0]
            #
            #     cv2.putText(img_gray_color, "\"Raw image\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            #     cv2.putText(add_img, "\"Raw + GT + Predict\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            #     cv2.putText(red_gt, "\"GT\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            #     cv2.putText(green_pred, "\"Predict\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            #     cv2.putText(img_overlap, "\"GT + predict\"", (5,25),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1,cv2.LINE_AA, bottomLeftOrigin=False)
            #
            #     img_all = np.concatenate([img_gray_color, add_img, red_gt, green_pred, img_overlap], axis=1)
            #
            #
            #     if 'dicom' in raw_path:
            #         all_img_file = dataset_test.dataset.imgs[dataset_test.indices[i]].split('.')[0]+'_maskrcnn.png'
            #         mask_file = dataset_test.dataset.imgs[dataset_test.indices[i]].split('.')[0] + '.png'
            #         cv2.imwrite(save_dir_analysis + all_img_file, img_all)
            #         cv2.imwrite(save_dir + mask_file, img_mask)
            #     else:
            #         cv2.imwrite(save_dir_analysis + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_maskrcnn.png'), img_all)
            #         cv2.imwrite(save_dir + dataset_test.dataset.imgs[dataset_test.indices[i]], img_mask)

            # 21.12.01
            # Save prediction file
            # make_prediction_file(save_dir)
            np_start_finish = check_detection_rate(slice_num=8, jump=False)
            # np_start_finish = np.load("./predict_start_finish.npy")

            for subject in test_ids:
                # print("Fold = %d, subject = %d"%(fold, subject))
                # overlap, jaccard, dice, fn, fp = eval_segmentation_volume(save_dir, str(subject), raw_path)
                overlap, jaccard, dice, fn, fp = eval_segmentation_volume_digits(save_dir, str(subject), raw_path, np_start_finish[int(subject) - 1, :])

                print(str(subject) + ' %.4f %.4f %.4f %.4f %.4f' % (overlap, jaccard, dice, fn, fp))

                # print("=" * 50)
                # print("\n")
                fold_ol.append(overlap)
                fold_ja.append(jaccard)
                fold_di.append(dice)
                fold_fn.append(fn)
                fold_fp.append(fp)
            total_ol.append(np.mean(fold_ol))
            total_ja.append(np.mean(fold_ja))
            total_di.append(np.mean(fold_di))
            total_fn.append(np.mean(fold_fn))
            total_fp.append(np.mean(fold_fp))

        print('[Average volume evaluation] overlap:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
            np.mean(total_ol), np.mean(total_ja), np.mean(total_di), np.mean(total_fn), np.mean(total_fp)))


def eval_segmentation_volume_digits(save_dir, subject, data_path, start_finish=[]):

    if len(start_finish)!=0:
        np_predict_list = np.arange(start_finish[0], start_finish[1]+1)

    # mask file load
    # pred_path = save_dir
    pred_path = os.path.join(data_path, "mask")

    gt_path = os.path.join(data_path, "mask")

    gt_mask_list = sorted([name for name in os.listdir(gt_path) if subject == name.split("_")[0]])
    pred_mask_list = sorted([name for name in os.listdir(pred_path) if subject == name.split("_")[0]])

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



if __name__ == '__main__':


    model_path_name = "1220_ours_mask"

    raw_path = 'data/1220_window'


    # Now: Only use 1 fold
    # epoch and step size modi
    gpu_idx = 0
    train_batch_size = 1

    print("*"*50)
    print("raw_Path : " + raw_path)
    print("Model_Path : " + model_path_name)
    print("Batch size: " + str(train_batch_size))
    print("*" * 50)

    main('test', model_path_name, gpu_idx, raw_path=raw_path)

