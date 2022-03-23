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

from gil_eval import *


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

    return model


def main(mode):
    GPU_NUM = 1

    dataset = GilAAADataset('data', get_transform(train=True))
    dataset_test = GilAAADataset('data', get_transform(train=False))

    test_subject_range = list(range(50, 59))

    test_idx = []

    index = 0
    for path in dataset.imgs:
        split_path = path.split("_")
        if int(split_path[0]) in test_subject_range:
            test_idx.append(index)

        index = index + 1

    total_index = list(range(0, len(dataset.imgs)))
    train_idx = [index for index in total_index if index not in test_idx]

    indices1 = train_idx  # training for 50 persons
    indices2 = test_idx  # testing for 8 persons

    np.random.shuffle(indices1)
    np.random.shuffle(indices2)
    dataset = torch.utils.data.Subset(dataset, indices1)
    dataset_test = torch.utils.data.Subset(dataset_test, indices2)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=24, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and ...
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)


    if 'train' in mode:
        print("*"*30)
        print("\n")
        print("\tTraining ...")
        print("\n")
        print("*" * 30)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            #v evaluate(model, data_loader_test, device=device)

            # if (epoch % 10 == 0):
            #     torch.save(model.state_dict(), './pretrained/pretrained_weight_%d.pth' % epoch)

        torch.save(model.state_dict(), './pretrained/weight_baseline_pos.pth')
        torch.save(dataset_test, './pretrained/test_baseline_pos.pth')


    if 'test' in mode:
        print("*"*25)
        print("\n")
        print("\t\tTesting ...")
        print("\n")
        print("*" * 25)
        save_dir = "result/"

        model.load_state_dict(torch.load('./pretrained/weight_baseline.pth'))
        dataset_test = torch.load('./pretrained/test_baseline.pth')

        total_ol = []
        total_ja = []
        total_di = []
        total_fp = []
        total_fn = []

        num_test = len(dataset_test.indices)

        for i in range(num_test):
            img_name = 'data/raw/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
            mask_name = 'data/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]]

            img = Image.open(img_name).convert("RGB")
            mask_gt = Image.open(mask_name).convert("RGB")
            img_rgb = np.array(img)
            img = F.to_tensor(img)

            model.eval()
            with torch.no_grad():
                prediction = model([img.to(device)])

            im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

            if (list(prediction[0]['boxes'].shape)[0] == 0):
                mask = np.zeros((512, 512), dtype=np.uint8)
            else:
                mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

            img_mask = np.array(mask)
            img_mask_gt = np.array(mask_gt)
            img_mask[img_mask > 50] = 255
            img_mask_gt_gray = cv2.cvtColor(img_mask_gt, cv2.COLOR_BGR2GRAY)


            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            img_result = np.concatenate([img_gray, img_mask_gt_gray, img_mask], axis=1)

            img_overlap = img_mask_gt.copy()
            img_overlap[:, :, 0] = 0
            img_overlap[:, :, 1] = img_mask

            # cv2.imwrite(save_dir + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_mask_gt.png'), img_mask_gt_gray)
            cv2.imwrite(save_dir + dataset_test.dataset.imgs[dataset_test.indices[i]], img_mask)

        for subject in test_subject_range:

            overlap, jaccard, dice, fn, fp = eval_segmentation_volume(save_dir, str(subject))
            print('[Subject = \"' + str(subject) + '\" volume evaluation] overlap:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
                overlap, jaccard, dice, fn, fp))
            print("=" * 50)
            print("\n")
            total_ol.append(overlap)
            total_ja.append(jaccard)
            total_di.append(dice)
            total_fn.append(fn)
            total_fp.append(fp)

        print('[Average volume evaluation] overlap:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
            np.mean(total_ol), np.mean(total_ja), np.mean(total_di), np.mean(total_fn), np.mean(total_fp)))


if __name__ == '__main__':
    main('train')
    # main('test')
