import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import operator
import natsort
from torchvision.transforms import functional as F

# from engine import train_one_epoch, evaluate
# import utils
import transforms as T
import cv2


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class GilAAADataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        # dataset path
        self.root = root
        # it==train, Set to true
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.imgs = list(natsort.natsorted(os.listdir(os.path.join(root, "raw"))))
        self.masks = list(natsort.natsorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "raw", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        img = Image.open(img_path)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = np.array(mask) / 255
        mask = mask.astype(np.uint8) # 0 ~ 255
        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        if len(obj_ids) == 1:

            masks = mask == obj_ids[:, None, None]
            masks = np.invert(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])

            target = {}
            target["boxes"] = torch.zeros((0,4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype = torch.int64)
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = torch.zeros(0, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        else:

            # first id is the background, so remove it
            obj_ids = obj_ids[1:] # [0,1]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None] # 1 * 512 * 512

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids) # num_objs = 1
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1]) -5
                ymin = np.min(pos[0]) -5
                xmax = np.max(pos[1]) +5
                ymax = np.max(pos[0]) +5
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8) # 512 * 512

            image_id = torch.tensor([idx])
            area = np.float32(0.0)
            if num_objs > 0:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # are

            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd


        if self.transforms is not None:
            # a = img
            # a = np.array(a)
            # flat = np.concatenate(
            #     [img.ravel() for img in a]
            # )
            # print("min = %f, max = %f" % (np.min(flat), np.max(flat)))
            # b = F.to_tensor(img)
            # a = b
            # a = np.array(a)
            # flat = np.concatenate(
            #     [img.ravel() for img in a]
            # )
            # print("min = %f, max = %f" % (np.min(flat), np.max(flat)))
            img, target = self.transforms(img, target)


        return img, target

    def __len__(self):
        return len(self.imgs)