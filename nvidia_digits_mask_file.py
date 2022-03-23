from PIL import Image
import os
import json
import numpy as np
import shutil
import random

size = [1280, 720]
val_ratio = 0.25

if __name__ =="__main__":
    print("Create nvidia digits data label")

    species = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    # work_dir = './data/fish_train'
    #
    # img_widths = []
    # img_heights = []
    # box_widths = []
    # box_heights = []
    #
    # if not os.path.exists(os.path.join('./data', 'train_1280*720')):
    #     os.makedirs(os.path.join('./data', 'train_1280*720'))
    #
    # for spec in species:
    #     input_dir = os.path.join(work_dir, spec)
    #     output_dir = os.path.join('./data', 'train_1280*720', spec)
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     json_file = spec + '_labels.json'
    #
    #     with open(os.path.join(input_dir, json_file)) as json_data:
    #         d = json.load(json_data)
    #
    #     for i in range(len(d)):
    #         # Resize image
    #         # image name
    #         fn = str(d[i]['filename'])
    #         # open image
    #         im = Image.open(os.path.join(input_dir, fn))
    #
    #         # Pick the side which needs to be scaled down the most
    #         scale = min(float(size[0]) / im.size[0], float(size[1]) / im.size[1])
    #         im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)),
    #                        Image.ANTIALIAS)
    #         # Padding on either right or bottom if necessary
    #         im = im.crop((0, 0, size[0], size[1]))
    #         resized_path = './data/train_1280*720/' + spec + '/' + fn
    #         # Save the resized image
    #         im.save(resized_path)
    #
    #         # Modify bounding boxes to match image scaling
    #         for j in range(len(d[i]['annotations'])):
    #             x = d[i]['annotations'][j]['x'] * scale
    #             y = d[i]['annotations'][j]['y'] * scale
    #             w = d[i]['annotations'][j]['width'] * scale
    #             h = d[i]['annotations'][j]['height'] * scale
    #             # Fix to avoid x, y, w, h out of bound
    #             if x < 0:
    #                 w += x
    #                 if w < 0: w = 20  # this should not happen
    #                 x = 0
    #             if y < 0:
    #                 h += y
    #                 if h < 0: h = 20  # this should not happen
    #                 y = 0
    #             if x + w > size[0]:
    #                 w = size[0] - x
    #             if y + h > size[1]:
    #                 h = size[1] - y
    #             d[i]['annotations'][j]['x'] = x
    #             d[i]['annotations'][j]['y'] = y
    #             d[i]['annotations'][j]['width'] = w
    #             d[i]['annotations'][j]['height'] = h
    #             box_widths.append(w)
    #             box_heights.append(h)
    #
    #     # Save the updated JSON file
    #     print('./data/train_1280*720/' + spec + '/' + spec +'_1280x720_labels.json')
    #     with open('./data/train_1280*720/' + spec + '/' + spec +'_1280x720_labels.json', 'w') as fp:
    #         json.dump(d, fp, indent=0)

    #######################################################################################################
    # Chapter #2
    work_dir = './data/'
    if os.path.exists(work_dir + 'for_detectnet/'):
        shutil.rmtree(work_dir + 'for_detectnet/')
    os.makedirs(work_dir + 'for_detectnet/train/images/')
    os.makedirs(work_dir + 'for_detectnet/train/labels/')
    os.makedirs(work_dir + 'for_detectnet/val/images/')
    os.makedirs(work_dir + 'for_detectnet/val/labels/')

    for spec in species:
        print('*** ' + spec + ' ***')
        input_dir = '/home/bh/Downloads/aaa_segmentation/data/train_1280x720/' + spec + '/'
        json_file = spec + '_1280x720_labels.json'
        json_path = os.path.join('/home/bh/Downloads/aaa_segmentation/data/train_1280*720',spec,json_file)

        with open(json_path) as json_data:
            d = json.load(json_data)

        for i in range(len(d)):
            output_dir = work_dir + 'for_detectnet/train/'
            if spec != 'NoF':
                if random.random() < val_ratio:
                    output_dir = work_dir + 'for_detectnet/val/'
            # Copy the image over
            fn = str(d[i]['filename'])
            shutil.copy(os.path.join('./data/train_1280*720', spec, fn), output_dir + 'images/')
            fnbase, ext = os.path.splitext(fn)
            # One Label file per one image
            with open(output_dir + 'labels/' + fnbase + '.txt', 'w') as fp:
                # Convert annotations to required format
                for j in range(len(d[i]['annotations'])):
                    l = d[i]['annotations'][j]['x']
                    t = d[i]['annotations'][j]['y']
                    r = l + d[i]['annotations'][j]['width']
                    b = t + d[i]['annotations'][j]['height']

                    if spec != 'NoF':
                        type = 'Car'
                        truncated = 0
                        occluded = 3
                        alpha = 0
                        tail = '0 0 0 0 0 0 0 0'
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

    #######################################################################################################
    # Chapter #3
    # Make digits label format using our format

