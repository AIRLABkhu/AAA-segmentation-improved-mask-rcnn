#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files
Use this script as an example to build your own tool
"""

import argparse
import os
import time
import cv2


from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output

import caffe
from caffe.proto import caffe_pb2
import cv2


def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net
    Arguments:
    caffemodel -- path to a .caffemodel file1v
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()


    network = caffe.Net(deploy_file, caffemodel, caffe.TEST)

    network = caffe.Net('/home/bh/docker_file/pretrain/deploy.prototxt', '/home/bh/docker_file/pretrain/snapshot_iter_218052.caffemodel',caffe.TEST)

    return network

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """

    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)

    # image = np.array(cv2.imread(img_file))
    # squash
    # image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = images
    # for image in images:
    #     caffe_images.append(image)
    #     # if image.ndim == 2:
    #     #     caffe_images.append(image[:,:,np.newaxis])
    #     # else:
    #     #     caffe_images.append(image)


    dims = transformer.inputs['data'][1:]

    scores = None

    caffe_images = caffe_images.reshape(3,512,512)
    chunk = caffe_images[np.newaxis, :, :, :]


    for index, image in enumerate(chunk):
        print(index)
        image_data = transformer.preprocess('data', image)
        net.blobs['data'].data[index] = image_data

    output = net.forward()[net.outputs[-1]]

    if scores is None:
        scores = np.copy(output)
    else:
        scores = np.vstack((scores, output))


    return scores

def read_labels(labels_file):
    """
    Returns a list of strings
    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print('WARNING: No labels file provided. Results will be difficult to interpret.')
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def classify(caffemodel, deploy_file, image_files,
        mean_file=None, labels_file=None, batch_size=None, use_gpu=False):
    """
    Classify some images against a Caffe model and print the results
    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images
    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)

    transformer = get_transformer(deploy_file, mean_file)

    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = load_image(image_files, height, width, mode)
    # images = image_files

    labels = read_labels(labels_file)

    # Classify the image
    scores = forward_pass(images, net, transformer, batch_size=batch_size)

    ### Process the results

    # Format of scores is [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    # https://github.com/NVIDIA/caffe/blob/v0.15.13/python/caffe/layers/detectnet/clustering.py#L81

    for i, image_results in enumerate(scores):
        print('==> Image #%d' % i)
        for left, top, right, bottom, confidence in image_results:
            if confidence == 0:
                continue

            print('Detected object at [(%d, %d), (%d, %d)] with "confidence" %f' % (
                int(round(left)),
                int(round(top)),
                int(round(right)),
                int(round(bottom)),
                confidence,
            ))


if __name__ == '__main__':
    caffemodel = '/home/bh/Downloads/20210907-110948-9ab6_epoch_324.0/snapshot_iter_218052.caffemodel'
    deploy_file = '/home/bh/Downloads/20210907-110948-9ab6_epoch_324.0/deploy.prototxt'

    # img_file = cv2.imread('/home/bh/Downloads/0906_modify_full_contrast/0906_rename_for_bh/Fold/fold2_test/16_0066.png')
    img_file = '/home/bh/Downloads/0906_modify_full_contrast/0906_rename_for_bh/Fold/fold2_test/16_0066.png'

    mean_file = '/home/bh/Downloads/20210907-110948-9ab6_epoch_324.0/mean.binaryproto'

    # classify(caffemodel=caffemodel, deploy_file=deploy_file, image_files=img_file, mean_file=mean_file)
    classify(caffemodel=caffemodel, deploy_file=deploy_file, image_files=img_file)
