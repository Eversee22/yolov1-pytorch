from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import random


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2, mode=0):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    if mode == 1:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def bbox_iou2(box1,box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = np.max([b1_x1, b2_x2])
    inter_rect_y1 = np.max([b1_y1, b2_y1])
    inter_rect_x2 = np.min([b1_x2, b2_x2])
    inter_rect_y2 = np.min([b1_y2, b2_y2])

    # Intersection area
    inter_area = np.max([inter_rect_x2 - inter_rect_x1, 0]) * np.max([inter_rect_y2 - inter_rect_y1, 0])

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def convert_box(box, h, w, mode=0):
    if mode == 1:
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    else:
        xmin, ymin, xmax, ymax = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > w-1: xmax = w-1
    if ymax > h-1: ymax = h-1

    return xmin, ymin, xmax, ymax


def imwrite(image, bbox, class_name, colors, prob=None):
    """

    :param image:
    :param bbox: (4,int)
    :param class_name: the corresponding class name
    :param colors: colors collection
    :param prob: predicted class probability
    :return: boxed image
    """
    # if mode == 1:
    # xmin,ymin,xmax,ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # if xmin<0: xmin = 0
    # if ymin<0: ymin = 0
    # if xmax
    c1 = (bbox[0], bbox[1])
    c2 = (bbox[2], bbox[3])
    # c1 = tuple(bbox[0:2].int())
    # c2 = tuple(bbox[2:4].int())
    img = image
    color = random.choice(colors)
    label = "{0}".format(class_name)
    if prob is not None:
        label += str(round(prob, 2))
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

    return img


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(names_file):
    with open(names_file) as f:
        names = f.readlines()
        names = [name.strip() for name in names]
        names = [name for name in names if len(name) > 0]

    return names


def readcfg(cfg_file):
    d = {}
    with open(cfg_file) as f:
        for l in f:
            key,val = l.strip().split('=')
            key = key.rstrip()
            val = val.lstrip()
            if val[0] == '[':
                val = val[1:val.find(']')]
                val = [v for v in val.split(',')]
            d[key] = val
    return d

if __name__ == '__main__':
    d = readcfg('cfg/yolond')
    for k in d:
        print(k,d[k])
