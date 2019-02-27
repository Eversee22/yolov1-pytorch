from ftd_model import get_model_ft
import torch
import cv2
import numpy as np
from util import imwrite, convert_box
# import matplotlib.pyplot as plt
import argparse
import sys
from util import load_classes, readcfg
import _pickle as pkl
from util import bbox_iou2
import os
import time
import threading

d = readcfg('cfg/yolond')
side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
inp_size = int(d['inp_size'])

colors = pkl.load(open('pallete', 'rb'))
voc_class_names = load_classes('data/voc.names')


def do_nms_1(boxes,probs,thres=0.5):
    """
    boxes: (tensor)[N,4]
    probs: (tensor)[N,]
    return: the keeping indices
    """
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)

    _, order = probs.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        #keep the max
        # print(order)
        if len(order.shape) > 0:
            i = order[0]
        else:
            i = order  # 0 dim
        keep.append(i)

        #NMS others
        if order.numel() == 1:
            break

        _x1 = x1[order[1:]].clamp(min=x1[i].item())
        _y1 = y1[order[1:]].clamp(min=y1[i].item())
        _x2 = x2[order[1:]].clamp(max=x2[i].item())
        _y2 = y2[order[1:]].clamp(max=x2[i].item())

        w = (_x2 - _x1).clamp(min=0)
        h = (_y2 - _y1).clamp(min=0)
        inter = w*h

        ovr = inter/(areas[i]+areas[order[1:]]-inter)
        print(ovr)
        ids = (ovr <= thres).nonzero().squeeze()

        if ids.numel() == 0:
            break
        order = order[ids+1]

    return torch.LongTensor(keep)


def get_detection_boxes_1(pred, prob_thres,nms=True):
    """
    pred: (1,side,side,(num*5+20))
    return: probs,boxes,cls
    """
    boxes = []
    probs = []
    cls_indices = []
    # grid_num= side
    pred = pred.squeeze(0)
    contain = []
    for i in range(num):
        contain.append(pred[:, :, i*5+4].unsqueeze(2))
    contain = torch.cat(contain, 2)
    # print(torch.sum(contain > 0))
    mask1 = contain > 0.1
    mask2 = contain == contain.max()
    mask = (mask1 + mask2).gt(0)
    print(torch.sum(mask))

    for i in range(side):
        for j in range(side):
            for k in range(num):
                if mask[i, j, k] == 1:
                    box = pred[i, j, k*5:k*5+4]
                    objc = torch.FloatTensor([pred[i,j,k*5+4]])
                    xy = torch.FloatTensor([j,i])
                    box[:2] = (box[:2]+xy)/side
                    box2 = torch.FloatTensor(box.size())
                    box2[:2] = box[:2] - 0.5*box[2:]
                    box2[2:] = box[:2] + 0.5*box[2:]
                    max_prob, cls_ind = torch.max(pred[i, j, num*5:], 0)
                    c_prob = objc*max_prob
                    if c_prob[0].item() > prob_thres:
                        boxes.append(box2.view(1,4))
                        cls_indices.append(torch.LongTensor([cls_ind]))
                        probs.append(c_prob)

    print(len(boxes))
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indices = torch.LongTensor([0])
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indices = torch.cat(cls_indices, 0)  # (n,)

    if nms:
        keep = do_nms_1(boxes, probs, 0.4)
        return boxes[keep], probs[keep], cls_indices[keep]
    else:
        return boxes, probs, cls_indices


def do_nms(boxes, probs, thresh=0.4):
    """
    boxes: [side*side*num,4]
    probs: [side*side*num,20]

    """
    total = side*side*num
    for k in range(classes):
        order = np.argsort(probs[:, k])[::-1]  # descending order
        for i in range(total):
            if probs[order[i]][k] == 0:
                continue
            box_a = boxes[order[i]]
            for j in range(i+1, total):
                box_b = boxes[order[j]]
                iou = bbox_iou2(box_a, box_b)
                if iou > thresh:
                    probs[order[j]][k] = 0


def get_detection_boxes(pred, prob_thresh, boxes, probs, nms=True):
    """
    pred: (1, side, side, num*5+20)
    return: probs, boxes
    """
    pred = pred.squeeze(0).data.cpu().numpy()
    # probs = np.zeros((side*side*num, classes))
    # boxes = np.zeros((side*side*num, 4))

    for g in range(side*side):
        i = g // side
        j = g % side
        for k in range(num):
            obj = pred[i, j, k*5+4]
            box = pred[i, j, k*5:k*5+4]
            cr = np.array([j, i], dtype=np.float32)
            xy = (box[:2] + cr) / side
            boxes[g*num + k][:2] = xy - 0.5 * box[2:]
            boxes[g*num + k][2:] = xy + 0.5 * box[2:]

            for z in range(classes):
                prob = obj * pred[i, j, num * 5 + z]
                probs[g*num+k][z] = prob if prob > prob_thresh else 0

    if nms:
        do_nms(boxes, probs)

    # return boxes, probs


def load_model(model_name, weight, mode, num):
    model = get_model_ft(model_name, False, num=num)
    assert model is not None

    if mode == 1:
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(torch.load(weight))
    model.eval()

    return model


def img_trans(img):
    # img = cv2.imread(img_name)

    img = cv2.resize(img, (inp_size, inp_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)

    return img


# def convert_box(box, h, w):
#     xmin, ymin, xmax, ymax = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
#
#     if xmin < 0: xmin = 0
#     if ymin < 0: ymin = 0
#     if xmax > w-1: xmax = w-1
#     if ymax > h-1: ymax = h-1
#
#     return xmin, ymin, xmax, ymax


def test(model_name, image_name, weight, mode=0, num=2):
    print('load weight')
    model = load_model(model_name,weight,mode,num)
    # model.eval()
    # model = model.to(device)

    print("detecting")

    image = cv2.imread(image_name)
    h, w, _ = image.shape

    img = img_trans(image.copy())
    # img = img.to(device)

    pred = model(img)

    thresh = 0.02

    if 0:
        boxes, probs, cls_indices = get_detection_boxes_1(pred, thresh)
        for i, box in enumerate(boxes):
            box = box * torch.FloatTensor([w, h, w, h])
            cls_index = cls_indices[i].item()
            # print(cls_index)
            prob = probs[i].item()
            image = imwrite(image, box, voc_class_names[cls_index], colors, prob)
    else:
        probs = np.zeros((side * side * num, classes))
        boxes = np.zeros((side*side*num, 4))
        get_detection_boxes(pred, thresh, boxes, probs)
        for i in range(probs.shape[0]):
            cls = np.argmax(probs[i])
            prob = probs[i][cls]
            if prob > 0:
                box = boxes[i]
                image = imwrite(image, convert_box(box, h, w), voc_class_names[cls], colors, prob)

    cv2.imshow("{}".format(image_name), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.title("{}".format(image_name))
    # plt.imshow(image)
    # plt.show()


# def write_detections(fps, boxes, probs, h, w, imgid):
#
#     for i in range(probs.shape[0]):
#         box = boxes[i]
#         xmin, ymin, xmax, ymax = box[0] * w, box[1] * h, box[2] * w, box[3] * h
#
#         if xmin < 0:
#             xmin = 0
#         if ymin < 0:
#             ymin = 0
#         if xmax > w:
#             xmax = w
#         if ymax > h:
#             ymax = h
#         for j in range(classes):
#             if probs[i][j] > 0:
#                 fps[j].write('%s %f %f %f %f %f\n' % (imgid, probs[i][j], xmin, ymin, xmax, ymax))
def eval_func(model, id, in_buff, out_buff):
    ins = in_buff[id]
    boxes = np.zeros((side * side * num, 4))
    probs = np.zeros((side * side * num, classes))

    for i in ins:
        img = cv2.imread(i)
        h, w, _ = img.shape
        imgid = i.split('/')[-1].split('.')[0]
        print("thread %d, %s"%(id,imgid))
        img = img_trans(img)
        pred = model(img)
        # print("get boxes")
        get_detection_boxes(pred, 0.002, boxes, probs)
        for i in range(probs.shape[0]):
            box = boxes[i]
            xmin, ymin, xmax, ymax = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmax > w: xmax = w
            if ymax > h: ymax = h
            for j in range(classes):
                if probs[i][j] > 0:
                    out_buff[id].append('%d %s %f %f %f %f %f' % (j, imgid, probs[i][j], xmin, ymin, xmax, ymax))
                    # print('[%d] %s %f %f %f %f %f' % (id, imgid, probs[i][j], xmin, ymin, xmax, ymax))


class EvalThread (threading.Thread):
    def __init__(self, threadID, model, in_buff, out_buff, name=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.in_buff = in_buff
        self.out_buff = out_buff
        self.model = model
        # self.probs = np.zeros((side * side * num, classes))
        # self.boxes = np.zeros((side * side * num, 4))

    def run(self):
        print("Starting thread %d" % self.threadID)
        eval_func(self.model, self.threadID, self.in_buff, self.out_buff)
        print('thread %d over' % self.threadID)


def eval_out(test_file, model_name, weight, mode=0, num=2):
    with open(test_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line)>0]

    model = load_model(model_name,weight,mode,num)
    if not os.path.exists('results'):
        os.mkdir('results')
    base = 'results/comp4_det_test'
    fps = []
    for k in range(classes):
        fps.append(open('%s_%s' % (base, voc_class_names[k]), 'w'))

    use_thread = False

    if use_thread:
        nThreads = 4
        tl = len(lines)
        print(tl)
        n = tl // nThreads
        r = tl % nThreads
        in_buff = [lines[i*n:(i+1)*n] for i in range(nThreads)]
        if r > 0:
            in_buff[0].append(lines[-r:])
        print(len(in_buff[0]))
        out_buff = [[]]*nThreads
        threads = []
        since = time.time()
        for i in range(nThreads):
            thread = EvalThread(i, model, in_buff, out_buff)
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()
        print("all get")
        # for out in out_buff:
        #     for l in out:
        #         pos = l.find(' ')
        #         cls = int(l[:pos])
        #         fps[cls].write(l[pos+1:])
    else:
        boxes = np.zeros((side * side * num, 4))
        probs = np.zeros((side * side * num, classes))
        tl = len(lines)
        since = time.time()
        for i,l in enumerate(lines):
            print('%d/%d'%(i+1,tl))
            imgid = l.split('/')[-1].split('.')[0]
            img = cv2.imread(l)
            h, w, _ = img.shape
            img = img_trans(img)
            pred = model(img)
            get_detection_boxes(pred, 0.001, boxes, probs)
            for i in range(probs.shape[0]):
                box = boxes[i]
                xmin, ymin, xmax, ymax = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                if xmin < 0: xmin = 0
                if ymin < 0: ymin = 0
                if xmax > w: xmax = w
                if ymax > h: ymax = h
                for j in range(classes):
                    if probs[i][j] > 0:
                        fps[j].write('%s %f %f %f %f %f\n' % (imgid, probs[i][j], xmin, ymin, xmax, ymax))

    print('spend {}s'.format(time.time() - since))

    for fp in fps:
        fp.close()


def arg_parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-t", dest="target", default="test", help="[test|eval]", type=str)
    arg_parser.add_argument("-i", dest="imgn", help="image name", type=str)
    arg_parser.add_argument("-m",dest="mn", default="vgg16", help="model name", type=str)
    arg_parser.add_argument("--mode", dest="mode", default=0, help="model save mode", type=int)
    arg_parser.add_argument("--num", dest="num", default=2, help="box number", type=int)
    arg_parser.add_argument("weight", nargs=1, help="weight file", type=str)

    if len(sys.argv) < 2:
        arg_parser.print_help()
        exit(0)

    args = arg_parser.parse_args()

    return args


if __name__ == '__main__':
    args = arg_parse()
    do = args.target
    image_name = args.imgn
    model_name = args.mn
    mode = args.mode
    num = args.num
    weight = args.weight[0]

    if do == "test":
        test(model_name, image_name, weight, mode=mode, num=num)

    elif do == "eval":
        eval_out(image_name, model_name, weight, mode=mode, num=num)
