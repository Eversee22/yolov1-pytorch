from ftd_model import get_model_ft
import torch
import cv2
import numpy as np
from util import imwrite, convert_box,prep_image
import argparse
import sys
from util import load_classes, readcfg
import _pickle as pkl
from util import bbox_iou2
import os
from tqdm import tqdm
import time
import random

d = readcfg('cfg/yolond')
side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
inp_size = int(d['inp_size'])

# colors = pkl.load(open('pallete', 'rb'))
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
        # print(ovr)
        ids = (ovr <= thres).nonzero().squeeze()

        if ids.numel() == 0:
            break
        order = order[ids+1]

    return torch.LongTensor(keep)


def get_detection_boxes_1(pred, prob_thres,nms_thresh, nms=True):
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
        keep = do_nms_1(boxes, probs, nms_thresh)
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
                # else:
                #     print(iou)


def get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs, nms=True):
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
        do_nms(boxes, probs, nms_thresh)

    # return boxes, probs


def load_model(model_name, weight, mode):
    model = get_model_ft(model_name, False)
    assert model is not None

    if mode == 1:
        # checkpoint = torch.load(weight)
        model.load_state_dict(torch.load(weight)['model'])
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


def predict_gpu_1(model_name, image_name, weight, mode=1):
    root_dir = '/home/blacksun2/github/darknet-2016-11-22/VOCdevkit/AllImages'
    result = []
    model = load_model(model_name, weight, mode)
    model.cuda()
    image = cv2.imread(os.path.join(root_dir, image_name))
    h, w, _ = image.shape
    img = img_trans(image)
    img = img.cuda()
    with torch.no_grad():
        pred = model(img)
    # pred = pred.cpu()

    probs = np.zeros((side * side * num, classes))
    boxes = np.zeros((side * side * num, 4))
    get_detection_boxes(pred, 0.1, 0.5, boxes, probs)

    for i in range(probs.shape[0]):
        box = boxes[i]
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        if x1<0:x1=0
        if x2>w:x2=w
        if y1<0:y1=0
        if y2>h:y2=h
        for j in range(classes):
            if probs[i,j] > 0:
                result.append([(x1,y1),(x2,y2),voc_class_names[j],image_name,probs[i,j]])

    return result


def print_yolo_detections(test_file, model_name, weight,use_gpu=True):
    with open(test_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    model = load_model(model_name, weight, 1)
    if use_gpu:
        model.cuda()
    if not os.path.exists('results'):
        os.mkdir('results')
    base = 'results/comp4_det_test'
    fps = []
    for k in range(classes):
        fps.append(open('%s_%s' % (base, voc_class_names[k]), 'w'))

    for imp in tqdm(lines):
        imgid = imp.split('/')[-1].split('.')[0]
        # print(imgid)
        img = cv2.imread(imp)
        h, w, _ = img.shape
        img = img_trans(img)
        if use_gpu:
            img = img.cuda()
        with torch.no_grad():
            pred = model(img)

        # pred = pred.cpu()
        probs = np.zeros((side * side * num, classes))
        boxes = np.zeros((side * side * num, 4))
        get_detection_boxes(pred, 0.1, 0.5, boxes, probs)
        for i in range(probs.shape[0]):
            box = boxes[i]
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            if x1 < 0: x1 = 0
            if x2 > w: x2 = w
            if y1 < 0: y1 = 0
            if y2 > h: y2 = h
            for j in range(classes):
                if probs[i, j] > 0:
                    fps[j].write('%s %f %f %f %f %f\n' % (imgid, probs[i][j], x1, y1, x2, y2))
            # fps[j].flush()

    for f in fps:
        f.close()


def test_canvas(model_name, image_name, weight, prob_thresh=0.2, nms_thresh=0.4, mode=1,use_gpu=True):
    print('load weight')
    model = load_model(model_name, weight, mode)
    if use_gpu:
        model.cuda()
    print("detecting")
    image = cv2.imread(image_name)
    img = prep_image(image, inp_size)
    im_dim = image.shape[1], image.shape[0]  # w,h
    im_dim = np.array(im_dim)

    output = []
    if use_gpu:
        # im_dim = im_dim.cuda()
        img = img.cuda()
    with torch.no_grad():
        pred = model(img)
    probs = np.zeros((side * side * num, classes))
    boxes = np.zeros((side * side * num, 4))
    get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs)
    # im_dim = torch.FloatTensor(im_dim).repeat(1, 2).numpy()
    scaling_factor = np.min(inp_size / im_dim)

    for i in range(probs.shape[0]):
        cls = np.argmax(probs[i])
        prob = probs[i][cls]
        if prob > 0:
            out = np.zeros(6)
            out[:4] = boxes[i]*inp_size
            out[[0,2]] -= (inp_size-scaling_factor*im_dim[0])/2
            out[[1,3]] -= (inp_size-scaling_factor*im_dim[1])/2

            out[:4] = out[:4]/scaling_factor
            out[[0,2]] = np.clip(out[[0,2]],0.0,im_dim[0])
            out[[1,3]] = np.clip(out[[1,3]],0.0,im_dim[1])

            out[4] = prob
            out[5] = cls
            output.append(out)

    for item in output:
        # item = output[i]
        cls = int(item[-1])
        prob = float(item[-2])
        box = item[:4]
        image = imwrite(image, box, voc_class_names[cls], cls, prob)

    cv2.imshow("{}".format(image_name), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return output


def test_many(model_name,test_file,weight, prob_thresh=0.2, nms_thresh=0.4, mode=1,use_gpu=True):
    print('load weight')
    model = load_model(model_name, weight, mode)
    if use_gpu:
        model.cuda()
    # model.eval()
    # model = model.to(device)
    print("detecting")
    images= []
    try:
        images = [os.path.join(test_file, img) for img in os.listdir(test_file)]
    except NotADirectoryError:
        try:
            with open(test_file) as f:
                for l in f:
                    images.append(l.strip())
        except Exception as e:
            print(e)
            exit()
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()
    if not os.path.exists('det'):
        os.mkdir('det')
    random.shuffle(images)
    for imp in images:
        image = cv2.imread(imp)
        h, w, _ = image.shape
        img = img_trans(image)
        if use_gpu:
            img = img.cuda()
        pred = model(img)
        probs = np.zeros((side * side * num, classes))
        boxes = np.zeros((side * side * num, 4))
        get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs)
        for i in range(probs.shape[0]):
            cls = np.argmax(probs[i])
            prob = probs[i][cls]
            if prob > 0:
                box = boxes[i]
                image = imwrite(image, convert_box(box, h, w, inp_size), voc_class_names[cls], cls, prob)

        cv2.imshow("img", image)
        cv2.imwrite('det/bbox_%s.png' % imp.split('/')[-1].split('.')[0], image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyWindow('img')
            break
        # elif key & 0xFF == ord('s'):
        #     cv2.imwrite('det/bbox_%s' % imp.split('/')[-1], image)
        time.sleep(2)


def test(model_name, image_name, weight, prob_thresh=0.2, nms_thresh=0.4, mode=1,use_gpu=True):

    print('load weight')
    model = load_model(model_name, weight, mode)
    if use_gpu:
        model.cuda()
    # model.eval()
    # model = model.to(device)
    print("detecting")
    image = cv2.imread(image_name)

    h, w, _ = image.shape
    img = img_trans(image)
    if use_gpu:
        img = img.cuda()
    pred = model(img)

    if 1:
        # default method
        probs = np.zeros((side * side * num, classes))
        boxes = np.zeros((side * side * num, 4))
        get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs)
        for i in range(probs.shape[0]):
            cls = np.argmax(probs[i])
            prob = probs[i][cls]
            if prob > 0:
                box = boxes[i]
                image = imwrite(image, convert_box(box, h, w, inp_size), voc_class_names[cls], cls, prob)
    else:
        # another method, in problem
        if use_gpu:
            pred = pred.cpu()
        boxes, probs, cls_indices = get_detection_boxes_1(pred, 0.2, 0.4)
        for i, box in enumerate(boxes):
            box = box * torch.FloatTensor([w, h, w, h])
            cls_index = cls_indices[i].item()
            # print(cls_index)
            prob = probs[i].item()
            image = imwrite(image, box, voc_class_names[cls_index], cls_index, prob)
    cv2.imwrite('bbox_%s.png' % image_name.split('/')[-1].split('.')[0], image)
    cv2.imshow("{}".format(image_name), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def arg_parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-t", dest="target", default="test", help="[test|eval|test2|test3]", type=str)
    arg_parser.add_argument("-i", dest="imgn", help="image name", type=str)
    arg_parser.add_argument("-m",dest="mn", default="resnet50", help="model name", type=str)
    arg_parser.add_argument("--mode", dest="mode", default=1, help="model save mode", type=int)
    arg_parser.add_argument("--nms", dest="nms", default=0.45, help="nms thresh", type=float)
    arg_parser.add_argument("--thresh", dest="thresh", default=0.2, help="confidence thresh", type=float)
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
    # num = args.num
    nms = args.nms
    thresh = args.thresh
    weight = args.weight[0]

    if do == "test":
        test(model_name, image_name, weight,thresh,nms, mode)
    elif do == "test2":
        test_many(model_name, image_name, weight, thresh, nms, mode)
    elif do == "test3":
        test_canvas(model_name, image_name, weight, thresh, nms, mode)
    elif do == "eval":
        print_yolo_detections(image_name,model_name,weight)
