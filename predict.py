import torch
from util import readcfg, bbox_iou, bbox_iou2,load_classes
from ftd_model import get_model_ft
import numpy as np
import cv2
import time
import os
from tqdm import tqdm
from util import convert_box

d = readcfg('cfg/yolond')
side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
inp_size = int(d['inp_size'])
voc_class_names = load_classes('data/voc.names')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_name, weight, mode):
    model = get_model_ft(model_name, False)
    assert model is not None

    if mode == 1:
        # checkpoint = torch.load(weight)
        model.load_state_dict(torch.load(weight,map_location=device)['model'])
    else:
        model.load_state_dict(torch.load(weight,map_location=device))
    model.eval()

    return model


def img_trans(img):
    # img = cv2.imread(img_name)

    img = cv2.resize(img, (inp_size, inp_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0)
    img = img.unsqueeze(0)

    return img


def do_nms_1(boxes,probs,thres=0.5):
    """
    boxes: (tensor)[N,4]
    probs: (tensor)[N,]
    return: the keeping indices
    """
    # print(probs.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = probs.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        # print(order)
        #keep the max
        try:
            i = order[0]
        except IndexError:
            i = order
        keep.append(i)

        if order.numel() == 1:
            break

        # NMS others
        _x1 = x1[order[1:]].clamp(min=x1[i])
        _y1 = y1[order[1:]].clamp(min=y1[i])
        _x2 = x2[order[1:]].clamp(max=x2[i])
        _y2 = y2[order[1:]].clamp(max=y2[i])

        w = (_x2 - _x1).clamp(min=0)
        h = (_y2 - _y1).clamp(min=0)

        inter = w * h

        # ovr = bbox_iou(boxes[order[1:]], boxes[i].unsqueeze(0))
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # print(ovr)
        ids = (ovr <= thres).nonzero().squeeze()
        # print('ids:',ids)

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
    pred = pred.data.squeeze(0)
    contain = []
    for i in range(num):
        contain.append(pred[:, :, i*5+4].unsqueeze(2))
    contain = torch.cat(contain, 2)
    # print(torch.sum(contain > 0))
    mask1 = contain > 0.01
    mask2 = contain == contain.max()
    mask = (mask1 + mask2).gt(0)
    # print(torch.sum(mask))

    for i in range(side):
        for j in range(side):
            for k in range(num):
                if mask[i, j, k] == 1:
                    objc = torch.FloatTensor([pred[i,j,k*5+4]])
                    xy = torch.FloatTensor([j,i])
                    max_prob, cls_ind = torch.max(pred[i, j, num*5:].view(-1, 1), 0)
                    c_prob = objc*max_prob
                    if c_prob.item() > prob_thres:
                        box = pred[i, j, k*5:k*5+4]
                        box[:2] = (box[:2]+xy)/side
                        box2 = torch.FloatTensor(box.size())
                        box2[:2] = box[:2] - 0.5*box[2:]
                        box2[2:] = box[:2] + 0.5*box[2:]
                        boxes.append(box2.view(1,4))
                        cls_indices.append(cls_ind)
                        # print(c_prob)
                        probs.append(c_prob)

    # print(len(boxes))
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indices = torch.LongTensor([0])
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indices = torch.cat(cls_indices, 0)  # (n,)

    if nms:
        since = time.time()
        keep = do_nms_1(boxes, probs, nms_thresh)
        print("do nms 1:%fs" % (time.time()-since))
        return boxes[keep], probs[keep], cls_indices[keep]
    else:
        return boxes, probs, cls_indices


def bboxes_iou_np(box_a, box_b):
    ixmin = np.maximum(box_b[:, 0] - 0.5 * box_b[:, 2], box_a[0] - 0.5 * box_a[2])
    iymin = np.maximum(box_b[:, 1] - 0.5 * box_b[:, 3], box_a[1] - 0.5 * box_a[3])
    ixmax = np.minimum(box_b[:, 0] + 0.5 * box_b[:, 2], box_a[0] + 0.5 * box_a[2])
    iymax = np.minimum(box_b[:, 1] + 0.5 * box_b[:, 3], box_a[1] + 0.5 * box_a[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih
    uni = ((box_a[2]) * (box_a[3]) +
           (box_b[:, 2]) * (box_b[:, 3]) - inters)
    overlaps = inters / uni
    return overlaps


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
            box_b = boxes[order[i+1:]]
            overlaps = bboxes_iou_np(box_a,box_b)
            for ind in order[i+1:][overlaps>thresh]:
                probs[ind][k] = 0
            # for j in range(i+1, total):
            #     box_b = boxes[order[j]]
            #     iou = bbox_iou2(box_a, box_b)
            #     if iou > thresh:
            #         probs[order[j]][k] = 0
                # else:
                #     print(iou)


def get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs, nms=True):
    """
    pred: (1, side, side, num*5+20)
    return: probs, boxes
    """
    pred = pred.data.squeeze(0).numpy()
    # probs = np.zeros((side*side*num, classes))
    # boxes = np.zeros((side*side*num, 4))

    for g in range(side*side):
        i = g // side
        j = g % side
        for k in range(num):
            obj = pred[i, j, k*5+4]
            if obj <= prob_thresh:
                continue
            box = pred[i, j, k*5:k*5+4]
            cr = np.array([j, i], dtype=np.float32)
            xy = (box[:2] + cr) / side
            #boxes[g*num + k][:2] = xy - 0.5 * box[2:]
            #boxes[g*num + k][2:] = xy + 0.5 * box[2:]
            boxes[g*num + k][:2] = xy
            boxes[g*num + k][2:] = box[2:]

            for z in range(classes):
                prob = obj * pred[i, j, num * 5 + z]
                probs[g*num+k][z] = prob if prob > prob_thresh else 0
    if nms:
        #since = time.time()
        do_nms(boxes, probs, nms_thresh)
        #print("do nms:%f"%(time.time()-since))
    # return boxes, probs


def get_pred_1(image,model_name,weight,mode=1,use_gpu=True):
    model = load_model(model_name, weight, mode)
    if use_gpu:
        model.cuda()
    # h, w, _ = image.shape
    img = img_trans(image)
    if use_gpu:
        img = img.cuda()
    with torch.no_grad():
        pred = model(img)
    if use_gpu:
        pred = pred.cpu()

    return pred


def get_pred(image,model,use_gpu=True):
    # if use_gpu:
    #     model.cuda()
    img = img_trans(image)
    if use_gpu:
        img = img.cuda()
    with torch.no_grad():
        pred = model(img)
    if use_gpu:
        pred = pred.cpu()

    return pred


def get_test_result_1(model_name, image_name, weight, prob_thresh=0.2, nms_thresh=0.4, mode=1, use_gpu=True):
    image = cv2.imread(image_name)
    h, w, _ = image.shape
    pred = get_pred_1(image,model_name,weight,mode,use_gpu)
    since = time.time()
    boxes, probs, clsinds = get_detection_boxes_1(pred, prob_thresh, nms_thresh)
    print('get detection boxes 1 {:f}s'.format(time.time() - since))
    output = []
    write = 0
    for i, box in enumerate(boxes):
        if probs[i] == 0:
            continue
        box = torch.FloatTensor(convert_box(box,h,w))
        prob = probs[i].unsqueeze(0)
        cls_ind = clsinds[i].unsqueeze(0).float()
        if write==0:
            output = torch.cat((box,prob,cls_ind)).unsqueeze(0)
            write = 1
        else:
            out = torch.cat((box,prob,cls_ind)).unsqueeze(0)
            output = torch.cat((output,out))
    return output, image

    # return boxes, probs, clsinds


def correct_box(box,h,w):
    boxout = box.copy()
    boxout[:2] = box[:2] - 0.5 * box[2:]
    boxout[2:] = box[:2] + 0.5 * box[2:]
    boxout[[0, 2]] *= w
    boxout[[1, 3]] *= h
    if boxout[0] < 0: boxout[0] = 0
    if boxout[1] < 0: boxout[1] = 0
    if boxout[2] > w - 1: boxout[2] = w - 1
    if boxout[3] > h - 1: boxout[3] = h - 1
    return boxout


def correct_boxes(boxes,h,w):
    boxesout = boxes.copy()
    boxesout[:, :2] = boxes[:, :2] - 0.5 * boxes[:, 2:]
    boxesout[:, 2:] = boxes[:, :2] + 0.5 * boxes[:, 2:]
    boxesout[:, [0,2]] *= w
    boxesout[:, [1,3]] *= h
    boxesout[boxesout[:,0]<0, 0] = 0
    boxesout[boxesout[:,1]<0, 1] = 0
    boxesout[boxesout[:,2]>w-1, 2] = w-1
    boxesout[boxesout[:,3]>h-1, 3] = h-1
    return boxesout


def get_test_result(model_name, image_name, weight, prob_thresh=0.2, nms_thresh=0.4, mode=1, use_gpu=True):
    image = cv2.imread(image_name)
    h, w, _ = image.shape
    pred = get_pred_1(image,model_name,weight,mode,use_gpu)
    # print('get pred')
    total = side*side*num
    probs = np.zeros((total, classes))
    boxes = np.zeros((total, 4))
    since = time.time()
    get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs,True)
    print("get detection boxe:%fs" % (time.time()-since))
    # with open('C/dets.np','wb') as fp:
    #     absp = os.path.abspath(image_name)
    #     buff = np.array([len(absp),side,num,classes])
    #     buff.astype('int32').tofile(fp)
    #     fp.write(bytes(absp.encode()))
    #     buff = np.array([])
    #     buff = np.append(buff,probs)
    #     buff = np.append(buff,boxes)
    #     buff.astype('float32').tofile(fp)
    output = []
    maxind = np.argmax(probs, 1)
    maxprob = np.max(probs, 1)
    mask = maxprob > 0
    if np.sum(mask) == 0:
        return output
    maskbox = boxes[mask]
    maskprob = maxprob[mask]
    maskind = maxind[mask]

    mmord = np.argsort(-maskprob)
    keep = {}
    while mmord.size > 0:
        i = mmord[0]
        keep[i] = []
        if mmord.size == 1:
            break
        ovs = bboxes_iou_np(maskbox[i], maskbox[mmord[1:]])
        mask = ovs > 0.5
        keep[i] = mmord[1:][mask]
        mmord = mmord[1:][1-mask==1]

    for i in keep:
        keepind = []
        keepprob = []
        keepind.append(maskind[i])
        keepprob.append(maskprob[i])
        for j in keep[i]:
            keepind.append(maskind[j])
            keepprob.append(maskprob[j])
        output.append([correct_box(maskbox[i],h,w),keepprob,keepind])
    #output = np.concatenate((maskbox, maskprob, maskind), 1)  # (n,6)
    # for i in range(total):
    #     cls = np.argmax(probs[i])
    #     prob = probs[i][cls]
    #     if prob > 0:
    #         print('{}:{:.3f}'.format(voc_class_names[cls], prob))
    #         box = correct_box(boxes[i],h,w)
    #         other = np.array([prob, cls])
    #         out = np.concatenate((box, other))
    #         output.append(out)
    # for i in range(probs.shape[0]):
    #     cls = np.argmax(probs[i])
    #     prob = probs[i][cls]
    #     #ind = np.nonzero(probs[i]>0)[0]
    #     # if ind.size == 0:
    #     #     continue
    #     if prob > 0:
    #         box = convert_box(boxes[i],h,w,0)
    #         out = np.zeros(6)
    #         out[:4] = box
    #         out[4] = prob
    #         out[5] = cls
    #         output.append(out)
    return output, image


def predict_eval_1(preds, model_name, image_name, weight, mode=1, use_gpu=True):
    root_dir = '/home/blacksun2/github/darknet-2016-11-22/VOCdevkit/AllImages'
    model = load_model(model_name, weight, mode)
    if use_gpu:
        model.cuda()
    if isinstance(image_name, list):
        for imp in tqdm(image_name):
            image = cv2.imread(os.path.join(root_dir, imp))
            h, w, _ = image.shape
            pred = get_pred(image,model,use_gpu)
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
                        preds[voc_class_names[j]].append([imp, probs[i,j], x1, y1, x2, y2])


def predict_eval(test_file, model_name, weight,use_gpu=True):
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
        pred = get_pred(img,model,use_gpu)
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
