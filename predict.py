import torch
# import torch.nn.functional as F
from util import readcfg, load_classes
from ftd_model import get_model_ft
import numpy as np
import cv2
import time
import os,sys
from tqdm import tqdm
from util import convert_box
import os

d = readcfg('cfg/yolond')
side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
inp_size = int(d['inp_size'])
softmax = int(d['softmax'])
sqrt = int(d['sqrt'])
voc_class_names = load_classes('data/voc.names')
gpudevice = torch.device('cuda:0')


def load_model(model_name, weight, mode, use_gpu):
    model = get_model_ft(model_name, False)
    assert model is not None
    if use_gpu:
        checkpoint = torch.load(weight, map_location=gpudevice)
    else:
        checkpoint = torch.load(weight, map_location=torch.device('cpu'))
    if mode == 1:
        model_dict = checkpoint['model']
        # for k in model_dict.keys():
        #     print(k, model_dict[k].shape)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)
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
    if softmax:
        pred[:,:,num*5:] = torch.softmax(pred[:,:,num*5:],2)
    contain = []
    for i in range(num):
        contain.append(pred[:, :, i*5+4].unsqueeze(2))
    contain = torch.cat(contain, 2)
    # print(torch.sum(contain > 0))
    mask1 = contain > prob_thres
    # mask2 = contain == contain.max()
    mask = mask1
    # print(torch.sum(mask))

    for i in range(side):
        for j in range(side):
            for k in range(num):
                if mask[i, j, k] == 1:
                    objc = torch.FloatTensor([pred[i,j,k*5+4]])
                    max_prob, cls_ind = torch.max(pred[i, j, num*5:].view(-1, 1), 0)
                    c_prob = objc*max_prob
                    if c_prob.item() > prob_thres:
                        xy = torch.FloatTensor([j, i])
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
        # since = time.time()
        keep = do_nms_1(boxes, probs, nms_thresh)
        # print("do nms 1:%fs" % (time.time()-since))
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


def bbox_iou_np(box_a, box_b):
    '''
    box1,box2: [xmin,ymin,xmax,ymax], value:0.0~1.0
    '''
    ixmin = np.maximum(box_a[0] - 0.5 * box_b[2], box_a[0] - 0.5 * box_a[2])
    iymin = np.maximum(box_b[1] - 0.5 * box_b[3], box_a[1] - 0.5 * box_a[3])
    ixmax = np.minimum(box_b[0] + 0.5 * box_b[2], box_a[0] + 0.5 * box_a[2])
    iymax = np.minimum(box_b[1] + 0.5 * box_b[3], box_a[1] + 0.5 * box_a[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inter = iw * ih
    uni = ((box_a[2]) * (box_a[3]) +
           (box_b[2]) * (box_b[3]) - inter)
    iou = inter / uni

    return iou


def do_nms(boxes, probs, thresh=0.4):
    """
    boxes: [side*side*num,4]
    probs: [side*side*num,20]

    """
    total = side*side*num
    for k in range(classes):
        order = np.argsort(probs[:, k])[::-1] # descending order
        for i in range(total):
            if probs[order[i]][k] == 0:
                continue
            box_a = boxes[order[i]]
            box_b = boxes[order[i+1:]]
            overlaps = bboxes_iou_np(box_a,box_b)
            probs[order[i+1:][overlaps>thresh], k] = 0
            # for j in range(i+1, total):
            #     box_b = boxes[order[j]]
            #     iou = bbox_iou_np(box_a, box_b)
            #     if iou > thresh:
            #         probs[order[j]][k] = 0


def get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs, nms=True):
    """
    pred: (1, side, side, num*5+20)
    return: probs, boxes
    """
    pred = pred.data.squeeze(0)
    if softmax:
        pred[:,:,num*5:] = torch.softmax(pred[:,:,num*5:],2)
    pred = pred.numpy()
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
            boxes[g*num + k][:2] = xy
            # if sqrt:
            #     boxes[g * num + k][2:] = np.square(box[2:])
            # else:
            boxes[g*num + k][2:] = box[2:]

            prob = obj*pred[i, j, num*5:]
            mask = prob > prob_thresh
            probs[g*num+k, mask] = prob[mask]
            # probs[g*num+k, prob != prob.max()] = 0
            # for z in range(classes):
            #     prob = obj * pred[i, j, num * 5 + z]
            #     probs[g*num+k, z] = prob if prob > prob_thresh else 0
    if nms:
        # since = time.time()
        do_nms(boxes, probs, nms_thresh)
        # print("do nms:%f"%(time.time()-since))
    # return boxes, probs


def get_detection_boxes2(pred, prob_thresh, nms_thresh, boxes, probs, nms=True):
    """
    pred: (1, side, side, num*(5+20))
    return: probs, boxes
    """
    pred = pred.data.squeeze(0)
    pred = pred.view(side, side, num, 5+classes)
    if softmax:
        pred[:, :, :, 5:] = torch.softmax(pred[:, :, :, 5:], 2)
    pred = pred.numpy()
    # probs = np.zeros((side*side*num, classes))
    # boxes = np.zeros((side*side*num, 4))

    for i in range(side):
        for j in range(side):
            for k in range(num):
                obj = pred[i, j, k, 4]
                if obj <= prob_thresh:
                    continue
                box = pred[i, j, k, :4]
                cr = np.array([j, i], dtype=np.float32)
                xy = (box[:2] + cr) / side
                ind = (i*side+j)*num+k
                boxes[ind, :2] = xy
                # if sqrt:
                #     boxes[ind, 2:] = np.square(box[2:])
                # else:
                boxes[ind, 2:] = box[2:]
                prob = obj * pred[i, j, k, 5:]
                mask = prob > prob_thresh
                probs[ind, mask] = prob[mask]
    if nms:
        #since = time.time()
        do_nms(boxes, probs, nms_thresh)
        #print("do nms:%f"%(time.time()-since))
    # return boxes, probs


def get_pred_1(image,model_name,weight,mode=1,use_gpu=True):
    model = load_model(model_name, weight, mode,use_gpu)
    if use_gpu:
        model.to(gpudevice)
    # h, w, _ = image.shape
    img = img_trans(image)
    if use_gpu:
        img = img.to(gpudevice)
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
        img = img.to(gpudevice)
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
    return output

    # return boxes, probs, clsinds


def correct_box(box,h,w,mode=0):
    boxout = box.copy()
    boxout[:2] = box[:2] - 0.5 * box[2:]
    boxout[2:] = box[:2] + 0.5 * box[2:]
    boxout[[0, 2]] *= w
    boxout[[1, 3]] *= h
    if boxout[0] < 0: boxout[0] = 0
    if boxout[1] < 0: boxout[1] = 0
    if mode==0:
        if boxout[2] > w - 1: boxout[2] = w - 1
        if boxout[3] > h - 1: boxout[3] = h - 1
    elif mode==1:
        if boxout[2] > w: boxout[2] = w
        if boxout[3] > h: boxout[3] = h
    return boxout


def correct_boxes(boxes,h,w,mode=0):
    boxesout = boxes.copy()
    boxesout[:, :2] = boxes[:, :2] - 0.5 * boxes[:, 2:]
    boxesout[:, 2:] = boxes[:, :2] + 0.5 * boxes[:, 2:]
    boxesout[:, [0,2]] *= w
    boxesout[:, [1,3]] *= h
    boxesout[boxesout[:,0]<0, 0] = 0
    boxesout[boxesout[:,1]<0, 1] = 0
    if mode==0:
        boxesout[boxesout[:,2]>w-1, 2] = w-1
        boxesout[boxesout[:,3]>h-1, 3] = h-1
    elif mode==1:
        boxesout[boxesout[:, 2] > w, 2] = w
        boxesout[boxesout[:, 3] > h, 3] = h
    return boxesout


def postdeal(box,prob,ind,h,w):
    mmord = np.argsort(prob)[::-1]
    keep = {}
    while mmord.size > 0:
        i = mmord[0]
        keep[i] = []
        if mmord.size == 1:
            break
        ovs = bboxes_iou_np(box[i], box[mmord[1:]])
        mask = ovs > 0.6
        keep[i] = mmord[1:][mask]
        mmord = mmord[1:][1 - mask == 1]
    output = []
    for i in keep:
        keepind = []
        keepprob = []
        keepind.append(ind[i])
        keepprob.append(prob[i])
        ki = ind[keep[i]]
        kp = prob[keep[i]]
        asind = np.argsort(ki)
        uni = ind[i]
        # print(ki)
        for j in asind:
            if ki[j] != uni:
                keepind.append(ki[j])
                keepprob.append(kp[j])
                uni = ki[j]
        output.append([correct_box(box[i], h, w), keepprob, keepind])
    return output


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
    pd = True
    output = []
    maxind = np.argmax(probs, 1)
    maxprob = np.max(probs, 1)
    mask = maxprob > 0
    if np.sum(mask) == 0:
        return output

    maskbox = boxes[mask]
    maskprob = maxprob[mask]
    maskind = maxind[mask]

    if pd:
        output = postdeal(maskbox,maskprob,maskind,h,w)
    else:
        maskbox = correct_boxes(maskbox,h,w)
        maskprob = maxprob[mask].reshape(-1, 1)
        maskind = maxind[mask].reshape(-1, 1)
        output = np.concatenate((maskbox, maskprob, maskind), 1)  # (n,6)
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
    return output


def predict_eval_1(preds, model_name, image_name, weight, mode=1, use_gpu=True):
    root_dir = '/home/blacksun2/github/darknet-2016-11-22/VOCdevkit/VOC2007/JPEGImages'
    model = load_model(model_name, weight, mode, use_gpu)
    if use_gpu:
        model.to(gpudevice)

    # if not os.path.exists('results'):
    #     os.mkdir('results')
    # base = 'results/comp4_det_test'
    # fps = []
    # for k in range(classes):
    #     fps.append(open('%s_%s' % (base, voc_class_names[k]), 'w'))

    if isinstance(image_name, list):
        for imid in tqdm(image_name):
            image = cv2.imread(os.path.join(root_dir, imid))
            h, w, _ = image.shape
            pred = get_pred(image,model,use_gpu)
            # pred = pred.cpu()
            boxes, probs, cls_inds = get_detection_boxes_1(pred, 0.1, 0.5)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = convert_box(box,h,w,3)
                cls_ind = int(cls_inds[i])
                prob = float(probs[i])
                preds[voc_class_names[cls_ind]].append([imid, prob, x1, y1, x2, y2])

            # for i, box in enumerate(boxes):
            #     prob = float(probs[i])
            #     # print(prob)
            #     if prob == 0:
            #         continue
            #     x1, y1, x2, y2 = convert_box(box,h,w,3)
            #     cls_ind = int(cls_inds[i])
            #     fps[cls_ind].write('%s %f %f %f %f %f\n' % (imid.split('.')[0], prob, x1, y1, x2, y2))
    # for f in fps:
    #     f.close()


def predict_eval(test_file, model_name, weight,use_gpu=True):
    with open(test_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    model = load_model(model_name, weight, 1,use_gpu)
    if use_gpu:
        model.to(gpudevice)
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
        get_detection_boxes(pred, 0.1, 0.5, boxes, probs,True)
        for i in range(probs.shape[0]):
            x1, y1, x2, y2 = correct_box(boxes[i],h,w,1)
            for j in range(classes):
                if probs[i, j] > 0:
                    fps[j].write('%s %f %f %f %f %f\n' % (imgid, probs[i][j], x1, y1, x2, y2))
            # fps[j].flush()

    for f in fps:
        f.close()
