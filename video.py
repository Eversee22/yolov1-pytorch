import argparse
import sys
import numpy as np
from predict import get_detection_boxes,get_detection_boxes_1,get_pred,load_model
from util import convert_box


def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLOv1 video')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", type=float, default=0.2)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", type=float, default=0.4)
    parser.add_argument("-m", dest='model', help="model name", default="resnet50", type=str)
    parser.add_argument('weightsfile', nargs=1, help="weights file", type=str)
    # parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
    #                     default="448", type=str)
    parser.add_argument("-v", dest="videofile", help="Video file to run detection on", type=str)

    # if len(sys.argv)<2:
    #     parser.print_help()
    #     exit(0)

    return parser.parse_args()


def predict_canvas(frame, model, prob_thresh=0.2,nms_thresh=0.4, CUDA=True):
    img = prep_image(frame, inp_dim)
    im_dim = frame.shape[1], frame.shape[0]  # w,h
    im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
    if CUDA:
        # im_dim = im_dim.cuda()
        img = img.cuda()
    with torch.no_grad():
        pred = model(img)
    probs = np.zeros((side * side * num, class_num))
    boxes = np.zeros((side * side * num, 4))
    get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs)
    # scaling_factor = np.min(inp_dim/im_dim)
    output = []
    # cls_indices = np.argmax(probs,1)
    probs = torch.from_numpy(probs).float()
    boxes = torch.from_numpy(boxes).float()
    max_prob, max_ind = torch.max(probs, 1)
    mask = max_prob > 0
    count = torch.sum(mask)
    # print(count)
    if count == 0:
        return output
    boxes_out = boxes[mask].contiguous()
    boxes_out = boxes_out * inp_dim
    im_dim = im_dim.repeat(boxes_out.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

    boxes_out[:,[0, 2]] -= (inp_dim - scaling_factor * im_dim[:,0].view(-1,1))/2
    boxes_out[:,[1, 3]] -= (inp_dim - scaling_factor * im_dim[:,1].view(-1,1))/2
    boxes_out /= scaling_factor

    for i in range(boxes_out.size(0)):
        boxes_out[i, [0, 2]] = torch.clamp(boxes_out[i, [0, 2]], 0.0, im_dim[i, 0])
        boxes_out[i, [1, 3]] = torch.clamp(boxes_out[i, [1, 3]], 0.0, im_dim[i, 1])

    prob_out = max_prob[mask].unsqueeze(1)
    ind_out = max_ind[mask].float().unsqueeze(1)
    output = torch.cat((boxes_out,prob_out,ind_out),1)

    # for i in range(probs.shape[0]):
    #     cls = np.argmax(probs[i])
    #     prob = probs[i][cls]
    #     if prob > 0:
    #         out = np.zeros(6)
    #         out[:4] = boxes[i]*inp_dim
    #         out[[0,2]] -= (inp_dim-scaling_factor*im_dim[0])/2
    #         out[[1,3]] -= (inp_dim-scaling_factor*im_dim[1])/2
    #
    #         out[:4] = out[:4]/scaling_factor
    #         out[[0,2]] = np.clip(out[[0,2]],0.0,im_dim[0])
    #         out[[1,3]] = np.clip(out[[1,3]],0.0,im_dim[1])
    #
    #         out[4] = prob
    #         out[5] = cls
    #         output.append(out)

    return output


def predictv1(frame, model, prob_thresh=0.2,nms_thresh=0.4, CUDA=True):
    h,w,_ = frame.shape
    pred = get_pred(frame,model,CUDA)
    probs = np.zeros((side * side * num, class_num))
    boxes = np.zeros((side * side * num, 4))
    get_detection_boxes(pred, prob_thresh, nms_thresh, boxes, probs)
    output = []

    for i in range(probs.shape[0]):
        cls = np.argmax(probs[i])
        prob = probs[i][cls]
        if prob > 0:
            box = boxes[i]
            out = np.zeros(6)
            out[:4] = convert_box(box,h,w)
            out[4] = prob
            out[5] = cls
            output.append(out)

    return output


def predictv2(frame, model, prob_thresh=0.2,nms_thresh=0.4, CUDA=True):
    h,w,_ = frame.shape
    pred = get_pred(frame,model,CUDA)
    boxes, probs, cls_inds = get_detection_boxes_1(pred,prob_thresh,nms_thresh)
    output = []
    write = 0
    for i, box in enumerate(boxes):
        if probs[i] == 0:
            continue
        box = torch.FloatTensor(convert_box(box, h, w))
        prob = probs[i].unsqueeze(0)
        cls_ind = cls_inds[i].unsqueeze(0).float()
        if write == 0:
            output = torch.cat((box, prob, cls_ind)).unsqueeze(0)
            write = 1
        else:
            out = torch.cat((box, prob, cls_ind)).unsqueeze(0)
            output = torch.cat((output, out))

    return output


if __name__ == '__main__':
    import torch
    from util import load_classes,imwrite,readcfg,prep_image
    import cv2
    import time

    args = arg_parse()
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    weightsfile = args.weightsfile[0]

    class_names = load_classes('data/voc.names')
    class_num = len(class_names)
    CUDA = torch.cuda.is_available()
    cfg = readcfg('cfg/yolond')
    inp_dim = int(cfg['inp_size'])
    side = int(cfg['side'])
    num = int(cfg['num'])

    print('Loading network')
    model = load_model(args.model,weightsfile,1)
    print('network successfully loaded')

    if CUDA:
        model.cuda()
    videofile = args.videofile
    if videofile is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output = predictv1(frame, model, CUDA=CUDA)
            for item in output:
                # item = output[i]
                cls = int(item[-1])
                prob = float(item[-2])
                box = item[:4]
                frame = imwrite(frame, box, class_names[cls], cls)

            frames += 1
            fps = frames / (time.time()-start)
            label='fps:{:.3f}'.format(fps)
            # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.putText(frame, label, (1, 10), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1);
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            print(time.time()-start)

            # print("FPS of the video is {:5.2f}".format(fps))
        else:
            break









