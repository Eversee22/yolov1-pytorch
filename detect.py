from util import imwrite, prep_image
import argparse
import sys
import random
from predict import *

# d = readcfg('cfg/yolond')
# side = int(d['side'])
# num = int(d['num'])
# classes = int(d['classes'])
# inp_size = int(d['inp_size'])


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
    if use_gpu:
        pred = pred.cpu()
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
    print("detecting")
    images = []
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
        pred = get_pred(image,model,use_gpu)
        boxes, probs, cls_indices = get_detection_boxes_1(pred, prob_thresh, nms_thresh, True)
        for i, box in enumerate(boxes):
            if probs[i] == 0:
                continue
            box = convert_box(box,h,w)
            cls_index = cls_indices[i].item()
            # print(cls_index)
            prob = probs[i].item()
            image = imwrite(image, box, voc_class_names[cls_index], cls_index, prob)

        cv2.imshow("%s"%imp, image)
        cv2.imwrite('det/bbox_%s.png' % imp.split('/')[-1].split('.')[0], image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyWindow('img')
            break
        time.sleep(2)


def test(model_name, image_name, weight, prob_thresh=0.2, nms_thresh=0.4, mode=1,use_gpu=True):
    result, image = get_test_result_1(model_name, image_name, weight, prob_thresh, nms_thresh, mode, use_gpu)
    print('get result:%d'%len(result))
    for item in result:
        box = item[:4]
        prob = float(item[4])
        cls_ind = int(item[-1])
        image = imwrite(image,box,voc_class_names[cls_ind],cls_ind,prob)

    cv2.imshow("{}".format(image_name), image)
    cv2.imwrite('bbox_%s.png' % image_name.split('/')[-1].split('.')[0], image)
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
        test(model_name, image_name, weight,thresh,nms, mode,False)
    elif do == "test2":
        test_many(model_name, image_name, weight, thresh, nms, mode)
    elif do == "test3":
        test_canvas(model_name, image_name, weight, thresh, nms, mode)
    elif do == "eval":
        predict_eval(image_name,model_name,weight)
