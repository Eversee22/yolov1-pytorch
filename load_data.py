from __future__ import print_function, division
import os
import torch
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random


class VocDataset(Dataset):
    def __init__(self, data_file, side=7, num=2, input_size=448, class_num=20, augmentation=False, transform=None):

        with open(data_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 0 and line[0] != '#']

        self.images = []
        self.bbox = []
        self.classes = []
        for line in lines:
            splited_line = line.split()
            self.images.append(splited_line[0])
            num_boxes = (len(splited_line) - 1) // 5
            _bbox = []
            _classes = []
            for i in range(num_boxes):
                x1 = float(splited_line[1 + 5 * i])
                y1 = float(splited_line[2 + 5 * i])
                x2 = float(splited_line[3 + 5 * i])
                y2 = float(splited_line[4 + 5 * i])
                c = int(splited_line[5 + 5 * i])
                _bbox.append([x1, y1, x2, y2, ])
                _classes.append(c)

            self.bbox.append(torch.FloatTensor(_bbox))
            self.classes.append(torch.ByteTensor(_classes))

        self.augmentation = augmentation
        self.transform = transform
        self.side = side
        self.num_box = num
        self.image_size = input_size
        self.class_num = class_num
        self.num_samples = len(self.images)
        # self.mean = (123, 117, 104)  # RGB

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # try:
        img = cv2.imread(self.images[idx])
        # except Exception as e:
        #     raise Exception(e)
        boxes = self.bbox[idx].clone()
        labels = self.classes[idx].clone()

        if self.augmentation:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift2(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.FloatTensor([w, h, w, h]).expand_as(boxes)
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        # img = self.subMean(img, self.mean)  #
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().div(255.0)
        gt = self.encoder(boxes, labels)

        if self.transform is not None:
            img = self.transform(img)

        return img, gt

    def get_item(self, idx):
        """
        idx - index
        return - image(cv2 reading),bbox(boxes in this image),classes(corresponding class of each box)
        """
        img = cv2.imread(self.images[idx])
        boxes = self.bbox[idx].clone()
        labels = self.classes[idx].clone()

        if self.augmentation:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            # img = self.randomBlur(img)
            # img = self.RandomBrightness(img)
            # img = self.RandomHue(img)
            # img = self.RandomSaturation(img)
            # img, boxes, labels = self.randomShift2(img, boxes, labels)
            # img, boxes, labels = self.randomCrop(img, boxes, labels)

        # if self.transform is not None:
        #     img = self.transform(img)

        return {"image":img, "bbox":boxes, "classes": labels}

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return side*side*25 (side, side, (x,y,w,h,objectness,classes))
        '''
        target = torch.zeros((self.side, self.side, self.num_box*5 + 20))
        # cell_size = 1./self.side
        wh = boxes[:, 2:] - boxes[:, :2]
        c_coords = (boxes[:, 2:] + boxes[:, :2]) / 2
        for k in range(c_coords.size()[0]):
            c_coord = c_coords[k]
            ij = (c_coord * self.side).ceil() - 1  # grid left-up int coord
            xy = ij / self.side  # grid left-up float coord
            offset = (c_coord - xy) * self.side
            row, col = int(ij[1]), int(ij[0])
            for i in range(self.num_box):
                target[row, col, i*5:i*5+2] = offset
                target[row, col, i*5+2:i*5+4] = wh[k]
                target[row, col, i*5+4] = 1
            target[row, col, int(labels[k]) + self.num_box*5] = 1

        return target

    def encoder2(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return side*side*50 (side, side, num*(x,y,w,h,objectness,classes))
        '''
        target = torch.zeros((self.side, self.side, self.num_box*(5 + self.class_num)))
        # cell_size = 1./self.side
        wh = boxes[:, 2:] - boxes[:, :2]
        c_coords = (boxes[:, 2:] + boxes[:, :2]) / 2
        for k in range(c_coords.size()[0]):
            c_coord = c_coords[k]
            ij = (c_coord * self.side).ceil() - 1  # grid left-up int coord
            xy = ij / self.side  # grid left-up float coord
            offset = (c_coord - xy) * self.side
            row, col = int(ij[1]), int(ij[0])
            cls = int(labels[k])
            for i in range(self.num_box):
                target[row, col, i*5:i*5+2] = offset
                target[row, col, i*5+2:i*5+4] = wh[k]
                target[row, col, i*5+4] = 1
                target[row, col, i*5+5+cls] = 1

        return target

    def PCA(self, img):
        h, w, c = img.shape
        renorm_img = np.reshape(img, (h * w, c))
        renorm_img = renorm_img.astype(np.float32)
        mean = np.mean(renorm_img, axis=0)
        std = np.std(renorm_img, axis=0)
        renorm_img -= mean
        renorm_img /= std
        # covariance matrix for RGB variables
        cov = np.cov(renorm_img, rowvar=False)
        # eigenvalue and eigenvector
        lambdas, p = np.linalg.eig(cov)
        alphas = np.random.normal(0, 0.1, 3)
        delta = np.dot(p, alphas * lambdas)
        pca_aug_img = renorm_img + delta
        pca_color_img = pca_aug_img * std + mean
        pca_color_img = np.clip(pca_color_img, 0, 255).astype(np.uint8)
        pca_color_img = np.reshape(pca_color_img, (h, w, c))

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift2(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            shift_x, shift_y = int(shift_x), int(shift_y)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img_shift = cv2.warpAffine(bgr, M, (width, height))
            shift_xy = torch.FloatTensor([[shift_x, shift_y]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[shift_x, shift_y, shift_x, shift_y]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]

            return img_shift, boxes_in, labels_in

        return bgr, boxes, labels

    def randomShift(self, bgr, boxes, labels):
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (0, 0, 0)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            shift_x, shift_y = int(shift_x), int(shift_y)
            if shift_x >= 0:
                if shift_y >= 0:  # rb
                    after_shfit_image[shift_y:, shift_x:, :] = \
                        bgr[:height-shift_y, :width-shift_x, :]
                else:  # ru
                    after_shfit_image[:height+shift_y, shift_x:, :] = \
                        bgr[-shift_y:, :width-shift_x, :]
            else:
                if shift_y >= 0:  # lb
                    after_shfit_image[shift_y:, :width+shift_x, :] = \
                        bgr[:height-shift_y, -shift_x:, :]
                else:  # lu
                    after_shfit_image[:height+shift_y, :width+shift_x, :] = \
                        bgr[-shift_y:, -shift_x:, :]

            shift_xy = torch.FloatTensor([[shift_x, shift_y]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[shift_x, shift_y, shift_x, shift_y]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]

            return after_shfit_image, boxes_in, labels_in

        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            # print(scale)
            # if scale < 1.0:
            #     interpl = cv2.INTER_AREA
            # else:
            #     interpl = cv2.INTER_LINEAR
            interpl = cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (int(width * scale), height), interpolation=interpl)
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w]
            return img_croped, boxes_in, labels_in

        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


# colors = pkl.load(open("pallete", "rb"))

if __name__ == '__main__':
    from util import imwrite, convert_box, load_classes
    # import _pickle as pkl
    import time

    voc_class_names = load_classes('data/voc.names')
    # colors = pkl.load(open("pallete", "rb"))
    voc_dataset = VocDataset('data/train07+12.txt', augmentation=True)
    dlen = len(voc_dataset)

    # plt.ion()
    # fig = plt.figure()
    while 1:
        idx = np.random.randint(dlen)
        # idx = 1
        # try:
        sample = voc_dataset.get_item(idx)
        # except Exception as e:
        #     print(e, idx)
            # continue
        image = sample["image"]
        boxes = sample["bbox"]
        classes = sample["classes"]
        h,w,_ = image.shape
        dir = 'gt'
        if not os.path.exists(dir):
            os.mkdir(dir)

        for j in range(len(boxes)):
            cls_ind = classes[j].item()
            # print(cls_ind)
            image = imwrite(image, convert_box(boxes[j], h, w, 1), voc_class_names[cls_ind], cls_ind)

        #plt.figure()

        # plt.title("pic{}".format(i+1))
        # plt.imshow(image)
        # cv2.imwrite("data/temp/pic{}.jpg".format(i+1),image)
        # plt.pause(1)
        cv2.imshow('pic', image)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            cv2.destroyWindow('pic')
            break
        elif key & 0xFF == ord('s'):
            cv2.imwrite('{}/pic_gt{}.png'.format(dir,idx),image)
        time.sleep(1)



    # plt.ioff()
    # plt.show()
    # plt.close()











