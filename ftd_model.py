from mmodels import mvgg,mresnet
from torchvision import models
import torch.nn as nn
import torch
from util import readcfg
import numpy as np
import sys

d = readcfg('cfg/yolond')
side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
softmax = int(d['softmax'])


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 use_relu=True,use_bn=True,reshape=False):
        super(Conv2d_BatchNorm, self).__init__()
        self.reshape = reshape
        self.use_relu = use_relu
        self.use_bn = use_bn
        # padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.reshape:
            x = x.view(x.size(0),-1)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0,
                 use_relu=True,reshape=False):
        super(Conv2d, self).__init__()
        self.reshape = reshape
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.reshape:
            x = x.view(x.size(0), -1)
        return x


class Maxpool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, reshape=False):
        super(Maxpool2d, self).__init__()
        self.reshape = reshape
        self.maxpool = nn.MaxPool2d(kernel_size,stride,padding)

    def forward(self, x):
        x = self.maxpool(x)
        if self.reshape:
            x = x.view(x.size(0),-1)
        return x


def AddExtraLayersVGG(in_channels=512,
                      relu=True, bn=True, need_fc=True, full_conv=True, reduced=True, dilated=False, dropout=False):
    layers = []
    nopool = False
    dilation = 1

    if full_conv:
        if need_fc:
            # pool5
            # if dilated:
            #     if nopool:
            #         layers.append(nn.Conv2d(in_channels, 512, 3, 1, padding=1))
            #     else:
            #         layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            # else:
            #     if nopool:
            #         layers.append(nn.Conv2d(in_channels, 512, 3, 2, padding=1))
            #     else:
            #         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            #
            # in_channels = 512

            if dilated:
                if reduced:
                    dilation = dilation * 6
                    kernel_size = 3
                    num_output = 1024
                else:
                    dilation = dilation * 2
                    kernel_size = 7
                    num_output = 4096
            else:
                if reduced:
                    dilation = dilation * 3
                    kernel_size = 3
                    num_output = 1024
                else:
                    kernel_size = 7
                    num_output = 4096
            # fc6
            pad = dilation * (kernel_size - 1) // 2
            layers.append(nn.Conv2d(in_channels, num_output, kernel_size, padding=pad, dilation=dilation))
            layers.append(nn.ReLU(inplace=True))

            in_channels = num_output

            if dropout:
                layers.append(nn.Dropout())

            # fc7
            if reduced:
                layers.append(nn.Conv2d(in_channels, 1024, 1))
                in_channels = 1024
            else:
                layers.append(nn.Conv2d(in_channels, 4096, 1))
                in_channels = 4096
            layers.append(nn.ReLU(inplace=True))

            if dropout:
                layers.append(nn.Dropout())

        # extra layers
        # layers += [nn.Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        # layers += [nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]

        # output
        # in_channels = 512
        # layers += [nn.Conv2d(in_channels, 256, 3,1,1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        # layers.append(nn.Conv2d(256, num * 5 + classes,1))
        layers.append(nn.Conv2d(in_channels, num * 5 + classes, 3, 1, 1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(num * 5 + classes))
    else:
        # layers += [Conv2d_BatchNorm(in_channels, 512, 3, 1, 1, use_bn=bn)]
        # layers += [Conv2d_BatchNorm(512, 512, 3, 1, 1, use_bn=bn)]
        # layers += [Conv2d_BatchNorm(512, 512, 3, 2, 1, use_bn=bn,use_relu=False,reshape=True)]
        # layers += [Maxpool2d(reshape=True)]
        # layers += [nn.Conv2d(in_channels,num*5+classes,3,1,1,bias=False),nn.BatchNorm2d(num*5+classes)]
        layers += [nn.Linear(512*7*7, 4096),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(4096, side * side * (num * 5 + classes))]

    return nn.Sequential(*layers)


def get_model_ft(name, pretrained=True):
    # models.vgg16()
    if name == "vgg16":
        model_ft = mvgg.vgg16(pretrained=pretrained, side=side, num=num, classes=classes)
        # num_ftrs = model_ft.classifier[0].in_features
        model_ft.classifier = AddExtraLayersVGG(dilated=False, full_conv=True, need_fc=True, bn=True)

    elif name == "vgg16_bn":
        model_ft = mvgg.vgg16_bn(pretrained=pretrained, side=side, num=num, classes=classes)
        # num_ftrs = model_ft.classifier[0].in_features
        model_ft.classifier = AddExtraLayersVGG(dilated=False)

    elif name == "resnet50":
        models.resnet50()
        downsm = True
        model_ft = mresnet.resnet50(num=num, side=side, num_classes=classes,
                                    softmax=softmax, detnet_block=not downsm, downsample=downsm)
        if pretrained:
            resnet = models.resnet50(pretrained=True)
            org_dict = resnet.state_dict()
            ft_dict = model_ft.state_dict()
            for k in org_dict.keys():
                if k in ft_dict.keys() and not k.startswith('fc'):
                    ft_dict[k] = org_dict[k]
            model_ft.load_state_dict(ft_dict)

    else:
        return None

    return model_ft


def convert_weight(name):
    if name not in ['vgg16','vgg16_bn']:
        print('Do not support '+name)
        sys.exit(1)
    base = '../darknet-2016-11-22/weights/'
    bn = False
    if name=='vgg16':
        model=models.vgg16(pretrained=True)
        model_dict = model.state_dict()
        wf = base + 'vgg16_test.bin'
    elif name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        model_dict = model.state_dict()
        wf = base + 'vgg16_bn_test.bin'
        bn = True

    if name.startswith('vgg16'):
        weights = {}
        for k in model_dict.keys():
            print(k)
            if k.startswith('features'):
                splitk = k.split('.')
                id = int(splitk[1])
                typ = splitk[2]
                w = model_dict[k]
                if id not in weights.keys():
                    weights[id] = {}
                print(id, typ, w.shape)
                if typ == 'bias' and w.size(0) == 64:
                    print(w)
                weights[id][typ] = w

        fp = open(wf, 'wb')
        head = np.random.randint(1,size=(4,))
        head.astype('int32').tofile(fp)
        seq = ['bias', 'weight'] if not bn else ['weight']
        seq1 = ['bias','weight','running_mean','running_var']
        count = 4*4
        # print(weights.keys())
        kl = list(weights.keys())
        kl.sort()
        # print(kl)
        step = 2 if bn else 1
        for i in range(len(kl)//step):
            # print(weights[i].keys())
            if bn:
                k = kl[i * step + 1]
                # print(k)
                for s in seq1:
                    w = weights[k][s].numpy()
                    c = 1
                    # print(w.shape)
                    for j in w.shape:
                        c *= j
                    count += c * 4
                    w.astype('float32').tofile(fp)
            k = kl[i * step]
            # print(k)
            for s in seq:
                w = weights[k][s].numpy()
                c = 1
                # print(w.shape)
                for j in w.shape:
                    c *= j
                count += c * 4
                w.astype('float32').tofile(fp)

        fp.close()
        print('write over,{} bytes'.format(count))


def load_model_trd(name, weight, mode=1, use_gpu=True):
    model = get_model_ft(name, pretrained=False)
    assert model is not None
    start_epoch = -1
    ckpt = torch.load(weight, map_location=torch.device('cuda:0' if use_gpu else 'cpu'))
    if mode == 1:
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch']
        lr = ckpt['lr']
    else:
        model.load_state_dict(torch.load(weight))

    return model,start_epoch,lr


if __name__ == '__main__':
    # model = mvgg.vgg16(pretrained=True)
    # l_dict = model.state_dict()
    # for k in l_dict.keys():
    #     print(k, l_dict[k].shape)
    # torch.random.manual_seed(6)
    # input = torch.rand(1,3,224,224)
    # output = model(input)
    # print(output.shape)
    convert_weight('vgg16_bn')
