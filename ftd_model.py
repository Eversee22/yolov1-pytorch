from mmodels import mvgg,mresnet
from torchvision import models
import torch.nn as nn
import torch
from util import readcfg

d = readcfg('cfg/yolond')
side = int(d['side'])
# num = int(d['num'])
classes = int(d['classes'])


def get_model_ft(name, pretrained=True, num=2):

    if name == "vgg16":
        model_ft = mvgg.vgg16(pretrained=pretrained, side=side, num=num, classes=classes)
        num_ftrs = model_ft.classifier[0].in_features
        model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, side*side*(num*5+20)),
                                            )
    elif name == "vgg16_bn":
        model_ft = mvgg.vgg16_bn(pretrained=pretrained, side=side, num=num, classes=classes)
        num_ftrs = model_ft.classifier[0].in_features
        model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(4096, side * side * (num * 5 + 20)),
                                            )

    elif name == "resnet50":
        model_ft = mresnet.resnet50(pretrained=False,num=num,side=side,num_classes=classes)
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


def load_model_trd(name, weight, mode=0):
    model = get_model_ft(name, pretrained=False)
    assert model is not None

    if mode == 1:
        checkpoint = torch.load(weight)
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(torch.load(weight))

    return model

