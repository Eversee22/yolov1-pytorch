from mmodels import mvgg
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
        model_ft = models.resnet50(pretrained=pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 4096),
                                    nn.ReLU(True),
                                    nn.Dropout(),
                                    nn.Linear(4096, side*side*(num*5+20)),
                                    )
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

