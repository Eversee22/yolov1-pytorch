from torch.utils.data import DataLoader
import torch
from load_data import VocDataset
import time
from loss import YOLOLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from ftd_model import get_model_ft,load_model_trd
from util import readcfg
from torchvision import transforms
import numpy as np
import torch.nn as nn
from mmodels import mvgg
import os
from visualize import Visualizer

# side = 7
# num = 2
# classes = 20
# sqrt = 1
# noobj_scale = .5
# coord_scale = 5.
# object_scale = 1.
# class_scale = 1.
# batch_size = 16
# inp_size = 448
initial_lr = 0.001
momentum = 0.9
weight_decay = 5e-4
steps = [30, 40]
lr_scale = [0.1, 0.1]
num_epochs = 50

d = readcfg('cfg/yolond')

side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
sqrt = int(d['sqrt'])
noobj_scale = float(d['noobj_scale'])
coord_scale = float(d['coord_scale'])
object_scale = float(d['object_scale'])
class_scale = float(d['class_scale'])
# batch_size = int(d['batch_size'])
batch_size = 16
inp_size = int(d['inp_size'])
# initial_lr = float(d['initial_lr'])
# momentum = float(d['momentum'])
# weight_decay = float(d['weight_decay'])
visualize = True
log = True

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = VocDataset('data/train.txt', side=side, num=num, input_size=inp_size, augmentation=True, transform=data_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# train_dataset_size = len(train_dataset)
train_dataloader_size = len(train_dataloader)

test_dataset = VocDataset('data/voc_2007_test.txt', side=side, num=num, input_size=inp_size, augmentation=False, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_size = len(test_loader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if visualize:
    vis = Visualizer(env='yolo')

if log:
    if not os.path.exists('log'):
        os.mkdir('log')
    logfile = open('log/train_log.txt','w')


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_test_loss = np.inf
    lr = initial_lr
    s = 0

    for epoch in range(num_epochs):
        model.train()
        if scheduler is None:
            # for i, step in enumerate(steps):
            #     if epoch == step:
            #         lr = lr * lr_scale[i]
            #         break
            if s < len(steps) and steps[s] == epoch:
                lr = lr * lr_scale[s]
                s += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            lr = scheduler.get_lr()

        print('Epoch {}/{}, lr:{}'.format(epoch + 1, num_epochs, lr))
        print('-' * 16)

        running_loss = 0.0
        # running_corrects = 0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                if visualize:
                    vis.plot_one(running_loss/(i+1), 'train', 10)
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, train_dataloader_size, loss.item(), running_loss / (i + 1)))

        if s < len(steps) and (epoch+1) == steps[s]:
            print("save {}, step {}, learning rate {}".format(model_name, epoch+1, lr))
            torch.save({'epoch':epoch, 'lr':lr, 'model': model.state_dict()}, "backup/{}_step_{}.pth".format(model_name, epoch+1))
            s += 1
        if log:
            logfile.write('epoch[{:%d}/{:%d}], average loss:{:%f}\n'.format(epoch+1, num_epochs, running_loss/train_dataloader_size))

        # validation, need more gpu memory

        if False:
            validation_loss = 0.0
            model.eval()
            for i, (imgs,target) in enumerate(test_loader):
                imgs = imgs.to(device)
                target = target.to(device)

                out = model(imgs)
                loss = criterion(out,target)
                validation_loss += loss.item()

            validation_loss /= test_loader_size
            if visualize:
                vis.plot_many_stack({'train':running_loss/train_dataloader_size,'val':validation_loss})
            if log:
                logfile.write('epoch[{:%d}/{:%d}], validation loss:{:%f}\n'.format(epoch+1, num_epochs, validation_loss))
            print('validation loss:{}'.format(validation_loss))

            if best_test_loss > validation_loss:
                best_test_loss = validation_loss
                print('epoch%d, get best test loss %.5f' % (epoch, best_test_loss))
                if log:
                    logfile.write('epoch[{:%d}/{:%d}], best test loss:{:%f}\n'.format(epoch + 1, num_epochs, best_test_loss))
                torch.save({'epoch':epoch, 'lr':lr, 'model':model.state_dict()}, 'backup/{}_best.pth'.format(model_name))

    if num_epochs > 20:
        torch.save({'epoch':num_epochs, 'lr':lr, 'model':model.state_dict()}, 'backup/{}_final.pth'.format(model_name))
    time_elapsed = time.time() - since
    print('Training complete in {:0f}h {:0f}m {:.0f}s'.format(time_elapsed//3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    if log:
        logfile.write('Training complete in {:0f}h {:0f}m {:.0f}s\n'.format(time_elapsed//3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        logfile.close()


model_name = "resnet50"
model_ft = get_model_ft(model_name)
#
# model_ft = load_model_trd(model_name, 'backup/vgg16_bn_model_30')
assert model_ft is not None


# if name == "vgg16":
#     model_ft = mvgg.vgg16(pretrained=True)
#     num_ftrs = model_ft.classifier[0].in_features
#     model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, 4096),
#                                         nn.ReLU(True),
#                                         nn.Dropout(),
#                                         nn.Linear(4096, side*side*(num*5+20)),
#                                         )

model_ft.to(device)

criterion = YOLOLoss(side=side, num=num, sqrt=sqrt, coord_scale=coord_scale, noobj_scale=noobj_scale)

# params = model_ft.parameters()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=steps, gamma=0.1)

if not os.path.exists('backup'):
    os.mkdir('backup')
train_model(model_ft, criterion, optimizer_ft, _lr_scheduler, num_epochs=num_epochs)








