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
# from mmodels import mvgg
import os
from visualize import Visualizer
#from adabound import adabound
import argparse
import sys
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
batch_size = 16  # if gpu memory is enough, 16 ~ 64 is ok
inp_size = int(d['inp_size'])
# initial_lr = float(d['initial_lr'])
# momentum = float(d['momentum'])
# weight_decay = float(d['weight_decay'])
visualize = True
log = True
validate = False
vischange = False
save_final = False

data_transforms = transforms.Compose([
    # transforms.ToTensor(),
])

train_dataset = VocDataset('data/train07.txt', side=side, num=num, input_size=inp_size, augmentation=True, transform=data_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# train_dataset_size = len(train_dataset)
train_loader_size = len(train_dataloader)

test_dataset = VocDataset('data/voc_2007_test.txt', side=side, num=num, input_size=inp_size, augmentation=False, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_size = len(test_loader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs, dyn=False):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_test_loss = np.inf
    lr = initial_lr
    s = 0
    prevloss = -1
    avg_loss = -1

    if dyn:
        print('using dynamic learning rate')
    for epoch in range(start_epoch+1, num_epochs):
        model.train()
        # if scheduler is None:
        #     # for i, step in enumerate(steps):
        #     #     if epoch == step:
        #     #         lr = lr * lr_scale[i]
        #     #         break
        #     if s < len(steps) and steps[s] == epoch:
        #         lr = lr * lr_scale[s]
        #         s += 1
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # else:
        #     scheduler.step()
        #     lr = scheduler.get_lr()
        if scheduler is not None and not dyn:
            scheduler.step()
            lr = scheduler.get_lr()

        print('Epoch {}/{}, lr:{}'.format(epoch + 1, num_epochs, lr))
        print('-' * 16)

        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if avg_loss < 0:
                avg_loss = loss.item()
            avg_loss = avg_loss*0.98+loss.item()*0.02

            if prevloss < 0:
                prevloss = running_loss
                if visualize:
                    vis.plot_one(running_loss, 'train', 4, 'iter')
                    if vischange:
                        vis.plot_one(0, 'change', 4, 'iter', 'rate')
                        diff = 4

            if (i+1) % 5 == 0 or i+1 == train_loader_size:
                # avg_loss = running_loss / (i + 1)
                if visualize:
                    step = 5
                    if train_loader_size%5 and (i+6)>train_loader_size and (i+1)<train_loader_size:
                        step = train_loader_size % 5
                    vis.plot_one(avg_loss, 'train', step, 'iter')
                    if vischange:
                        change = prevloss-avg_loss
                        vis.plot_one(change / diff, 'change', step, 'iter', 'rate')
                        prevloss = avg_loss
                        diff = step
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' %
                      (epoch+1, num_epochs, i+1, train_loader_size, loss.item(), avg_loss))
                if log:
                    logfile.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f\n' %
                                  (epoch+1, num_epochs, i+1, train_loader_size, loss.item(), avg_loss))

        # print('\nEpoch[{}], average loss: {:.4f}\n'.format(epoch+1, running_loss/train_loader_size))
        # if log:
        #     logfile.write('Epoch[{}], average loss: {:.4f}\n'.format(epoch+1, running_loss/train_loader_size))

        if s < len(steps) and (epoch+1) == steps[s]:
            print("save {}, step {}, learning rate {}".format(model_name, epoch+1, lr))
            torch.save({'epoch': epoch, 'lr': lr, 'model': model.state_dict()}, backupdir+"{}_step_{}.pth".format(model_name, epoch+1))
            s += 1

        # validation
        if validate:
            validation_loss = 0.0
            model.eval()
            for i, (imgs, target) in enumerate(test_loader):
                imgs = imgs.to(device)
                target = target.to(device)

                out = model(imgs)
                loss = criterion(out, target)
                validation_loss += loss.item()

            validation_loss /= test_loader_size
            if scheduler is not None and dyn:
                scheduler.step(validation_loss)

            if visualize:
                vis.plot_many_stack({'train': running_loss / train_loader_size, 'val': validation_loss})
            if log:
                logfile.write('epoch[{}/{}], validation loss:{}\n'.format(epoch + 1, num_epochs, validation_loss))
            print('validation loss:{}'.format(validation_loss))

            if best_test_loss > validation_loss:
                best_test_loss = validation_loss
                print('epoch%d, get best test loss %.5f' % (epoch+1, best_test_loss))
                if log:
                    logfile.write('epoch[{}/{}], best test loss:{}\n'.format(epoch + 1, num_epochs, best_test_loss))
                torch.save({'epoch': epoch, 'best_loss':best_test_loss, 'lr': lr, 'model': model.state_dict()}, backupdir+'{}_best.pth'.format(model_name))

        if log:
            logfile.flush()

    # end
    if num_epochs > 20 or save_final:
        torch.save({'epoch':num_epochs-1, 'lr':lr, 'model':model.state_dict()}, backupdir+'{}_final.pth'.format(model_name))
    time_elapsed = int(time.time() - since)
    h = time_elapsed // 3600
    m = (time_elapsed % 3600) // 60
    s = time_elapsed % 60
    print('{} epochs, spend {}h:{}m:{:.0f}s'.format(num_epochs, h, m, s))
    if log:
        logfile.write('{} epochs, spend {}h:{}m:{:.0f}s\n'.format(num_epochs, h, m, s))
        logfile.close()


def arg_parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-w", dest="weight", help="load weight", type=str)
    arg_parser.add_argument("-n", dest="net", default="resnet50", help="backbone net", type=str)
    arg_parser.add_argument("-env",dest="env", help="visdom environment",type=str)

    return arg_parser.parse_args()


args = arg_parse()
model_name = args.net
env = args.env
if env is None:
    print('no visdom-environment specified, visualization off')
    visualize = False
    # sys.exit(1)
vis = None
if visualize:
    vis = Visualizer(env='{}{}_{}'.format(model_name, env, time.strftime('%m%d%H%M')))
if log:
    if not os.path.exists('log'):
        os.mkdir('log')
    logfile = open('log/{}_{}.train.log'.format(model_name, time.strftime('%m%d%H%M')),'w')

start_epoch = -1
if args.weight is not None:
    model_ft, start_epoch, lr = load_model_trd(model_name, args.weight)
    print('weight loaded', 'epoch:', start_epoch+1, 'lr:', lr)
else:
    print('no weight specified, training from 0')
    model_ft = get_model_ft(model_name)
    # sys.exit(1)
assert model_ft is not None
# print(model_ft)

model_ft.to(device)

# vis = Visualizer('cood2loss_2_cmp_{}'.format(time.strftime('%m%d%H%M')))
criterion = YOLOLoss(side=side, num=num, sqrt=sqrt, coord_scale=coord_scale, noobj_scale=noobj_scale, vis=vis,device=device)

# params=[]
# params_dict = dict(model_ft.named_parameters())
# for key,value in params_dict.items():
#     print(key,value.shape)
#     if key.startswith('features'):
#         params += [{'params':[value],'lr':initial_lr*1}]
#     else:
#         params += [{'params':[value],'lr':initial_lr}]
optimizer_ft = optim.SGD(model_ft.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
# optimizer_ft = adabound.AdaBound(model_ft.parameters(),lr=1e-3,final_lr=0.1)
# scheduler_dyn = lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='min',patience=3,verbose=True)
if start_epoch != -1:
    for group in optimizer_ft.param_groups:
        group.setdefault('initial_lr', initial_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=steps, gamma=0.1, last_epoch=start_epoch)

if not os.path.exists('backup'):
    os.mkdir('backup')
backupdir = 'backup/db07/'
if not os.path.exists(backupdir):
        os.mkdir(backupdir)
train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=num_epochs,dyn=False)
