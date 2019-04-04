import torch
import torch.nn as nn
import torch.nn.functional as F
from util import readcfg

d = readcfg('cfg/yolond')
softmax = int(d['softmax'])


class YOLOLoss(nn.Module):
    def __init__(self, side, num, sqrt, coord_scale, noobj_scale, use_gpu=True):
        super(YOLOLoss, self).__init__()
        self.side = side
        self.num = num
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.sqrt = sqrt
        # self.use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu

    def compute_iou(self, box1, box2):
        """
        compute n predicted
        :param box1: [N,4(xmin,ymin,xmax,ymax)]
        :param box2: [1,4(xmin,ymin,xmax,ymax)]
        :return: [N]

        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0)

        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou

    def forward(self, preds, labels):
        return self.loss_1(preds, labels)

    # def loss(self, preds, labels):
    #     """
    #     preds-[batchsize, side, side, n*(4+1)+20]
    #     labels-[batchsize, side, side, 4+1+20]
    #
    #     """
    #     avg_iou = 0
    #     avg_cat = 0
    #     avg_allcat = 0
    #     avg_obj = 0
    #     avg_anyobj = 0
    #     # count = 0
    #     # cost = 0
    #     bbox_size = self.num*5
    #     cell_size = bbox_size + 20
    #     batchsize = preds.size(0)
    #
    #     preds = preds.view(-1, cell_size)
    #     delta = torch.FloatTensor(preds.size()).zero_()
    #
    #     #no object
    #     delta[:,4] = self.noobj_scale*(0 - preds[:,4])
    #     # cost += torch.sum(self.noobj_scale*torch.pow(preds[:,4],2))
    #     avg_anyobj += torch.sum(preds[:,4])
    #
    #
    #     labels = labels.view(-1, 25)
    #
    #     # indices of object confidence non-zero batch
    #     non_zero_ind = torch.nonzero(labels[:, 4])
    #
    #     labels_contain = labels[non_zero_ind].view(-1, 25)
    #     preds_contain = preds[non_zero_ind].view(-1, cell_size)
    #
    #     #class
    #     delta[non_zero_ind, bbox_size:] = 1.0 * (labels_contain[:,5:]-preds_contain[:,bbox_size:])
    #     # cost += 1.0 * torch.sum(torch.pow(labels_contain[:,5:]-preds_contain[:,bbox_size:], 2))
    #     avg_allcat += torch.sum(preds_contain[:, bbox_size:])
    #     for i in range(labels_contain.size(0)):
    #         label = labels_contain[i]
    #         pred = preds_contain[i]
    #         class_non_zero_ind = torch.nonzero(label[5:])
    #         avg_cat += torch.sum(pred[bbox_size+class_non_zero_ind])
    #
    #     #bbox
    #     box_label = labels_contain[:,:5]
    #     box_pred = preds_contain[:,:bbox_size]
    #
    #
    #     box_label_iou = box_label.clone()
    #     box_pred_iou = box_pred.clone()
    #
    #     box_label_iou[:,:2] = box_label[:,:2]/self.side - 0.5*box_label[:,2:4]
    #     box_label_iou[:,2:4] = box_label[:,:2]/self.side + 0.5*box_label[:,2:4]
    #
    #     for i in range(self.num):
    #         if self.sqrt:
    #             box_pred_wh = torch.pow(box_pred[:, i*5+2:i*5+4], 2)
    #         else:
    #             box_pred_wh = box_pred[:, i*5+2:i*5+4]
    #         box_pred_iou[:, i*5:i*5+2] = box_pred[:, i*5:i*5+2]/self.side - 0.5*box_pred_wh
    #         box_pred_iou[:, i*5+2:i*5+4] = box_pred[:, i*5:i*5+2]/self.side + 0.5*box_pred_wh
    #
    #     for i in range(box_label_iou.size(0)):
    #         truth = box_label_iou[i].view(-1, 5)
    #         out = box_pred_iou[i].view(-1, 5)
    #         iou = self.compute_iou(out, truth)
    #         assert iou.shape == (self.num, 1)
    #
    #         best_iou, best_index = iou.max(0)
    #         # best_bbox = out[best_index]
    #         pindex = non_zero_ind[i]
    #         # cost -= self.noobj_scale * torch.pow(preds[pindex, best_index*5+4],2)
    #         # cost += 1.0 * torch.pow(1.0-preds[pindex, best_index*5+4], 2)
    #
    #         avg_obj += preds[pindex, best_index*5+4]
    #         delta[pindex, best_index*5+4] = 1.0 * (1.0-preds[pindex, best_index*5+4])
    #         delta[pindex, best_index*5:best_index*5+4] = \
    #             self.coord_scale * (box_label[i, 0:4] - box_pred[i, best_index*5:best_index*5+4])
    #         if self.sqrt:
    #             delta[pindex, best_index*5+2:best_index*5+4] = \
    #                 self.coord_scale * (torch.sqrt(box_label[i,2:4]) - box_pred[i, best_index*5+2:best_index*5+4])
    #
    #         # cost += torch.pow(1-iou,2)
    #         avg_iou += best_iou
    #
    #     count = labels_contain.size(0)
    #     cost = F.mse_loss(torch.FloatTensor(delta.shape).zero_(), delta, reduction="sum")
    #
    #     print("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n"
    #           % (avg_iou/count, avg_cat/iou, avg_allcat/(count*20), avg_obj/count,
    #              avg_anyobj/(batchsize*self.side*self.side*self.num), count))
    #
    #     return cost

    def loss_1(self,preds,labels):
        '''
        preds: (tensor) size(batchsize,S,S,Bx5+20) [x,y,w,h,c]
        labels: (tensor) size(batchsize,S,S,Bx5+20)
        '''

        # print(preds.shape)
        # print(labels.shape)

        N = preds.size(0)
        bbox_size = self.num * 5
        cell_size = bbox_size + 20

        obj_mask = labels[:, :, :, 4] > 0
        noobj_mask = labels[:, :, :, 4] == 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(labels)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(labels)

        obj_pred = preds[obj_mask].view(-1, cell_size)
        box_pred = obj_pred[:, :bbox_size].contiguous().view(-1, 5)
        class_pred = obj_pred[:, bbox_size:]

        obj_label = labels[obj_mask].view(-1, cell_size)
        box_label = obj_label[:, :bbox_size].contiguous().view(-1, 5)
        class_label = obj_label[:, bbox_size:]

        # compute not containing loss
        noobj_pred = preds[noobj_mask].view(-1, cell_size)
        noobj_label = labels[noobj_mask].view(-1, cell_size)
        noobj_pred_mask = torch.cuda.ByteTensor(noobj_pred.size()) if self.use_gpu else torch.ByteTensor(
            noobj_pred.size())
        noobj_pred_mask.zero_()
        for i in range(self.num):
            noobj_pred_mask[:, i * 5 + 4] = 1
        noobj_pred_c = noobj_pred[noobj_pred_mask]
        noobj_label_c = noobj_label[noobj_pred_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_label_c, reduction="sum")

        # object containing loss
        obj_response_mask = torch.cuda.ByteTensor(box_label.size()) if self.use_gpu else torch.ByteTensor(
            box_label.size())
        obj_response_mask.zero_()
        obj_not_response_mask = torch.cuda.ByteTensor(box_label.size()) if self.use_gpu else torch.ByteTensor(
            box_label.size())
        obj_not_response_mask.zero_()
        box_label_iou = torch.zeros(box_label.size())
        if self.use_gpu:
            box_label_iou = box_label_iou.cuda()

        s = 1/self.side
        for i in range(0, box_label.size(0), self.num):
            box1 = box_pred[i:i+self.num]
            box1_coord = torch.FloatTensor(box1.size())
            box1_coord[:, :2] = box1[:, :2] * s - 0.5 * box1[:, 2:4]
            box1_coord[:, 2:4] = box1[:, :2] * s + 0.5 * box1[:, 2:4]

            box2 = box_label[i].view(-1, 5)
            box2_coord = torch.FloatTensor(box2.size())
            box2_coord[:, :2] = box2[:, :2] * s - 0.5 * box2[:, 2:4]
            box2_coord[:, 2:4] = box2[:, :2] * s + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_coord[:, :4], box2_coord[:, :4])
            # print(iou.shape)
            # assert iou.shape[0] == self.num

            max_iou, max_index = iou.max(0)

            obj_response_mask[i + max_index] = 1
            # obj_not_response_mask[i + 1 - max_index] = 1
            obj_not_response_mask[i:i + self.num] = 1
            obj_not_response_mask[i + max_index] = 0

            box_label_iou[i + max_index, torch.LongTensor([4])] = max_iou.data.cuda()  # no grad

        # response loss
        box_pred_response = box_pred[obj_response_mask].view(-1, 5)
        box_label_response_iou = box_label_iou[obj_response_mask].view(-1, 5)
        box_label_response = box_label[obj_response_mask].view(-1, 5)
        response_loss = F.mse_loss(box_pred_response[:, 4], box_label_response_iou[:, 4], reduction="sum")
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_label_response[:, :2], reduction="sum")
        wh_loss = F.mse_loss(torch.sqrt(box_pred_response[:,2:4]), torch.sqrt(box_label_response[:,2:4]), reduction="sum")

        # not response loss
        box_pred_not_response = box_pred[obj_not_response_mask].view(-1, 5)
        box_label_not_response = box_label[obj_not_response_mask].view(-1, 5)
        box_label_not_response[:, 4] = 0
        not_response_loss = F.mse_loss(box_pred_not_response[:, 4], box_label_not_response[:, 4], reduction="sum")

        # class loss
        class_loss = F.mse_loss(class_pred, class_label, reduction="sum")

        total_loss = self.coord_scale*(xy_loss+wh_loss)+2.*response_loss+not_response_loss+self.noobj_scale*noobj_loss+class_loss

        return total_loss / N

    def loss_2(self,preds,labels):
        '''

        preds: (tensor) size(batchsize,S,S,Bx5+20) [x,y,w,h,c]
        labels: (tensor) size(batchsize,S,S,Bx5+20)

        '''

        # print(preds.shape)
        # print(labels.shape)

        N = preds.size(0)
        bbox_size = self.num * 5
        cell_size = bbox_size + 20

        obj_mask = labels[:, :, :, 4] > 0
        noobj_mask = labels[:, :, :, 4] == 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(labels)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(labels)

        obj_pred = preds[obj_mask].view(-1, cell_size)
        box_pred = obj_pred[:, :bbox_size].contiguous().view(-1, 5)
        class_pred = obj_pred[:, bbox_size:]

        obj_label = labels[obj_mask].view(-1, cell_size)
        box_label = obj_label[:, :bbox_size].contiguous().view(-1, 5)
        class_label = obj_label[:, bbox_size:]

        # compute not containing loss
        noobj_pred = preds[noobj_mask].view(-1, cell_size)
        noobj_label = labels[noobj_mask].view(-1, cell_size)
        noobj_pred_mask = torch.cuda.ByteTensor(noobj_pred.size()) if self.use_gpu else torch.ByteTensor(
            noobj_pred.size())
        noobj_pred_mask.zero_()
        for i in range(self.num):
            noobj_pred_mask[:, i * 5 + 4] = 1
        noobj_pred_c = noobj_pred[noobj_pred_mask]
        noobj_label_c = noobj_label[noobj_pred_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_label_c, reduction="sum")

        # object containing loss
        obj_response_mask = torch.cuda.ByteTensor(box_label.size()) if self.use_gpu else torch.ByteTensor(
            box_label.size())
        obj_response_mask.zero_()
        obj_not_response_mask = torch.cuda.ByteTensor(box_label.size()) if self.use_gpu else torch.ByteTensor(
            box_label.size())
        obj_not_response_mask.zero_()
        box_label_iou = torch.zeros(box_label.size())
        if self.use_gpu:
            box_label_iou = box_label_iou.cuda()

        s = 1/self.side
        for i in range(0, box_label.size(0), self.num):
            box1 = box_pred[i:i + self.num]
            box1_coord = torch.FloatTensor(box1.size())  # Variable
            box1_coord[:, :2] = box1[:, :2] * s - 0.5 * box1[:, 2:4]
            box1_coord[:, 2:4] = box1[:, :2] * s + 0.5 * box1[:, 2:4]

            box2 = box_label[i].view(-1, 5)
            box2_coord = torch.FloatTensor(box2.size())  # Variable
            box2_coord[:, :2] = box2[:, :2] * s - 0.5 * box2[:, 2:4]
            box2_coord[:, 2:4] = box2[:, :2] * s + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_coord[:, :4], box2_coord[:, :4])
            # print(iou.shape)
            # assert iou.shape[0] == self.num

            max_iou, max_index = iou.max(0)
            # max_index = max_index.data.cuda()

            obj_response_mask[i + max_index] = 1
            # obj_not_response_mask[i + 1 - max_index] = 1
            obj_not_response_mask[i:i + self.num] = 1
            obj_not_response_mask[i + max_index] = 0

            box_label_iou[i + max_index, torch.LongTensor([4])] = max_iou.data.cuda()

        # response loss
        box_pred_response = box_pred[obj_response_mask].view(-1, 5)
        box_label_response_iou = box_label_iou[obj_response_mask].view(-1, 5)
        box_label_response = box_label[obj_response_mask].view(-1, 5)
        response_loss = F.mse_loss(box_pred_response[:, 4], box_label_response_iou[:, 4], reduction="sum")
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_label_response[:, :2], reduction="sum")
        # xy_loss = F.smooth_l1_loss(box_pred_response[:, :2], box_label_response[:, :2], reduction="sum")
        if self.sqrt:
            # wh_loss = F.mse_loss(torch.sqrt(box_pred_response[:,2:4]), torch.sqrt(box_label_response[:,2:4]), reduction="sum")
            # wh_loss = F.smooth_l1_loss(torch.sqrt(box_pred_response[:,2:4]), torch.sqrt(box_label_response[:,2:4]), reduction="sum")
            scale = 2. * torch.sigmoid(torch.pow(box_pred_response[:, 2:4] / box_label_response[:, 2:4] - 1., 2))
            wh_loss = torch.sum(scale * torch.pow(box_pred_response[:, 2:4] - box_label_response[:, 2:4], 2))
        else:
            wh_loss = F.mse_loss(box_pred_response[:, 2:4], box_label_response[:, 2:4], reduction="sum")

        # not response loss
        box_pred_not_response = box_pred[obj_not_response_mask].view(-1, 5)
        box_label_not_response = box_label[obj_not_response_mask].view(-1, 5)
        box_label_not_response[:, 4] = 0
        not_response_loss = F.mse_loss(box_pred_not_response[:, 4], box_label_not_response[:, 4], reduction="sum")

        # class loss
        if softmax:
            class_loss = F.cross_entropy(class_pred, class_label.max(1)[1], reduction="sum")
        else:
            class_loss = F.mse_loss(class_pred, class_label, reduction="sum")
        print("xy loss:{:.4f},wh loss:{:.4f},class loss:{:.4f},noobj loss:{:.4f}".format(xy_loss, wh_loss, class_loss,noobj_loss))
        total_loss = self.coord_scale*(xy_loss+wh_loss)+2.*response_loss+not_response_loss+self.noobj_scale*noobj_loss+class_loss

        return total_loss / N


if __name__ == '__main__':
    yololoss = YOLOLoss(14,2,1,5,.5)
    torch.manual_seed(1)
    pred = torch.rand(1, 14, 2, 30).cuda()
    target = torch.rand(pred.shape).cuda()
    loss = yololoss(pred, target)
    # loss_1 =yololoss.loss(pred,target)
    print(loss)  # 181.4906


