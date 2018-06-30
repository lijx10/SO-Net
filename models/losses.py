import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import time
import torch.nn.functional as F
import faiss

import json
import os
import os.path
from collections import OrderedDict


def robust_norm(var):
    '''
    :param var: Variable of BxCxHxW
    :return: p-norm of BxCxW
    '''
    result = ((var**2).sum(dim=2) + 1e-8).sqrt()
    # result = (var ** 2).sum(dim=2)

    # try to make the points less dense, caused by the backward loss
    # result = result.clamp(min=7e-3, max=None)
    return result


class CrossEntropyLossSeg(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLossSeg, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)

    def forward(self, inputs, targets):
        '''
        :param inputs: BxclassxN
        :param targets: BxN
        :return:
        '''
        inputs = inputs.unsqueeze(3)
        targets = targets.unsqueeze(2)
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def visualize_pc_seg(score, seg, label, visualizer, opt, input_pc, batch_num):
    # display only one instance of pc/img
    input_pc_np = input_pc.cpu().numpy().transpose()  # Nx3
    pc_color_np = np.ones(input_pc_np.shape, dtype=int)  # Nx3
    gt_pc_color_np = np.ones(input_pc_np.shape, dtype=int)  # Nx3

    # construct color map
    _, predicted_seg = torch.max(score, dim=0, keepdim=False)  # 50xN -> N
    predicted_seg_np = predicted_seg.cpu().numpy()  # N
    gt_seg_np = seg.cpu().numpy()  # N

    color_map_file = os.path.join(opt.dataroot, 'part_color_mapping.json')
    color_map = json.load(open(color_map_file, 'r'))
    color_map_np = np.fabs((np.asarray(color_map) * 255)).astype(int)  # 50x3

    for i in range(input_pc_np.shape[0]):
        pc_color_np[i] = color_map_np[predicted_seg_np[i]]
        gt_pc_color_np[i] = color_map_np[gt_seg_np[i]]
        if gt_seg_np[i] == 49:
            gt_pc_color_np[i] = np.asarray([1, 1, 1]).astype(int)

    dict = OrderedDict([('pc_colored_predicted', [input_pc_np, pc_color_np]),
                        ('pc_colored_gt', [input_pc_np, gt_pc_color_np])])

    visualizer.display_current_results(dict, 1, 1)


def compute_iou_np_array(score, seg, label, visualizer, opt, input_pc):
    part_label = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49]
    ]

    _, seg_predicted = torch.max(score, dim=1)  # BxN

    iou_batch = []
    for i in range(score.size()[0]):
        iou_pc = []
        for part in part_label[label[i]]:
            gt = seg[i] == part
            predict = seg_predicted[i] == part

            intersection = (gt + predict) == 2
            union = (gt + predict) >= 1

            if union.sum() == 0:
                iou_part = 1.0
            else:
                iou_part = intersection.int().sum().item() / (union.int().sum().item() + 0.0001)

            iou_pc.append(iou_part)

        iou_batch.append(np.asarray(iou_pc).mean())

    iou_np = np.asarray(iou_batch)

    return iou_np


def compute_iou(score, seg, label, visualizer, opt, input_pc):
    '''
    :param score: BxCxN tensor
    :param seg: BxN tensor
    :return:
    '''

    part_label = [
        [0, 1, 2, 3],
        [4, 5],
        [6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23],
        [24, 25, 26, 27],
        [28, 29],
        [30, 31, 32, 33, 34, 35],
        [36, 37],
        [38, 39, 40],
        [41, 42, 43],
        [44, 45, 46],
        [47, 48, 49]
    ]

    _, seg_predicted = torch.max(score, dim=1)  # BxN

    iou_batch = []
    vis_flag = False
    for i in range(score.size()[0]):
        iou_pc = []
        for part in part_label[label[i]]:
            gt = seg[i] == part
            predict = seg_predicted[i] == part

            intersection = (gt + predict) == 2
            union = (gt + predict) >= 1

            # print(intersection)
            # print(union)
            # assert False

            if union.sum() == 0:
                iou_part = 1.0
            else:
                iou_part = intersection.int().sum().item() / (union.int().sum().item() + 0.0001)

            # debug to see what happened
            # if iou_part < 0.1:
            #     print(part)
            #     print('predict:')
            #     print(predict.nonzero())
            #     print('gt')
            #     print(gt.nonzero())
            #     vis_flag = True

            iou_pc.append(iou_part)

        # debug to see what happened
        if vis_flag:
            print('============')
            print(iou_pc)
            print(label[i])
            visualize_pc_seg(score[i], seg[i], label[i], visualizer, opt, input_pc[i], i)

        iou_batch.append(np.asarray(iou_pc).mean())

    iou = np.asarray(iou_batch).mean()

    return iou


class ChamferLoss(nn.Module):
    def __init__(self, opt):
        super(ChamferLoss, self).__init__()
        self.opt = opt
        self.dimension = 3
        self.k = 1

        # we need only a StandardGpuResources per GPU
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemoryFraction(0.1)
        self.flat_config = faiss.GpuIndexFlatConfig()
        self.flat_config.device = opt.gpu_id

        # place holder
        self.forward_loss = torch.FloatTensor([0])
        self.backward_loss = torch.FloatTensor([0])

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)
        index = faiss.index_cpu_to_gpu(self.res, self.opt.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        '''
        D, I = index.search(query, k)

        D_var =torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        if self.opt.gpu_id >= 0:
            D_var = D_var.to(self.opt.device)
            I_var = I_var.to(self.opt.device)

        return D_var, I_var

    def forward(self, predict_pc, gt_pc):
        '''
        :param predict_pc: Bx3xM Variable in GPU
        :param gt_pc: Bx3xN Variable in GPU
        :return:
        '''

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy())  # BxMx3
        gt_pc_np = np.ascontiguousarray(torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy())  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2])
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2])

        if self.opt.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.opt.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.opt.device)

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i,k,...] = gt_pc[i].index_select(1, I_var[:,k])

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i,k,...] = predict_pc[i].index_select(1, I_var[:,k])

        # compute loss ===================================================
        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        forward_loss_element = robust_norm(selected_gt_by_predict-predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict))
        self.forward_loss = forward_loss_element.mean()
        self.forward_loss_array = forward_loss_element.mean(dim=1).mean(dim=1)

        # selected_predict(Bxkx3xN) vs gt_pc(Bx3xN)
        backward_loss_element = robust_norm(selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt))  # BxkxN
        self.backward_loss = backward_loss_element.mean()
        self.backward_loss_array = backward_loss_element.mean(dim=1).mean(dim=1)

        self.loss_array = self.forward_loss_array + self.backward_loss_array
        return self.forward_loss + self.backward_loss # + self.sparsity_loss

    def __call__(self, predict_pc, gt_pc):
        # start_time = time.time()
        loss = self.forward(predict_pc, gt_pc)
        # print(time.time()-start_time)
        return loss