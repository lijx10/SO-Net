import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import time

from util import som
from . import operations
from .layers import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import index_max


class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.opt = opt

        self.first_pointnet = PointNet(3, (32, 64, 128), activation=self.opt.activation, normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(128+128, (256, 256), activation=self.opt.activation, normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        # regressor to get a sin(theta)
        self.fc1 = MyLinear(256, 128, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(128, 64, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(64, 1, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, x, sn=None, epoch=None):
        '''

        :param x: BxMx3 som nodes / Bx3xN points
        :param sn: Bx3xN surface normal
        :return:
        '''

        first_pn_out = self.first_pointnet(x, epoch)  # BxCxN
        feature_1, _ = torch.max(first_pn_out, dim=2, keepdim=False)  # BxC


        second_pn_out = self.second_pointnet(torch.cat((first_pn_out, feature_1.unsqueeze(2).expand_as(first_pn_out)), dim=1), epoch)
        feature_2, _ = torch.max(second_pn_out, dim=2, keepdim=False)

        # get sin(theta)
        fc1_out = self.fc1(feature_2, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)

        sin_theta = torch.tanh(self.fc3(self.fc2_out, epoch))  # Bx1

        return sin_theta


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # transformer
        self.transformer = Transformer(opt)

        # first PointNet
        if self.opt.surface_normal == True:
            self.first_pointnet = PointResNet(6, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointResNet(3, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        if self.opt.som_k >= 2:
            # second PointNet
            self.knnlayer = KNNModule(3+384, (512, 512), activation=self.opt.activation, normalization=self.opt.normalization,
                                      momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

            # final PointNet
            self.final_pointnet = PointNet(3+512, (768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            # final PointNet
            self.final_pointnet = PointResNet(3+384, (512, 512, 768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)



        # build som for clustering, node initalization is done in __init__
        rows = int(math.sqrt(self.opt.node_num))
        cols = rows
        self.som_builder = som.BatchSOM(rows, cols, 3, self.opt.gpu_id, self.opt.batch_size)

        # padding
        self.zero_pad = torch.nn.ZeroPad2d(padding=1)

    def forward(self, x, sn, node, node_knn_I, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param node_knn_I: BxMxk_som LongTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''

        # optimize the som, access the Tensor's tensor, the optimize function should not modify the tensor
        # self.som_builder.optimize(x.data)
        self.som_builder.node.resize_(node.size()).copy_(node)

        # modify the x according to the nodes, minus the center
        self.mask, mask_row_max, min_idx = self.som_builder.query_topk(x.data, k=self.opt.k)  # BxkNxnode_num, Bxnode_num
        mask_row_sum = torch.sum(self.mask, dim=1)  # Bxnode_num
        mask = self.mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_list, sn_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            sn_list.append(sn)
        x_stack = torch.cat(tuple(x_list), dim=2)
        sn_stack = torch.cat(tuple(sn_list), dim=2)

        # re-compute center, instead of using som.node
        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float()+1e-5)  # BxCxnode_num
        self.som_builder.node = cluster_mean
        self.som_node = self.som_builder.node


        # ====== apply transformer to rotate x_stack, sn_stack, som_node ======
        # sin_theta = self.transformer(x=self.som_node, sn=None, epoch=epoch)  # Bx1
        # # sin_theta = self.transformer(x=torch.cat((x_stack, sn_stack), dim=1), sn=None, epoch=epoch)  # Bx1
        # cos_theta = torch.sqrt(1 + 1e-5 - sin_theta*sin_theta)  # Bx1
        # B = x.size()[0]
        # rotation_matrix = torch.Tensor(B, 3, 3).zero_().to(self.opt.device)  # Bx3x3
        # rotation_matrix[:, 0, 0] = cos_theta[:, 0]
        # rotation_matrix[:, 0, 2] = sin_theta[:, 0]
        # rotation_matrix[:, 1, 1] = 1
        # rotation_matrix[:, 2, 0] = -1 * sin_theta[:, 0]
        # rotation_matrix[:, 2, 2] = cos_theta[:, 0]
        # # print(rotation_matrix)

        # x_stack = torch.matmul(rotation_matrix, x_stack)
        # sn_stack = torch.matmul(rotation_matrix, sn_stack)
        # self.som_node = torch.matmul(rotation_matrix, self.som_node)
        # self.som_builder.node = torch.matmul(rotation_matrix.data, self.som_builder.node)
        # ====== apply transformer to rotate x_stack, sn_stack, som_node ======


        # assign each point with a center
        node_expanded = self.som_node.data.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        self.centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        self.x_decentered = (x_stack - self.centers).detach()  # Bx3xkN
        x_augmented = torch.cat((self.x_decentered, sn_stack), dim=1)  # Bx6xkN

        # go through the first PointNet
        if self.opt.surface_normal == True:
            self.first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            self.first_pn_out = self.first_pointnet(self.x_decentered, epoch)

        M = node.size()[2]
        with torch.cuda.device(self.first_pn_out.get_device()):
            gather_index = index_max.forward_cuda(self.first_pn_out.detach(),
                                                  min_idx.int(),
                                                  M).detach().long()
        self.first_pn_out_masked_max = self.first_pn_out.gather(dim=2, index=gather_index * mask_row_max.unsqueeze(1).long())  # BxCxM

        if self.opt.som_k >= 2:
            # second pointnet, knn search on SOM nodes: ----------------------------------
            self.knn_center_1, self.knn_feature_1 = self.knnlayer(self.som_node, self.first_pn_out_masked_max, node_knn_I, self.opt.som_k, self.opt.som_k_type, epoch)

            # final pointnet --------------------------------------------------------------
            self.final_pn_out = self.final_pointnet(torch.cat((self.knn_center_1, self.knn_feature_1), dim=1), epoch)  # Bx1024xM
        else:
            # final pointnet --------------------------------------------------------------
            self.final_pn_out = self.final_pointnet(torch.cat((self.som_node, self.first_pn_out_masked_max), dim=1), epoch)  # Bx1024xM

        self.feature, _ = torch.max(self.final_pn_out, dim=2, keepdim=False)

        return self.feature


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # classifier
        self.fc1 = MyLinear(self.feature_num, 512, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(512, 256, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(256, self.opt.classes, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, feature, epoch=None):
        fc1_out = self.fc1(feature, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)
        score = self.fc3(self.fc2_out, epoch)

        return score


class Segmenter(nn.Module):
    def __init__(self, opt):
        super(Segmenter, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # segmenter
        if self.opt.surface_normal == True:
            if self.opt.som_k >= 2:
                in_channels = 3 + 3 + 3 + 3 + 16 + 384 + 384 + 512 + self.feature_num * 2
            else:
                in_channels = 3 + 3 + 3 + 3 + 16 + 384 + 384 + self.feature_num * 2
        else:
            if self.opt.som_k >= 2:
                in_channels = 3 + 3 + 3 + 16 + 384 + 384 + 512 + self.feature_num * 2
            else:
                in_channels = 3 + 3 + 3 + 16 + 384 + 384 + self.feature_num * 2

        self.layer1 = EquivariantLayer(in_channels,
                                       1024,
                                       activation=self.opt.activation,
                                       normalization=self.opt.normalization)
        self.layer2 = EquivariantLayer(1024, 512, activation=self.opt.activation, normalization=self.opt.normalization)
        self.layer3 = EquivariantLayer(512, 256, activation=self.opt.activation, normalization=self.opt.normalization)
        self.drop3 = nn.Dropout(p=self.opt.dropout)
        self.layer4 = EquivariantLayer(256, 128, activation=self.opt.activation, normalization=self.opt.normalization)
        self.drop4 = nn.Dropout(p=self.opt.dropout)
        self.layer5 = EquivariantLayer(128, self.opt.classes, activation=None, normalization=None)

    def forward(self, x_decentered, x, centers, sn, label,
                first_pn_out,
                feature_max_first_pn_out,
                feature_max_knn_feature_1,
                feature_max_final_pn_out,
                feature):
        '''
        :param x_decentered: Bx3xkN
        :param x: Bx3xN
        :param centers: Bx3xkN
        :param sn: Bx3xkN
        :param label: B, tensor
        :param first_pn_out: Bx384xkN
        :param feature_max_first_pn_out: Bx384xM
        :param feature_max_knn_feature_1: Bx512xM
        :param feature_max_final_pn_out: Bx1024xM
        :param feature: Bx1024
        :return:
        '''

        B = x.size()[0]
        N = x.size()[2]
        k = self.opt.k
        kN= round(k*N)

        # if necessary, stack the x
        x_list, sn_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            sn_list.append(sn)
        x = torch.cat(tuple(x_list), dim=2)
        sn = torch.cat(tuple(sn_list), dim=2)

        label_onehot = torch.FloatTensor(B, 16).zero_().to(self.opt.device)  # Bx16
        label_onehot.scatter_(1, label.unsqueeze(1), 1)  # Bx16
        label_onehot = label_onehot.unsqueeze(2).expand(B, 16, kN).detach()  # Bx16xkN

        feature_expanded = feature.unsqueeze(2).expand(B, self.feature_num, kN)

        if self.opt.surface_normal == True:
            if self.opt.som_k >= 2:
                layer1_in = torch.cat((x_decentered, x, centers, sn, label_onehot,
                                       first_pn_out,
                                       feature_max_first_pn_out,
                                       feature_max_knn_feature_1,
                                       feature_max_final_pn_out,
                                       feature_expanded), dim=1)
            else:
                layer1_in = torch.cat((x_decentered, x, centers, sn, label_onehot,
                                       first_pn_out,
                                       feature_max_first_pn_out,
                                       feature_max_final_pn_out,
                                       feature_expanded), dim=1)
        else:
            if self.opt.som_k >= 2:
                layer1_in = torch.cat((x_decentered, x, centers, label_onehot,
                                       first_pn_out,
                                       feature_max_first_pn_out,
                                       feature_max_knn_feature_1,
                                       feature_max_final_pn_out,
                                       feature_expanded), dim=1)
            else:
                layer1_in = torch.cat((x_decentered, x, centers, label_onehot,
                                       first_pn_out,
                                       feature_max_first_pn_out,
                                       feature_max_final_pn_out,
                                       feature_expanded), dim=1)
        layer1_out = self.layer1(layer1_in)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)

        # split into k * BxN
        splited_list = torch.split(layer3_out, self.opt.input_pc_num, dim=2)  # BxCxkN -> k * BxCxN
        assert (len(splited_list) == k)
        if k==2:
            layer3_avg_out = 0.5 * (splited_list[0] + splited_list[1])  # BxCxN
        elif k==3:
            layer3_avg_out = (1.0/3.0) * (splited_list[0] + splited_list[1] + splited_list[2])  # BxCxN

        layer4_out = self.layer4(layer3_avg_out)
        if self.opt.dropout > 0.1:
            layer4_out = self.drop4(layer4_out)
        layer5_out = self.layer5(layer4_out)  # Bx50xN


        return layer5_out


class DecoderLinear(nn.Module):
    def __init__(self, opt):
        super(DecoderLinear, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_number = opt.output_fc_pc_num

        self.linear1 = MyLinear(self.feature_num, self.output_point_number*2, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear2 = MyLinear(self.output_point_number*2, self.output_point_number*3, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear3 = MyLinear(self.output_point_number*3, self.output_point_number*4, activation=self.opt.activation, normalization=self.opt.normalization)
        self.linear_out = MyLinear(self.output_point_number*4, self.output_point_number*3, activation=None, normalization=None)

        # special initialization for linear_out, to get uniform distribution over the space
        self.linear_out.linear.bias.data.uniform_(-1, 1)

    def forward(self, x):
        # reshape from feature vector NxC, to NxC
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear_out(x)

        return x.view(-1, 3, self.output_point_number)


class ConvToPC(nn.Module):
    def __init__(self, in_channels, opt):
        super(ConvToPC, self).__init__()
        self.in_channels = in_channels
        self.opt = opt

        self.conv1 = MyConv2d(self.in_channels, int(self.in_channels), kernel_size=1, stride=1, padding=0, bias=True, activation=opt.activation, normalization=opt.normalization)
        self.conv2 = MyConv2d(int(self.in_channels), 3, kernel_size=1, stride=1, padding=0, bias=True, activation=None, normalization=None)

        # special initialization for conv2, to get uniform distribution over the space
        # self.conv2.conv.bias.data.normal_(0, 0.3)
        self.conv2.conv.bias.data.uniform_(-1, 1)

        # self.conv2.conv.weight.data.normal_(0, 0.01)
        # self.conv2.conv.bias.data.uniform_(-3, 3)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class DecoderConv(nn.Module):
    def __init__(self, opt):
        super(DecoderConv, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num
        self.output_point_num = opt.output_conv_pc_num

        # __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True, activation=None, normalization=None)
        # 1x1 -> 2x2
        self.deconv1 = UpConv(self.feature_num, int(self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization)
        # 2x2 -> 4x4
        self.deconv2 = UpConv(int(self.feature_num), int(self.feature_num/2), activation=self.opt.activation, normalization=self.opt.normalization)
        # 4x4 -> 8x8
        self.deconv3 = UpConv(int(self.feature_num/2), int(self.feature_num/4), activation=self.opt.activation, normalization=self.opt.normalization)
        # 8x8 -> 16x16
        self.deconv4 = UpConv(int(self.feature_num/4), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc4 = ConvToPC(int(self.feature_num/8), opt)
        # 16x16 -> 32x32
        self.deconv5 = UpConv(int(self.feature_num/8), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc5 = ConvToPC(int(self.feature_num/8), opt)
        # 32x32 -> 64x64
        self.deconv6 = UpConv(int(self.feature_num/8), int(self.feature_num/8), activation=self.opt.activation, normalization=self.opt.normalization)
        self.conv2pc6 = ConvToPC(int(self.feature_num/8), opt)


    def forward(self, x):
        # reshape from feature vector NxC, to NxCx1x1
        x = x.view(-1, self.feature_num, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        self.pc4 = self.conv2pc4(x)
        x = self.deconv5(x)
        self.pc5 = self.conv2pc5(x)
        x = self.deconv6(x)
        self.pc6 = self.conv2pc6(x)

        return self.pc6


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        if self.opt.output_fc_pc_num > 0:
            self.fc_decoder = DecoderLinear(opt)
        self.conv_decoder = DecoderConv(opt)

    def forward(self, x):
        if self.opt.output_fc_pc_num > 0:
            self.linear_pc = self.fc_decoder(x)

        if self.opt.output_conv_pc_num > 0:
            self.conv_pc6 = self.conv_decoder(x).view(-1, 3, 4096)
            self.conv_pc4 = self.conv_decoder.pc4.view(-1, 3, 256)
            self.conv_pc5 = self.conv_decoder.pc5.view(-1, 3, 1024)

        if self.opt.output_fc_pc_num == 0:
            if self.opt.output_conv_pc_num == 4096:
                return self.conv_pc6
            elif self.opt.output_conv_pc_num == 1024:
                return self.conv_pc5
        else:
            if self.opt.output_conv_pc_num == 4096:
                return torch.cat([self.linear_pc, self.conv_pc6], 2)
            elif self.opt.output_conv_pc_num == 1024:
                return torch.cat([self.linear_pc, self.conv_pc5], 2)
            else:
                return self.linear_pc