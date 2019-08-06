import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import time

from . import operations


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MyBatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm1d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay**(epoch//self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01


        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class MyBatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm2d, self).__init__(num_features, eps, momentum, affine)
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(MyBatchNorm2d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None) and (epoch >= 1) and (self.momentum_decay_step is not None) and (self.momentum_decay_step > 0):
            # perform momentum decay
            self.momentum = self.momentum_original * (self.momentum_decay**(epoch//self.momentum_decay_step))
            if self.momentum < 0.01:
                self.momentum = 0.01

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=None, normalization=None, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(MyLinear, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.linear = nn.Linear(in_features, out_features, bias=True)
        if self.normalization == 'batch':
            self.norm = MyBatchNorm1d(out_features, momentum=momentum, affine=True, momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm1d(out_features, momentum=momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif 'elu' == activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                n = m.in_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        x = self.linear(x)
        if self.normalization=='batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=None, momentum=0.1, normalization=None, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(MyConv2d, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if self.normalization == 'batch':
            self.norm = MyBatchNorm2d(out_channels, momentum=momentum, affine=True, momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif self.normalization == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, momentum=momentum, affine=True)
        if self.activation == 'relu':
            self.act = nn.ReLU()
        elif self.activation == 'elu':
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        x = self.conv(x)
        if self.normalization=='batch':
            x = self.norm(x, epoch)
        elif self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True, activation=None, normalization=None):
        super(UpConv, self).__init__()
        self.activation = activation
        self.normalization = normalization

        self.up_sample = nn.Upsample(scale_factor=2)
        self.conv = MyConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, activation=activation, normalization=normalization)

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0.001)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv(x)

        return x


class EquivariantLayer(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, activation='relu', normalization=None, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(EquivariantLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.activation = activation
        self.normalization = normalization

        self.conv = nn.Conv1d(self.num_in_channels, self.num_out_channels, kernel_size=1, stride=1, padding=0)

        if 'batch' == self.normalization:
            self.norm = MyBatchNorm1d(self.num_out_channels, momentum=momentum, affine=True, momentum_decay_step=bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif 'instance' == self.normalization:
            self.norm = nn.InstanceNorm1d(self.num_out_channels, momentum=momentum, affine=True)

        if 'relu' == self.activation:
            self.act = nn.ReLU()
        elif 'elu' == self.activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)


        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        # x is NxK, x_max is 1xK
        # x_max, _ = torch.max(x, 0, keepdim=True)
        # y = self.conv(x - x_max.expand_as(x))
        y = self.conv(x)

        if self.normalization=='batch':
            y = self.norm(y, epoch)
        elif self.normalization is not None:
            y = self.norm(y)

        if self.activation is not None:
            y = self.act(y)

        return y


class KNNModule(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(KNNModule, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for c_out in out_channels_list:
            self.layers.append(MyConv2d(previous_out_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True,
                                        activation=activation, normalization=normalization,
                                        momentum=momentum, bn_momentum_decay_step=bn_momentum_decay_step,
                                        bn_momentum_decay=bn_momentum_decay))
            previous_out_channels = c_out

    def forward(self, coordinate, x, precomputed_knn_I, K, center_type, epoch=None):
        '''

        :param coordinate: Bx3xM Variable
        :param x: BxCxM Variable
        :param precomputed_knn_I: BxMxK'
        :param K: K neighbors
        :param center_type: 'center' or 'avg'
        :return:
        '''
        # 0. compute knn
        # 1. for each node, calculate the center of its k neighborhood
        # 2. normalize nodes with the corresponding center
        # 3. fc for these normalized points
        # 4. maxpool for each neighborhood

        coordinate_tensor = coordinate.data  # Bx3xM
        if precomputed_knn_I is not None:
            assert precomputed_knn_I.size()[2] >= K
            knn_I = precomputed_knn_I[:, :, 0:K]
        else:
            coordinate_Mx1 = coordinate_tensor.unsqueeze(3)  # Bx3xMx1
            coordinate_1xM = coordinate_tensor.unsqueeze(2)  # Bx3x1xM
            norm = torch.sum((coordinate_Mx1 - coordinate_1xM) ** 2, dim=1)  # BxMxM, each row corresponds to each coordinate - other coordinates
            knn_D, knn_I = torch.topk(norm, k=K, dim=2, largest=False, sorted=True)  # BxMxK

        # debug
        # print(knn_D[0])
        # print(knn_I[0])
        # assert False

        # get gpu_id
        device_index = x.device.index
        neighbors = operations.knn_gather_wrapper(coordinate_tensor, knn_I)  # Bx3xMxK
        if center_type == 'avg':
            neighbors_center = torch.mean(neighbors, dim=3, keepdim=True)  # Bx3xMx1
        elif center_type == 'center':
            neighbors_center = coordinate_tensor.unsqueeze(3)  # Bx3xMx1
        neighbors_decentered = (neighbors - neighbors_center).detach()
        neighbors_center = neighbors_center.squeeze(3).detach()

        # debug
        # print(neighbors[0, 0])
        # print(neighbors_avg[0, 0])
        # print(neighbors_decentered[0, 0])
        # assert False

        x_neighbors = operations.knn_gather_by_indexing(x, knn_I)  # BxCxMxK
        x_augmented = torch.cat((neighbors_decentered, x_neighbors), dim=1)  # Bx(3+C)xMxK

        for layer in self.layers:
            x_augmented = layer(x_augmented, epoch)
        feature, _ = torch.max(x_augmented, dim=3, keepdim=False)

        return neighbors_center, feature


class PointNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        super(PointNet, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list)-1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

    def forward(self, x, epoch=None):
        for layer in self.layers:
            x = layer(x, epoch)
        return x


class PointResNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1, bn_momentum_decay_step=None, bn_momentum_decay=1):
        '''
        in -> out[0]
        out[0] -> out[1]             ----
        out[1] -> out[2]                |
             ... ...                    |
        out[k-2]+out[1] -> out[k-1]  <---
        :param in_channels:
        :param out_channels_list:
        :param activation:
        :param normalization:
        :param momentum:
        :param bn_momentum_decay_step:
        :param bn_momentum_decay:
        '''
        super(PointResNet, self).__init__()
        self.out_channels_list = out_channels_list

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list)-1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels+out_channels_list[0], c_out, None, None))
            previous_out_channels = c_out

    def forward(self, x, epoch=None):
        '''
        :param x: BxCxN
        :param epoch: None or number of epoch, for BN decay.
        :return:
        '''
        layer0_out = self.layers[0](x, epoch)  # BxCxN
        for l in range(1, len(self.out_channels_list)-1):
            if l == 1:
                x_tmp = self.layers[l](layer0_out, epoch)
            else:
                x_tmp = self.layers[l](x_tmp, epoch)
        layer_final_out = self.layers[len(self.out_channels_list)-1](torch.cat((layer0_out, x_tmp), dim=1), epoch)
        return layer_final_out
