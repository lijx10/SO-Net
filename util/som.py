import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import time
import gc

import torch
import torchvision

from . import potential_field


class SOM():
    def __init__(self, rows=4, cols=4, dim=3, gpu_id=-1):
        '''
        Can't put into dataloader, because dataloader keeps only 1 class instance. So this should be used offline,
        to save som result into numpy array.
        :param rows:
        :param cols:
        :param dim:
        :param gpu_id:
        '''
        self.rows = rows
        self.cols = cols
        self.dim = dim
        self.node_num = rows * cols

        self.sigma = 0.4
        self.learning_rate = 0.5
        self.max_iteration = 60

        self.gpu_id = gpu_id

        # node: Cx(rowsxcols), tensor
        self.node = torch.FloatTensor(self.dim, self.rows * self.cols).zero_()
        self.node_idx_list = torch.from_numpy(np.arange(self.rows * self.cols).astype(np.float32))
        self.init_weighting_matrix = torch.FloatTensor(self.node_num, self.rows, self.cols)  # node_numxrowsxcols
        if self.gpu_id >= 0:
            self.node = self.node.to(self.device)
            self.node_idx_list = self.node_idx_list.to(self.device)
            self.init_weighting_matrix = self.init_weighting_matrix.to(self.device)

        self.get_init_weighting_matrix()

        # initialize the node by potential field
        pf = potential_field.PotentialField(self.node_num, self.dim)
        pf.optimize()
        self.node_init_value = torch.from_numpy(pf.node.transpose().astype(np.float32))

    def node_init(self):
        self.node.copy_(self.node_init_value)

    def get_init_weighting_matrix(self):
        '''
        get the initial weighting matrix, later the weighting matrix wil base on the init.
        '''
        for idx in range(self.rows * self.cols):
            (i, j) = self.idx2multi(idx)
            self.init_weighting_matrix[idx, :] = self.gaussian((i, j), self.sigma)
        if self.gpu_id >= 0:
            self.init_weighting_matrix = self.init_weighting_matrix.to(self.device)

    def get_weighting_matrix(self, sigma):
        scale = 1.0 / ((sigma / self.sigma) ** 2)
        weighting_matrix = torch.exp(torch.log(self.init_weighting_matrix) * scale)
        return weighting_matrix

    def gaussian(self, c, sigma):
        """Returns a Gaussian centered in c"""
        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(np.arange(self.rows) - c[0], 2) / d)
        ay = np.exp(-np.power(np.arange(self.cols) - c[1], 2) / d)
        return torch.from_numpy(np.outer(ax, ay).astype(np.float32))

    def idx2multi(self, i):
        return (i // self.cols, i % self.cols)

    def query(self, x):
        '''
        :param x: input data CxN tensor
        :return: mask: Nxnode_num
        '''
        # expand as CxNxnode_num
        node = self.node.unsqueeze(1).expand(x.size(0), x.size(1), self.rows * self.cols)
        x_expanded = x.unsqueeze(2).expand_as(node)

        # calcuate difference between x and each node
        diff = x_expanded - node  # CxNxnode_num
        diff_norm = (diff ** 2).sum(dim=0)  # Nxnode_num

        # find the nearest neighbor
        _, min_idx = torch.min(diff_norm, dim=1)  # N
        min_idx_expanded = min_idx.unsqueeze(1).expand(min_idx.size()[0], self.rows * self.cols).float()  # Nxnode_num

        node_idx_list = self.node_idx_list.unsqueeze(0).expand_as(min_idx_expanded)  # Nxnode_num
        mask = torch.eq(min_idx_expanded, node_idx_list).float()  # Nxnode_num
        mask_row_max, _ = torch.max(mask, dim=0)  # node_num, this indicates whether the node has nearby x

        return mask, mask_row_max

    def batch_update(self, x, iteration):
        # x is CxN tensor, C==self.dim, W=1
        assert (x.size()[0] == self.dim)

        # get learning_rate and sigma
        learning_rate = self.learning_rate / (1 + 2 * iteration / self.max_iteration)
        sigma = self.sigma / (1 + 2 * iteration / self.max_iteration)

        # expand as CxNxnode_num
        node = self.node.unsqueeze(1).expand(x.size(0), x.size(1), self.rows * self.cols)
        x_expanded = x.unsqueeze(2).expand_as(node)

        # calcuate difference between x and each node
        diff = x_expanded - node  # CxNxnode_num
        diff_norm = (diff ** 2).sum(dim=0)  # Nxnode_num

        # find the nearest neighbor
        _, min_idx = torch.min(diff_norm, dim=1)  # N
        min_idx_expanded = min_idx.unsqueeze(1).expand(min_idx.size()[0], self.rows * self.cols).float()  # Nxnode_num

        node_idx_list = self.node_idx_list.unsqueeze(0).expand_as(min_idx_expanded)  # Nxnode_num
        mask = torch.eq(min_idx_expanded, node_idx_list).float()  # Nxnode_num
        mask_row_sum = torch.sum(mask, dim=0) + 0.00001  # node_num
        mask_row_max, _ = torch.max(mask, dim=0)  # node_num, this indicates whether the node has nearby x

        # calculate the mean x for each node
        x_expanded_masked = x_expanded * mask.unsqueeze(0).expand_as(x_expanded)  # CxNxnode_num
        x_expanded_masked_sum = torch.sum(x_expanded_masked, dim=1)  # Cxnode_num
        x_expanded_mask_mean = x_expanded_masked_sum / mask_row_sum.unsqueeze(0).expand_as(
            x_expanded_masked_sum)  # Cxnode_num

        # each x_expanded_mask_mean (in total node_num vectors) will calculate its diff with all nodes
        # multiply the mask_row_max, so that the isolated node won't be pulled to the center
        x_expanded_mask_mean_expanded = x_expanded_mask_mean.unsqueeze(2).expand(self.dim, self.rows * self.cols,
                                                                                 self.rows * self.cols)  # Cxnode_numxnode_num
        node_expanded_transposed = self.node.unsqueeze(1).expand_as(x_expanded_mask_mean_expanded)  # .transpose(1,2)
        diff_masked_mean = x_expanded_mask_mean_expanded - node_expanded_transposed  # Cxnode_numxnode_num
        diff_masked_mean = diff_masked_mean * mask_row_max.unsqueeze(1).unsqueeze(0).expand_as(diff_masked_mean)

        # compute the neighrbor weighting
        #         weighting_matrix = torch.FloatTensor(self.rows*self.cols, self.rows, self.cols) # node_numxrowsxcols
        #         for idx in range(self.rows*self.cols):
        #             (i,j) = self.idx2multi(idx)
        #             weighting_matrix[idx,:] = self.gaussian((i,j), sigma)
        #         if self.gpu_id >= 0:
        #             weighting_matrix = weighting_matrix.to(self.device)
        # compute the neighrbor weighting using pre-computed matrix
        weighting_matrix = self.get_weighting_matrix(sigma)  # node_numxrowsxcols

        # compute the update
        weighting_matrix = weighting_matrix.unsqueeze(0).expand(self.dim, self.node_num, self.rows,
                                                                self.cols)  # Cxnode_numxrowsxcols
        diff_masked_mean_matrix_view = diff_masked_mean.view(self.dim, self.node_num, self.rows, self.cols)
        delta = diff_masked_mean_matrix_view * weighting_matrix * learning_rate  # Cxnode_numxrowsxcols
        delta = delta.sum(dim=1)

        # apply the update
        node_matrix_view = self.node.view(self.dim, self.rows, self.cols)  # Cxrowsxcols
        node_matrix_view += delta

        # print(self.node)

    def optimize(self, x):
        self.node_init()
        for iter in range(int(self.max_iteration / 3)):
            self.batch_update(x, 0)
        for iter in range(self.max_iteration):
            self.batch_update(x, iter)


class BatchSOM():
    def __init__(self, rows=4, cols=4, dim=3, gpu_id=None, batch_size=10):
        self.rows = rows
        self.cols = cols
        self.dim = dim
        self.node_num = rows * cols

        self.sigma = 0.4
        self.learning_rate = 0.5
        self.max_iteration = 60

        self.gpu_id = gpu_id
        assert gpu_id >= 0
        self.device = torch.device("cuda:%d"%(gpu_id) if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # node: BxCx(rowsxcols), tensor
        self.node = torch.FloatTensor(self.batch_size, self.dim, self.rows * self.cols).zero_()
        self.node_idx_list = torch.from_numpy(np.arange(self.node_num).astype(np.int64))  # node_num LongTensor
        self.init_weighting_matrix = torch.FloatTensor(self.node_num, self.rows, self.cols)  # node_numxrowsxcols
        if self.gpu_id >= 0:
            self.node = self.node.to(self.device)
            self.node_idx_list = self.node_idx_list.to(self.device)

        # get initial weighting matrix
        self.get_init_weighting_matrix()

        # initialize the node by potential field
        pf = potential_field.PotentialField(self.node_num, self.dim)
        pf.optimize()
        self.node_init_value = torch.from_numpy(pf.node.transpose().astype(np.float32))

    def node_init(self, batch_size):
        self.batch_size = batch_size
        self.node.resize_(self.batch_size, self.dim, self.node_num)
        self.node.copy_(torch.unsqueeze(self.node_init_value, dim=0).expand_as(self.node))

    def gaussian(self, c, sigma):
        """Returns a Gaussian centered in c"""
        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(np.arange(self.rows) - c[0], 2) / d)
        ay = np.exp(-np.power(np.arange(self.cols) - c[1], 2) / d)
        return torch.from_numpy(np.outer(ax, ay).astype(np.float32))

    def get_init_weighting_matrix(self):
        '''
        get the initial weighting matrix, later the weighting matrix wil base on the init.
        '''
        for idx in range(self.rows * self.cols):
            (i, j) = self.idx2multi(idx)
            self.init_weighting_matrix[idx, :] = self.gaussian((i, j), self.sigma)
        if self.gpu_id >= 0:
            self.init_weighting_matrix = self.init_weighting_matrix.to(self.device)

    def get_weighting_matrix(self, sigma):
        scale = 1.0 / ((sigma / self.sigma) ** 2)
        weighting_matrix = torch.exp(torch.log(self.init_weighting_matrix) * scale)
        return weighting_matrix

    def idx2multi(self, i):
        return (i // self.cols, i % self.cols)

    def query_topk(self, x, k):
        '''
        :param x: input data BxCxN tensor
        :param k: topk
        :return: mask: Nxnode_num
        '''

        # expand as BxCxNxnode_num
        node = self.node.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2), self.rows * self.cols)
        x_expanded = x.unsqueeze(3).expand_as(node)

        # calcuate difference between x and each node
        diff = x_expanded - node  # BxCxNxnode_num
        diff_norm = (diff ** 2).sum(dim=1)  # BxNxnode_num

        # find the nearest neighbor
        _, min_idx = torch.topk(diff_norm, k=k, dim=2, largest=False, sorted=False)  # BxNxk
        min_idx_expanded = min_idx.unsqueeze(2).expand(min_idx.size()[0], min_idx.size()[1], self.rows * self.cols,
                                                       k)  # BxNxnode_numxk

        node_idx_list = self.node_idx_list.unsqueeze(0).unsqueeze(0).unsqueeze(3).expand_as(min_idx_expanded).long()  # BxNxnode_numxk
        mask = torch.eq(min_idx_expanded, node_idx_list).int()  # BxNxnode_numxk
        # mask = torch.sum(mask, dim=3)  # BxNxnode_num

        mask_list, min_idx_list = [], []
        for i in range(k):
            mask_list.append(mask[..., i])
            min_idx_list.append(min_idx[..., i])
        mask = torch.cat(tuple(mask_list), dim=1)  # BxkNxnode_num
        min_idx = torch.cat(tuple(min_idx_list), dim=1)  # BxkN
        mask_row_max, _ = torch.max(mask, dim=1)  # Bxnode_num, this indicates whether the node has nearby x

        return mask, mask_row_max, min_idx

    def query(self, x):
        '''
        :param x: input data CxN tensor
        :return: mask: Nxnode_num
        '''
        # expand as BxCxNxnode_num
        node = self.node.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2), self.rows * self.cols)
        x_expanded = x.unsqueeze(3).expand_as(node)

        # calcuate difference between x and each node
        diff = x_expanded - node  # BxCxNxnode_num
        diff_norm = (diff ** 2).sum(dim=1)  # BxNxnode_num

        # find the nearest neighbor
        _, min_idx = torch.min(diff_norm, dim=2)  # BxN
        min_idx_expanded = min_idx.unsqueeze(2).expand(min_idx.size()[0], min_idx.size()[1],
                                                       self.rows * self.cols)  # BxNxnode_num

        node_idx_list = self.node_idx_list.unsqueeze(0).unsqueeze(0).expand_as(min_idx_expanded).long()  # BxNxnode_num
        mask = torch.eq(min_idx_expanded, node_idx_list).float()  # BxNxnode_num
        mask_row_max, _ = torch.max(mask, dim=1)  # Bxnode_num, this indicates whether the node has nearby x

        return mask, mask_row_max

    def batch_update(self, x, learning_rate, sigma):
        # x is BxCxN tensor, C==self.dim, W=1
        assert (x.size()[1] == self.dim)
        assert (x.size()[0] == self.batch_size)

        # expand as BxCxNxnode_num
        node = self.node.unsqueeze(2).expand(x.size(0), x.size(1), x.size(2), self.rows * self.cols)
        x_expanded = x.unsqueeze(3).expand_as(node)

        # calcuate difference between x and each node
        diff = x_expanded - node  # BxCxNxnode_num
        diff_norm = (diff ** 2).sum(dim=1)  # BxNxnode_num

        # find the nearest neighbor
        _, min_idx = torch.min(diff_norm, dim=2)  # BxN
        min_idx_expanded = min_idx.unsqueeze(2).expand(min_idx.size()[0], min_idx.size()[1],
                                                       self.rows * self.cols)  # BxNxnode_num

        node_idx_list = self.node_idx_list.unsqueeze(0).unsqueeze(0).expand_as(min_idx_expanded).long()  # BxNxnode_num
        mask = torch.eq(min_idx_expanded, node_idx_list).float()  # BxNxnode_num
        mask_row_sum = torch.sum(mask, dim=1) + 0.00001  # Bxnode_num
        mask_row_max, _ = torch.max(mask, dim=1)  # Bxnode_num, this indicates whether the node has nearby x

        # calculate the mean x for each node
        x_expanded_masked = x_expanded * mask.unsqueeze(1).expand_as(x_expanded)  # BxCxNxnode_num
        x_expanded_masked_sum = torch.sum(x_expanded_masked, dim=2)  # BxCxnode_num
        x_expanded_mask_mean = x_expanded_masked_sum / mask_row_sum.unsqueeze(1).expand_as(
            x_expanded_masked_sum)  # Cxnode_num

        # each x_expanded_mask_mean (in total node_num vectors) will calculate its diff with all nodes
        # multiply the mask_row_max, so that the isolated node won't be pulled to the center
        x_expanded_mask_mean_expanded = x_expanded_mask_mean.unsqueeze(3).expand(self.batch_size, self.dim,
                                                                                 self.rows * self.cols,
                                                                                 self.rows * self.cols)  # BxCxnode_numxnode_num
        node_expanded_transposed = self.node.unsqueeze(2).expand_as(x_expanded_mask_mean_expanded)  # .transpose(1,2)
        diff_masked_mean = x_expanded_mask_mean_expanded - node_expanded_transposed  # BxCxnode_numxnode_num
        diff_masked_mean = diff_masked_mean * mask_row_max.unsqueeze(2).unsqueeze(1).expand_as(diff_masked_mean)

        # compute the neighrbor weighting using pre-computed matrix
        weighting_matrix = self.get_weighting_matrix(sigma)

        # expand weighting_matrix to be batch, Bxnode_numxrowsxcols
        weighting_matrix = weighting_matrix.unsqueeze(0).expand(self.batch_size, self.rows * self.cols, self.rows,
                                                                self.cols)

        # compute the update
        weighting_matrix = weighting_matrix.unsqueeze(1).expand(self.batch_size, self.dim, self.rows * self.cols,
                                                                self.rows, self.cols)  # BxCxnode_numxrowsxcols
        diff_masked_mean_matrix_view = diff_masked_mean.view(self.batch_size, self.dim, self.rows * self.cols,
                                                             self.rows, self.cols)
        delta = diff_masked_mean_matrix_view * weighting_matrix * learning_rate  # BxCxnode_numxrowsxcols
        delta = delta.sum(dim=2)

        # apply the update
        node_matrix_view = self.node.view(self.batch_size, self.dim, self.rows, self.cols)  # BxCxrowsxcols
        node_matrix_view += delta

        # print(self.node)
        # print(delta.max())

    def optimize(self, x):
        self.node_init(x.size()[0])
        for iter in range(int(self.max_iteration / 3)):
            # get learning_rate and sigma
            learning_rate = self.learning_rate
            sigma = self.sigma
            self.batch_update(x, learning_rate, sigma)
        for iter in range(self.max_iteration):
            # get learning_rate and sigma
            learning_rate = self.learning_rate / (1 + 2*iter / self.max_iteration)
            sigma = self.sigma / (1 + 2*iter / self.max_iteration)
            self.batch_update(x, learning_rate, sigma)


