import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.multiprocessing as mp
import threading
import ctypes
import numba
from numba import vectorize, cuda


def zero_edge(x, padding):
    '''

    :param x: BxCxHxW Variable/Tensor
    :param padding: int
    :return:
    '''
    if (padding is None) or (padding <= 0):
        return x

    H = x.size()[2]
    W = x.size()[3]

    H_padding_idx = list(range(0, padding))
    H_padding_idx_tail = list(range(H-padding, H))
    H_padding_idx.extend(H_padding_idx_tail)

    W_padding_idx = list(range(0, padding))
    W_padding_idx_tail = list(range(W-padding, W))
    W_padding_idx.extend(W_padding_idx_tail)

    x[:, :, H_padding_idx, :] = 0
    x[:, :, :, W_padding_idx] = 0

    return x


# ================================== mask max ======================================
class MaskedMaxThread:
    def __init__(self, thread_num):
        self.batch_each_worker = 2
        self.thread_num = thread_num

    def worker(self, i):
        batch_size = self.data.size()[0]
        node_num = self.mask.size()[3]
        for i in range(i * self.batch_each_worker, (i + 1) * self.batch_each_worker):
            if i>=batch_size:
                break
            # iterate over the clusters
            for j in range(node_num):
                indexes = torch.nonzero(self.mask[i, 0, :, j])
                if len(indexes.size()) > 0:
                    selected_rows = self.data[i].index_select(dim=1, index=indexes[:, 0])  # Cxk
                    _, idx = selected_rows.max(dim=1)
                    self.gather_idx[i, :, j] = indexes[:, 0][idx]

    def compute(self, data, mask):
        '''
        :param data: BxCxN tensor in CPU
        :param mask: Bx1xNxnode_num tensor in CPU
        :return gather_idx: BxCxnode_num tensor in CPU
        '''
        batch_size = data.size()[0]
        self.batch_each_worker = math.ceil(batch_size / self.thread_num)

        self.data = data.cpu()
        self.mask = mask.cpu()
        self.gather_idx = torch.LongTensor(batch_size, data.size()[1], mask.size()[3]).zero_()

        threads = []
        for i in range(self.thread_num):
            t = threading.Thread(target=self.worker, args=(i, ))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        return self.gather_idx


def get_devicendarray_float32(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('float32'),
                                                  gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)


def get_devicendarray_int32(t):
    assert t.type() == 'torch.cuda.IntTensor'
    ctx = cuda.cudadrv.driver.driver.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('int32'),
                                                  gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)


@cuda.jit('float32[:,:,:], int32[:,:], int32[:,:,:], float32[:,:,:]')
def indexed_max(data, index, max_idx, max_val):
    '''
    :param data: BxCxN
    :param index: BxN
    :param max_idx: BxCxK
    :param max_val: BxCxK
    :return:
    '''

    b = cuda.blockIdx.x
    c = cuda.threadIdx.x
    for n in range(index.shape[1]):
        k = index[b, n]
        if data[b, c, n] > max_val[b, c, k]:
            max_val[b, c, k] = data[b, c, n]
            max_idx[b, c, k] = n
        # cuda.syncthreads()


class MaskedMax:
    def __init__(self, som_node_number, gpu_id):
        self.cpu_masked_max = MaskedMaxThread(thread_num=8)
        self.M = som_node_number

        # initialization for indexed_max
        self.device = torch.device("cuda:%d"%gpu_id)
        self.max_idx = torch.IntTensor(8, 384, self.M).zero_().to(self.device)
        self.max_val = torch.FloatTensor(8, 384, self.M).fill_(-100).to(self.device)

    def compute(self, data, min_idx, mask):
        '''
        som node number M.
        :param data: BxCxkN, FloatTensor
        :param min_idx: BxkN, LongTensor containing indices of [0,M-1]
        :param mask: Bx1xNxM, ByteTensor
        :return: gather_index, BxCxM LongTensor
        '''

        # ============= cuda ===================
        B = data.size()[0]
        C = data.size()[1]

        data_cuda = get_devicendarray_float32(data.data)
        index_cuda = get_devicendarray_int32(min_idx.int().data)
        max_idx_cuda = get_devicendarray_int32(self.max_idx.resize_((B, C, self.M)).zero_().data)
        max_val_cuda = get_devicendarray_float32(self.max_val.resize_((B, C, self.M)).fill_(-1000).data)

        indexed_max[B, C](data_cuda, index_cuda, max_idx_cuda, max_val_cuda)
        gather_index = self.max_idx.to(self.device, torch.int64)
        # ============= cuda ===================

        # ============= cpu ===================
        # gather_index_cpu = self.cpu_masked_max.compute(data.cpu(), mask.cpu()).to(self.device)
        # ============= cpu ===================

        # debug
        # print(self.max_idx.device)
        # print(gather_index.device)
        # print(torch.min(gather_index - gather_index_cpu))
        # print(torch.max(gather_index - gather_index_cpu))

        return gather_index
# ================================== mask max ======================================


# ================================== coordinate - kernel center Bx3xHxW ======================================
@cuda.jit('float32[:,:,:,:], float32[:,:,:,:], int32, int32')
def unroll_decenter_conv2d_same(data, result, kernel_size, padding):
    '''
    square kernel, padding (p,p). p = floor(K/2)
    Output: Bx3xKHxKW
    :param data: Bx3x(H+2p)x(W+2p), contains coordinates, after padding
    :param kernel_size: K
    '''
    b = cuda.blockIdx.x
    c = cuda.blockIdx.y
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    for kx in range(kernel_size):
        for ky in range(kernel_size):
            result[b, c, kernel_size*i+kx, kernel_size*j+ky] = \
                data[b, c, i+kx, j+ky] - data[b, c, i+padding, j+padding]

    # result[b, c, kernel_size*i:kernel_size*(i+1), kernel_size*j:kernel_size*(j+1)] = \
    #     data[b, c, i:i+kernel_size, j:j+kernel_size] - data[b, c, i+padding, j+padding]


@cuda.jit('float32[:,:,:,:], float32[:,:,:,:], int32, int32')
def unroll_decenter_conv2d_valid(data, result, kernel_size, half_kernel_size):
    '''

    :param data: Bx3xHxW
    :param result:
    :param kernel_size:
    :return:
    '''
    b = cuda.blockIdx.x
    c = cuda.blockIdx.y
    i = cuda.threadIdx.x  # 0 -> H-2*half_kernel_size
    j = cuda.threadIdx.y  # 0 -> W-2*half_kernel_size

    for kx in range(kernel_size):
        for ky in range(kernel_size):
            result[b, c, kernel_size * i + kx, kernel_size * j + ky] = \
                data[b, c, i + kx, j + ky] - data[b, c, i + half_kernel_size, j + half_kernel_size]


def unroll_decenter(coordinate, kernel_size, padding, gpu_id):
    '''
    if there is padding, it is "SAME" convolution, result is KHxKW
    if no padding, it is "VALID" convolution, result is K(H-K/2) x K(W-K/2)
    :param coordinate: Bx3xHxW Variable
    :param kernel_size:
    :param padding:
    :return:
    '''
    B = coordinate.size()[0]
    C = 3
    H = coordinate.size()[2]
    W = coordinate.size()[3]

    device = torch.device("cuda:%d"%gpu_id)

    if (padding is not None) and (padding >= 1):
        # ensure SAME style
        assert math.floor(kernel_size/2) == padding

        coordinate_padded = F.pad(coordinate, (padding, padding, padding, padding), mode='constant', value=0)
        coordinate_padded_cuda = get_devicendarray_float32(coordinate_padded.data)

        result = torch.FloatTensor(B, C, kernel_size * H, kernel_size * W).to(device).zero_()
        result_cuda = get_devicendarray_float32(result.data)

        unroll_decenter_conv2d_same[(B, C), (H, W)](coordinate_padded_cuda, result_cuda, kernel_size, padding)

        # maintain 0 on the padding area
        # result[:, :, 0:padding, :] = 0
        # result[:, :, :, 0:padding] = 0
        # result[:, :, kernel_size*H-padding:kernel_size*H, :] = 0
        # result[:, :, :, kernel_size*W-padding:kernel_size*W] = 0
        zero_edge(result, padding=padding)
    else:
        half_kernel_size = math.floor(kernel_size / 2)

        coordinate_cuda = get_devicendarray_float32(coordinate.data)

        result = torch.FloatTensor(B, C, kernel_size * (H-2*half_kernel_size), kernel_size * (W-2*half_kernel_size)).to(device).zero_()
        result_cuda = get_devicendarray_float32(result.data)

        unroll_decenter_conv2d_valid[(B, C), (H-2*half_kernel_size, W-2*half_kernel_size)](coordinate_cuda, result_cuda, kernel_size, half_kernel_size)

    # debug
    # print(coordinate[0, 0])
    # print(result[0, 0])
    # assert False

    return result
# ================================== coordinate - kernel center Bx3xHxW ======================================


# ================================== coordinate - average Bx3xHxW ======================================
@cuda.jit('float32[:,:,:,:], float32[:,:,:,:], int32, int32, float32[:, :, :, :]')
def unroll_deaverage_conv2d_same(data, result, kernel_size, padding, avg_matrix):
    '''
    square kernel, padding (p,p). p = floor(K/2)
    Output: Bx3xKHxKW
    :param data: Bx3x(H+2p)x(W+2p), contains coordinates, after padding
    :param avg_matrix: Bx3x(H+2p)x(W+2p), to save average coordinates, after padding
    :param kernel_size: K
    '''
    b = cuda.blockIdx.x
    c = cuda.blockIdx.y
    i = cuda.threadIdx.x  # i anchor index rows
    j = cuda.threadIdx.y  # j anchor index cols

    # calcuate average
    H_valid_begin = padding  # e.g. som_node is 8x8, padding=1, then blockDim.x=8, H_valid_begin=1
    H_valid_end = cuda.blockDim.x+padding  # e.g. som_node is 8x8, padding=1, then blockDim.y=8, H_valid_end=9
    W_valid_begin = padding
    W_valid_end = cuda.blockDim.y+padding
    counter = 0
    for kx in range(kernel_size):
        for ky in range(kernel_size):
            if i+kx>=H_valid_begin and i+kx<H_valid_end and j+ky>=W_valid_begin and j+ky<W_valid_end:
                counter += 1
                avg_matrix[b, c, i + padding, j + padding] += data[b, c, i + kx, j + ky]
    avg_matrix[b, c, i + padding, j + padding] /= counter

    for kx in range(kernel_size):
        for ky in range(kernel_size):
            result[b, c, kernel_size * i + kx, kernel_size * j + ky] = \
                data[b, c, i + kx, j + ky] - avg_matrix[b, c, i + padding, j + padding]


def unroll_average(coordinate, kernel_size, padding, gpu_id):
    '''
    if there is padding, it is "SAME" convolution, result is KHxKW
    if no padding, it is "VALID" convolution, result is K(H-K/2) x K(W-K/2)
    :param coordinate: Bx3xHxW Variable
    :param kernel_size:
    :param padding:
    :return:
    '''
    B = coordinate.size()[0]
    C = 3
    H = coordinate.size()[2]
    W = coordinate.size()[3]

    device = torch.device("cuda:%d" % gpu_id)

    if (padding is not None) and (padding >= 1):
        # ensure SAME style
        assert math.floor(kernel_size / 2) == padding

        coordinate_padded = F.pad(coordinate, (padding, padding, padding, padding), mode='constant', value=0)
        coordinate_padded_cuda = get_devicendarray_float32(coordinate_padded.data)

        result = torch.FloatTensor(B, C, kernel_size * H, kernel_size * W).to(device).zero_()
        result_cuda = get_devicendarray_float32(result.data)

        avg_matrix_padded = torch.FloatTensor(coordinate_padded.size()).to(device).zero_()
        avg_matrix_padded_cuda = get_devicendarray_float32(avg_matrix_padded.data)

        unroll_deaverage_conv2d_same[(B, C), (H, W)](coordinate_padded_cuda, result_cuda, kernel_size, padding, avg_matrix_padded_cuda)

        # maintain 0 on the padding area
        zero_edge(result, padding=padding)

        # get the unpadded avg_matrix
        avg_matrix = avg_matrix_padded[:, :, padding:avg_matrix_padded.size()[2]-padding, padding:avg_matrix_padded.size()[3]-padding]
    else:
        assert False
        pass
        # half_kernel_size = math.floor(kernel_size / 2)
        #
        # coordinate_cuda = get_devicendarray_float32(coordinate.data)
        #
        # result = torch.FloatTensor(B, C, kernel_size * (H - 2 * half_kernel_size),
        #                            kernel_size * (W - 2 * half_kernel_size)).to(device).zero_()
        # result_cuda = get_devicendarray_float32(result.data)
        #
        # unroll_decenter_conv2d_valid[(B, C), (H - 2 * half_kernel_size, W - 2 * half_kernel_size)](
        #     coordinate_cuda, result_cuda, kernel_size, half_kernel_size)

    # debug
    # print(coordinate[0, 0])
    # print(avg_matrix[0, 0])
    # print(result[0, 0])
    # assert False

    return result, avg_matrix
# ================================== coordinate - average Bx3xHxW ======================================


# ================================== feature Bx(3+C)xHxW ======================================
class UnrollFeature:
    def __init__(self, H, W, kernel_size, padding=0, gpu_id=0):
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        self.half_kernel_size = math.floor(kernel_size/2)
        self.padding = padding

        self.zero_pad = None
        if (self.padding is not None) and (self.padding >= 1):
            self.zero_pad = nn.ZeroPad2d(padding)

        # cuda Variable of dimension 1
        # SAME: input: (H+2*padding)x(W+2*padding)    output: KHxKW
        # VALID: input: HxW                           output: K(H-2*half_kernel)xK(W-2*half_kernel)
        self.gather_index = self.get_gather_index()

        self.device = torch.device("cuda:%d"%gpu_id)

    def get_gather_index(self):
        '''
        operates in CPU, convert to cuda at return
        :return:
        '''
        if (self.padding is not None) and (self.padding >= 1):
            # ensure SAME style
            assert self.half_kernel_size == self.padding

            H_padded = self.H + 2*self.padding
            W_padded = self.W + 2*self.padding

            padded_feature_index = torch.arange(0, round(H_padded*W_padded)).view(H_padded, W_padded).long()
            unrolled_feature_index = torch.LongTensor(round(self.kernel_size*self.H), round(self.kernel_size*self.W))
            # unroll
            for i in range(self.H):
                for j in range(self.W):
                    for kx in range(self.kernel_size):
                        for ky in range(self.kernel_size):
                            unrolled_feature_index[self.kernel_size*i+kx, self.kernel_size*j+ky] = padded_feature_index[i+kx, j+ky]

            # debug
            # print(padded_feature_index)
            # print(unrolled_feature_index)

            # input: (H+2*padding)x(W+2*padding)    output: KHxKW
            gather_index = unrolled_feature_index.view(round(self.kernel_size*self.H) * round(self.kernel_size*self.W))

        else:
            H_cropped = self.H - 2*self.half_kernel_size
            W_cropped = self.W - 2*self.half_kernel_size

            feature_index = torch.arange(0, self.H*self.W).view(self.H, self.W).long()
            unrolled_feature_index = torch.LongTensor(round(self.kernel_size*H_cropped), round(self.kernel_size*W_cropped))
            # unroll
            for i in range(H_cropped):
                for j in range(W_cropped):
                    for kx in range(self.kernel_size):
                        for ky in range(self.kernel_size):
                            unrolled_feature_index[self.kernel_size*i+kx, self.kernel_size*j+ky] = feature_index[i+kx, j+ky]

            # debug
            # print(feature_index)
            # print(unrolled_feature_index)

            # input: HxW      output: K(H-2*half_kernel)xK(W-2*half_kernel)
            gather_index = unrolled_feature_index.view(self.kernel_size*round(self.H-2*self.half_kernel_size) * self.kernel_size*round(self.W-2*self.half_kernel_size))

        return gather_index.to(self.device)

    def unroll(self, x):
        '''

        :param x: BxCxHxW Variable
        :return:
        '''
        B = x.size()[0]
        C = x.size()[1]
        H = x.size()[2]
        W = x.size()[3]
        assert H==self.H
        assert W==self.W

        # get gather_index, expand it to the correct dimension
        gather_index = self.get_gather_index()

        if (self.padding is not None) and (self.padding>=1):
            # pad the input feature
            H_unrolled = self.kernel_size*H
            W_unrolled = self.kernel_size*W
            x_padded = self.zero_pad(x).view(B, C, -1)

            # expand gather_index to the correct dimension: BxCx(kH*kW)
            gather_index = gather_index.unsqueeze(0).unsqueeze(0).expand(B, C, H_unrolled*W_unrolled)
            x_unrolled = x_padded.gather(dim=2, index=gather_index).view(B, C, H_unrolled, W_unrolled)
        else:
            H_unrolled = self.kernel_size * (H - 2 * self.half_kernel_size)
            W_unrolled = self.kernel_size * (W - 2 * self.half_kernel_size)
            x = x.contiguous().view(B, C, -1)

            # expand gather_index to the correct dimension: BxCx{K(H-2*half_kernel) * K(W-2*half_kernel)}
            gather_index = gather_index.unsqueeze(0).unsqueeze(0).expand(B, C, H_unrolled*W_unrolled)
            x_unrolled = x.gather(dim=2, index=gather_index).view(B, C, H_unrolled, W_unrolled)

        return x_unrolled

# ================================== feature Bx(3+C)xHxW ======================================


# ================================== get the k nearest neighbors of the SOM nodes / features ======================================
@cuda.jit('float32[:, :, :], int32[:, :, :], float32[:, :, :, :]')
def knn_gather(som_node, som_node_knn_I, som_node_neighbors):
    '''

    :param som_node: Bx3xN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: Bx3xNxK
    :return:
    '''

    b = cuda.blockIdx.x
    c = cuda.blockIdx.y
    n = cuda.threadIdx.x  # n \in [0, N-1]
    k = cuda.threadIdx.y  # k

    som_node_neighbors[b, c, n, k] = som_node[b, c, som_node_knn_I[b, n, k]]


def knn_gather_wrapper(som_node, som_node_knn_I, gpu_id):
    '''

    :param som_node: Bx3xN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: Bx3xNxK
    :return:
    '''
    B = som_node.size()[0]
    C = som_node.size()[1]
    N = som_node.size()[2]
    K = som_node_knn_I.size()[2]
    assert C==3

    som_node_neighbors = torch.FloatTensor(B, C, N, K).to(torch.device("cuda:%d"%gpu_id)).zero_()

    som_node_cuda = get_devicendarray_float32(som_node.data)
    som_node_knn_I_cuda = get_devicendarray_int32(som_node_knn_I.int().data)
    som_node_neighbors_cuda = get_devicendarray_float32(som_node_neighbors.data)

    knn_gather[(B, C), (N, K)](som_node_cuda, som_node_knn_I_cuda, som_node_neighbors_cuda)

    return som_node_neighbors


def knn_gather_by_indexing(som_node, som_node_knn_I):
    '''

    :param som_node: BxCxN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: BxCxNxK
    :return:
    '''
    B = som_node.size()[0]
    C = som_node.size()[1]
    N = som_node.size()[2]
    K = som_node_knn_I.size()[2]

    som_node_knn_I = som_node_knn_I.unsqueeze(1).expand(B, C, N, K).contiguous().view(B, C, N*K)
    som_node_neighbors = torch.gather(som_node, dim=2, index=som_node_knn_I).view(B, C, N, K)

    return som_node_neighbors

# ================================== get the k nearest neighbors of the SOM nodes / features ======================================


if __name__=='__main__':
    # unroll_feature = UnrollFeature(8, 8, 3, 1)
    # x = torch.arange(1, 65).cuda().view(8, 8)
    # x = x.unsqueeze(0).unsqueeze(0).expand(8, 384, 8, 8).detach()
    #
    # x_unrolled = unroll_feature.unroll(x)
    #
    # print(x[0,0,:,:])
    # print(x_unrolled[0,0,:,:])

    coordinate = torch.arange(1, 65).cuda().view(8, 8)
    coordinate = coordinate.unsqueeze(0).unsqueeze(0).expand(8, 384, 8, 8).detach()

    coordinate_unrolled = unroll_average(coordinate, 3, 1)
