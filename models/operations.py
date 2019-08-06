import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp
import threading
import ctypes


# generalized batch size
CUDA_SHARED_MEM_DIM_X = 24
# size of SOM
CUDA_SHARED_MEM_DIM_Y = 512


def knn_gather_wrapper(som_node, som_node_knn_I):
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
    assert C==3 or C==2

    som_node_neighbors = knn_gather_by_indexing(som_node, som_node_knn_I)

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
