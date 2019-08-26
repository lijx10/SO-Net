import torch.utils.data as data

import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py
import faiss

from .augmentation import *


# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def make_dataset_modelnet40_10k(root, mode, opt):
    dataset = []
    rows = round(math.sqrt(opt.node_num))
    cols = rows

    f = open(os.path.join(root, 'modelnet%d_shape_names.txt' % opt.classes))
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join(root, 'modelnet%d_train.txt' % opt.classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test' == mode:
        f = open(os.path.join(root, 'modelnet%d_test.txt' % opt.classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    for i, name in enumerate(lines):
        # locate the folder name
        folder = name[0:-5]
        file_name = name

        # get the label
        label = shape_list.index(folder)

        # som node locations
        som_nodes_folder = '%dx%d_som_nodes' % (rows, cols)

        item = (os.path.join(root, folder, file_name + '.npy'),
                label,
                os.path.join(root, som_nodes_folder, folder, file_name + '.npy'))
        dataset.append(item)

    return dataset


def make_dataset_shrec2016(root, mode, opt):
    rows = round(math.sqrt(opt.node_num))
    cols = rows
    dataset = []

    # load category txt
    f = open(os.path.join(root, 'category.txt'), 'r')
    category_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train'==mode:
        f = open(os.path.join(root, 'train.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'val'==mode:
        f = open(os.path.join(root, 'val.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test'==mode:
        f = open(os.path.join(root, 'test.txt'), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    if 'train' == mode or 'val' == mode:
        for line in lines:
            line_split = [x.strip() for x in line.split(',')]
            name, category = line_split[0], line_split[1]

            npz_file = os.path.join(root, '%dx%d'%(rows,cols), mode, 'model_'+name+'.npz')
            try:
                category = category_list.index(category)
            except ValueError:
                continue

            item = (npz_file, category)
            dataset.append(item)
    elif 'test' == mode:
        for line in lines:
            name, category = line, int(line) % 55
            npz_file = os.path.join(root, '%dx%d'%(rows,cols), mode, 'model_'+name+'.npz')

            item = (npz_file, category)
            dataset.append(item)

    return dataset


class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''

        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class ModelNet_Shrec_Loader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(ModelNet_Shrec_Loader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        if self.opt.dataset == 'modelnet':
            self.dataset = make_dataset_modelnet40_10k(self.root, mode, opt)
        elif self.opt.dataset == 'shrec':
            self.dataset = make_dataset_shrec2016(self.root, mode, opt)
        else:
            raise Exception('Dataset incorrect.')

        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.opt.dataset == 'modelnet':
            pc_np_file, class_id, som_node_np_file = self.dataset[index]

            data = np.load(pc_np_file)
            data = data[np.random.choice(data.shape[0], self.opt.input_pc_num, replace=False), :]

            pc_np = data[:, 0:3]  # Nx3
            surface_normal_np = data[:, 3:6]  # Nx3
            som_node_np = np.load(som_node_np_file)  # node_numx3
        elif self.opt.dataset == 'shrec':
            npz_file, class_id = self.dataset[index]
            data = np.load(npz_file)

            pc_np = data['pc']
            surface_normal_np = data['sn']
            som_node_np = data['som_node']

            # random choice
            choice_idx = np.random.choice(pc_np.shape[0], self.opt.input_pc_num, replace=False)
            pc_np = pc_np[choice_idx, :]
            surface_normal_np = surface_normal_np[choice_idx, :]
        else:
            raise Exception('Dataset incorrect.')

        # augmentation
        if self.mode == 'train':
            # rotate by 0/90/180/270 degree over z axis
            # pc_np = rotate_point_cloud_90(pc_np)
            # som_node_np = rotate_point_cloud_90(som_node_np)

            # rotation perturbation, pc and som should follow the same rotation, surface normal rotation is unclear
            if self.opt.rot_horizontal:
                pc_np, surface_normal_np, som_node_np = rotate_point_cloud_with_normal_som(pc_np, surface_normal_np, som_node_np)
            if self.opt.rot_perturbation:
                pc_np, surface_normal_np, som_node_np = rotate_perturbation_point_cloud_with_normal_som(pc_np, surface_normal_np, som_node_np)

            # random jittering
            pc_np = jitter_point_cloud(pc_np)
            surface_normal_np = jitter_point_cloud(surface_normal_np)
            som_node_np = jitter_point_cloud(som_node_np, sigma=0.04, clip=0.1)

            # random scale
            scale = np.random.uniform(low=0.8, high=1.2)
            pc_np = pc_np * scale
            som_node_np = som_node_np * scale
            surface_normal_np = surface_normal_np * scale

            # random shift
            if self.opt.translation_perturbation:
                shift = np.random.uniform(-0.1, 0.1, (1,3))
                pc_np += shift
                som_node_np += shift

        # convert to tensor
        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN

        # surface normal
        surface_normal = torch.from_numpy(surface_normal_np.transpose().astype(np.float32))  # 3xN

        # som
        som_node = torch.from_numpy(som_node_np.transpose().astype(np.float32))  # 3xnode_num

        # kNN search: som -> som
        if self.opt.som_k >= 2:
            D, I = self.knn_builder.self_build_search(som_node_np)
            som_knn_I = torch.from_numpy(I.astype(np.int64))  # node_num x som_k
        else:
            som_knn_I = torch.from_numpy(np.arange(start=0, stop=self.opt.node_num, dtype=np.int64).reshape((self.opt.node_num, 1)))  # node_num x 1

        # print(som_node_np)
        # print(D)
        # print(I)
        # assert False

        if self.opt.dataset == 'shrec':
            return pc, surface_normal, class_id, som_node, som_knn_I, index
        else:
            return pc, surface_normal, class_id, som_node, som_knn_I


if __name__=="__main__":
    # dataset = make_dataset_modelnet40('/ssd/dataset/modelnet40_ply_hdf5_2048/', True)
    # print(len(dataset))
    # print(dataset[0])


    class VirtualOpt():
        def __init__(self):
            self.load_all_data = False
            self.input_pc_num = 5000
            self.batch_size = 8
            self.dataset = '10k'
            self.node_num = 64
            self.classes = 10
            self.som_k = 9
    opt = VirtualOpt()
    trainset = ModelNet_Shrec_Loader('/ssd/dataset/modelnet40-normal_numpy/', 'train', opt)
    print('---')
    print(len(trainset))
    print(trainset[0])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
