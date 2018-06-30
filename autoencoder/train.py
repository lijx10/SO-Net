import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.autoencoder import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from data.shapenet_loader import ShapeNetLoader
from util.visualizer import Visualizer


if __name__=='__main__':
    if opt.dataset=='modelnet' or opt.dataset=='shrec':
        trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
        dataset_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    elif opt.dataset=='shapenet':
        trainset = ShapeNetLoader(opt.dataroot, 'train', opt)
        dataset_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
        print('#training point clouds = %d' % len(trainset))

        tesetset = ShapeNetLoader(opt.dataroot, 'test', opt)
        testloader = torch.utils.data.DataLoader(tesetset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    else:
        raise Exception('Dataset error.')

    model = Model(opt)

    visualizer = Visualizer(opt)

    best_loss = 99
    for epoch in range(601):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size

            if opt.dataset=='modelnet' or opt.dataset=='shrec':
                input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
            elif opt.dataset=='shapenet':
                input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)

            model.optimize()

            if i % 100 == 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt.batch_size

                errors = model.get_current_errors()

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # print(model.autoencoder.encoder.feature)
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, i)

        # test network
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model.test_loss.data.zero_()
            for i, data in enumerate(testloader):
                if opt.dataset == 'modelnet' or opt.dataset=='shrec':
                    input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                    model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                elif opt.dataset == 'shapenet':
                    input_pc, input_sn, input_label, input_seg, input_node, input_node_knn_I = data
                    model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                model.test_model()

                batch_amount += input_label.size()[0]

                # # accumulate loss
                model.test_loss += model.loss_chamfer.detach() * input_label.size()[0]

            model.test_loss /= batch_amount
            if model.test_loss.item() < best_loss:
                best_loss = model.test_loss.item()
            print('Tested network. So far lowest loss: %f' % best_loss )

        # learning rate decay
        if epoch%20==0 and epoch>0:
            model.update_learning_rate(0.5)

        # save network
        if epoch%1==0 and epoch>0:
            print("Saving network...")
            model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)
            model.save_network(model.decoder, 'decoder', '%d_%f' % (epoch, model.test_loss.item()), opt.gpu_id)





