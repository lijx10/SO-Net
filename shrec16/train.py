import time
import copy
import numpy as np
import math

from shrec16.options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.classifier import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from util.visualizer import Visualizer


if __name__=='__main__':
    trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    print('#training point clouds = %d' % len(trainset))

    testset = ModelNet_Shrec_Loader(opt.dataroot, 'val', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)

    # create model, optionally load pre-trained model
    model = Model(opt)
    if opt.pretrain is not None:
        model.encoder.load_state_dict(torch.load(opt.pretrain))
    ############################# automation for ModelNet10 / 40 configuration ####################
    # if opt.classes == 10:
    #     opt.lr = opt.lr * 0.1
    #     opt.dropout = opt.dropout + 0.1
    ############################# automation for ModelNet10 / 40 configuration ####################

    visualizer = Visualizer(opt)

    best_accuracy = 0
    for epoch in range(201):

        epoch_iter = 0
        for i, data in enumerate(trainloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size

            input_pc, input_sn, input_label, input_node, input_node_knn_I = data
            model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)

            model.optimize(epoch=epoch)

            if i % 600 == 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / opt.batch_size

                errors = model.get_current_errors()

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # print(model.autoencoder.encoder.feature)
                # visuals = model.get_current_visuals()
                # visualizer.display_current_results(visuals, epoch, i)

        # test network
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model.test_loss.data.zero_()
            model.test_accuracy.data.zero_()
            for i, data in enumerate(testloader):
                input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                model.test_model()

                batch_amount += input_label.size()[0]

                # # accumulate loss
                model.test_loss += model.loss.detach() * input_label.size()[0]

                # # accumulate accuracy
                _, predicted_idx = torch.max(model.score.data, dim=1, keepdim=False)
                correct_mask = torch.eq(predicted_idx, model.input_label).float()
                test_accuracy = torch.mean(correct_mask).cpu()
                model.test_accuracy += test_accuracy * input_label.size()[0]

            model.test_loss /= batch_amount
            model.test_accuracy /= batch_amount
            if model.test_accuracy.item() > best_accuracy:
                best_accuracy = model.test_accuracy.item()
            print('Tested network. So far best: %f' % (best_accuracy) )

            # save network
            saving_acc_threshold = 0.0
            if model.test_accuracy.item() > saving_acc_threshold:
                print("Saving network...")
                model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_accuracy.item()), opt.gpu_id)
                model.save_network(model.classifier, 'classifier', '%d_%f' % (epoch, model.test_accuracy.item()), opt.gpu_id)

        # learning rate decay
        if opt.classes == 10:
            lr_decay_step = 40
        else:
            lr_decay_step = 20
        if epoch%lr_decay_step==0 and epoch>0:
            model.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (opt.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % opt.bn_momentum_decay_step == 0):
            current_bn_momentum = opt.bn_momentum * (
            opt.bn_momentum_decay ** (next_epoch // opt.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)

        # save network
        # if epoch%20==0 and epoch>0:
        #     print("Saving network...")
        #     model.save_network(model.classifier, 'cls', '%d' % epoch, opt.gpu_id)





