import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from collections import OrderedDict
import os
import random

from . import networks
from . import losses

class Model():
    def __init__(self, opt):
        self.opt = opt

        self.old_lr = opt.lr

        self.encoder = networks.Encoder(opt)
        self.decoder = networks.Decoder(opt)
        self.chamfer_criteria = losses.ChamferLoss(opt)
        if self.opt.gpu_ids:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.chamfer_criteria = self.chamfer_criteria.cuda()

        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(),
                                          lr=self.opt.lr,
                                          betas=(0.9, 0.999))
        self.optimizer_decoder = torch.optim.Adam(self.decoder.parameters(),
                                                  lr=self.opt.lr,
                                                  betas=(0.9, 0.999))

        # place holder for GPU tensors
        self.input_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.input_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.input_label = torch.LongTensor(self.opt.batch_size).fill_(1)
        self.input_node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.input_node_knn_I = torch.LongTensor(self.opt.batch_size, self.opt.node_num, self.opt.som_k)

        # record the test loss and accuracy
        self.test_loss = Variable(torch.FloatTensor([0]), requires_grad=False)

        if self.opt.gpu_ids:
            self.input_pc = self.input_pc.cuda()
            self.input_sn = self.input_sn.cuda()
            self.input_label = self.input_label.cuda()
            self.input_node = self.input_node.cuda()
            self.input_node_knn_I = self.input_node_knn_I.cuda()
            self.test_loss = self.test_loss.cuda()

    def set_input(self, input_pc, input_sn, input_label, input_node, input_node_knn_I):
        self.input_pc.resize_(input_pc.size()).copy_(input_pc)
        self.input_sn.resize_(input_sn.size()).copy_(input_sn)
        self.input_label.resize_(input_label.size()).copy_(input_label)
        self.input_node.resize_(input_node.size()).copy_(input_node)
        self.input_node_knn_I.resize_(input_node_knn_I.size()).copy_(input_node_knn_I)
        self.pc = Variable(self.input_pc, requires_grad=False)
        self.sn = Variable(self.input_sn, requires_grad=False)
        self.label = Variable(self.input_label, requires_grad=False)

    def forward(self, is_train=False, epoch=None):
        self.feature = self.encoder(self.pc, self.sn, self.input_node, self.input_node_knn_I, is_train, epoch)  # Bx1024
        self.predicted_pc = self.decoder(self.feature)

    def optimize(self, epoch=None):
        # random point dropout
        if self.opt.random_pc_dropout_lower_limit < 0.99:
            dropout_keep_ratio = random.uniform(self.opt.random_pc_dropout_lower_limit, 1.0)
            resulting_pc_num = round(dropout_keep_ratio * self.opt.input_pc_num)
            chosen_indices = np.random.choice(self.opt.input_pc_num, resulting_pc_num, replace=False)
            chosen_indices_tensor = torch.from_numpy(chosen_indices).cuda()
            self.pc = torch.index_select(self.pc, dim=2, index=Variable(chosen_indices_tensor, requires_grad=False))
            self.sn = torch.index_select(self.sn, dim=2, index=Variable(chosen_indices_tensor, requires_grad=False))

        self.encoder.train()
        self.decoder.train()
        self.forward(is_train=True, epoch=epoch)

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        if self.opt.output_conv_pc_num > 0:
            # loss for second last conv pyramid # 32x32
            self.loss_chamfer_conv5 = self.chamfer_criteria(self.decoder.conv_pc5, self.pc)

            # loss for third last conv pyramid # 16x16
            self.loss_chamfer_conv4 = self.chamfer_criteria(self.decoder.conv_pc4, self.pc)

        # loss for the last pyramid, i.e., the final pc
        self.loss_chamfer = self.chamfer_criteria(self.predicted_pc, self.pc)

        if self.opt.output_conv_pc_num == 1024:
            self.loss = self.loss_chamfer + self.loss_chamfer_conv4
        elif self.opt.output_conv_pc_num == 4096:
            self.loss = self.loss_chamfer + self.loss_chamfer_conv5 + self.loss_chamfer_conv4
        else:
            self.loss = self.loss_chamfer

        self.loss.backward()

        self.optimizer_encoder.step()
        self.optimizer_decoder.step()

    def test_model(self):
        self.encoder.eval()
        self.decoder.eval()
        self.forward(is_train=False)

        if self.opt.output_conv_pc_num > 0:
            # loss for second last conv pyramid # 32x32
            if self.opt.output_conv_pc_num == 4096:
                self.loss_chamfer_conv5 = self.chamfer_criteria(self.decoder.conv_pc5, self.pc)

            # loss for third last conv pyramid # 16x16
            self.loss_chamfer_conv4 = self.chamfer_criteria(self.decoder.conv_pc4, self.pc)

        # loss for the last pyramid, i.e., the final pc
        self.loss_chamfer = self.chamfer_criteria(self.predicted_pc, self.pc)

        if self.opt.output_conv_pc_num == 1024:
            self.loss = self.loss_chamfer + self.loss_chamfer_conv4
        elif self.opt.output_conv_pc_num == 4096:
            self.loss = self.loss_chamfer + self.loss_chamfer_conv5 + self.loss_chamfer_conv4
        elif self.opt.output_conv_pc_num == 0:
            self.loss = self.loss_chamfer

    # visualization with visdom
    def get_current_visuals(self):
        # display only one instance of pc/img
        input_pc_np = self.input_pc[0].cpu().numpy()
        predicted_pc_np = self.predicted_pc.cpu().data[0].numpy()

        return OrderedDict([('input_pc', input_pc_np),('predicted_pc', predicted_pc_np)])

    def get_current_errors(self):
        return OrderedDict([
            ('total', self.loss_chamfer.data[0]),
            ('forward', self.chamfer_criteria.forward_loss.data[0]),
            ('backward', self.chamfer_criteria.backward_loss.data[0]),
            ('test_loss', self.test_loss.data[0])
        ])

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            # torch.cuda.device(gpu_ids[0])
            network.cuda()

    def update_learning_rate(self, ratio):
        # encoder + decoder
        lr = self.old_lr * ratio
        for param_group in self.optimizer_encoder.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_decoder.param_groups:
            param_group['lr'] = lr
        print('update encoder-decoder learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

