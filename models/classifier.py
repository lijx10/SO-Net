import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
from collections import OrderedDict
import os
import sys
import random

from . import networks
from . import losses


class Model():
    def __init__(self, opt):
        self.opt = opt

        self.encoder = networks.Encoder(opt)
        self.classifier = networks.Classifier(opt)

        # learning rate_control
        if self.opt.pretrain is not None:
            self.old_lr_encoder = self.opt.lr * self.opt.pretrain_lr_ratio
        else:
            self.old_lr_encoder = self.opt.lr
        self.old_lr_classifier = self.opt.lr

        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(),
                                                  lr=self.old_lr_encoder,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=0)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(),
                                                     lr= self.old_lr_classifier,
                                                     betas=(0.9, 0.999),
                                                     weight_decay=0)

        self.softmax_criteria = nn.CrossEntropyLoss()
        if self.opt.gpu_id >= 0:
            self.encoder = self.encoder.to(self.opt.device)
            self.classifier = self.classifier.to(self.opt.device)
            self.softmax_criteria = self.softmax_criteria.to(self.opt.device)

        # place holder for GPU tensors
        self.input_pc = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.input_sn = torch.FloatTensor(self.opt.batch_size, 3, self.opt.input_pc_num).uniform_()
        self.input_label = torch.LongTensor(self.opt.batch_size).fill_(1)
        self.input_node = torch.FloatTensor(self.opt.batch_size, 3, self.opt.node_num)
        self.input_node_knn_I = torch.LongTensor(self.opt.batch_size, self.opt.node_num, self.opt.som_k)

        # record the test loss and accuracy
        self.test_loss = torch.tensor([0], dtype=torch.float32, requires_grad=False)
        self.test_accuracy = torch.tensor([0], dtype=torch.float32, requires_grad=False)

        if self.opt.gpu_id >= -1:
            self.input_pc = self.input_pc.to(self.opt.device)
            self.input_sn = self.input_sn.to(self.opt.device)
            self.input_label = self.input_label.to(self.opt.device)
            self.input_node = self.input_node.to(self.opt.device)
            self.input_node_knn_I = self.input_node_knn_I.to(self.opt.device)
            self.test_loss = self.test_loss.to(self.opt.device)
            # self.test_accuracy = self.test_accuracy.to(self.opt.device)

    def set_input(self, input_pc, input_sn, input_label, input_node, input_node_knn_I):
        self.input_pc.resize_(input_pc.size()).copy_(input_pc)
        self.input_sn.resize_(input_sn.size()).copy_(input_sn)
        self.input_label.resize_(input_label.size()).copy_(input_label)
        self.input_node.resize_(input_node.size()).copy_(input_node)
        self.input_node_knn_I.resize_(input_node_knn_I.size()).copy_(input_node_knn_I)
        self.pc = self.input_pc.detach()
        self.sn = self.input_sn.detach()
        self.label = self.input_label.detach()

    def forward(self, is_train=False, epoch=None):
        self.feature = self.encoder(self.pc, self.sn, self.input_node, self.input_node_knn_I, is_train, epoch)  # Bx1024
        self.score = self.classifier(self.feature, epoch)

    def optimize(self, epoch=None):
        # random point dropout
        if self.opt.random_pc_dropout_lower_limit < 0.99:
            dropout_keep_ratio = random.uniform(self.opt.random_pc_dropout_lower_limit, 1.0)
            resulting_pc_num = round(dropout_keep_ratio*self.opt.input_pc_num)
            chosen_indices = np.random.choice(self.opt.input_pc_num, resulting_pc_num, replace=False)
            chosen_indices_tensor = torch.from_numpy(chosen_indices).to(self.opt.device)
            self.pc = torch.index_select(self.pc, dim=2, index=chosen_indices_tensor)
            self.sn = torch.index_select(self.sn, dim=2, index=chosen_indices_tensor)

        self.encoder.train()
        self.classifier.train()
        self.forward(is_train=True, epoch=epoch)

        self.encoder.zero_grad()
        self.classifier.zero_grad()

        self.loss = self.softmax_criteria(self.score, self.label)
        self.loss.backward()

        self.optimizer_encoder.step()
        self.optimizer_classifier.step()

    def test_model(self):
        self.encoder.eval()
        self.classifier.eval()
        self.forward(is_train=False)
        self.loss = self.softmax_criteria(self.score, self.label)

    # visualization with visdom
    def get_current_visuals(self):
        # display only one instance of pc/img
        input_pc_np = self.input_pc[0].cpu().numpy()

        return OrderedDict([('input_pc', input_pc_np)])

    def get_current_errors(self):
        # get the accuracy
        _, predicted_idx = torch.max(self.score.data, dim=1, keepdim=False)
        correct_mask = torch.eq(predicted_idx, self.input_label).float()
        train_accuracy = torch.mean(correct_mask)

        return OrderedDict([
            ('train_loss', self.loss.item()),
            ('train_accuracy', train_accuracy.item()),
            ('test_loss', self.test_loss.item()),
            ('test_accuracy', self.test_accuracy.item())
        ])

    def save_network(self, network, network_label, epoch_label, gpu_id):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id>=0 and torch.cuda.is_available():
            # torch.cuda.device(gpu_id)
            network.to(self.opt.device)

    def update_learning_rate(self, ratio):
        lr_clip = 0.00001

        # encoder
        lr_encoder = self.old_lr_encoder * ratio
        if lr_encoder < lr_clip:
            lr_encoder = lr_clip
        for param_group in self.optimizer_encoder.param_groups:
            param_group['lr'] = lr_encoder
        print('update encoder learning rate: %f -> %f' % (self.old_lr_encoder, lr_encoder))
        self.old_lr_encoder = lr_encoder

        # classifier
        lr_classifier = self.old_lr_classifier * ratio
        if lr_classifier < lr_clip:
            lr_classifier = lr_clip
        for param_group in self.optimizer_classifier.param_groups:
            param_group['lr'] = lr_classifier
        print('update classifier learning rate: %f -> %f' % (self.old_lr_classifier, lr_classifier))
        self.old_lr_classifier = lr_classifier