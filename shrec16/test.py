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
import os

from models.classifier import Model
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from util.visualizer import Visualizer


if __name__=='__main__':
    testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#testing point clouds = %d' % len(testset))

    # create model, optionally load pre-trained model
    model = Model(opt)
    model.encoder.load_state_dict(torch.load('/ssd/jiaxin/SO-Net/shrec16/checkpoints/0_0.748621_net_encoder.pth'))
    model.classifier.load_state_dict(torch.load('/ssd/jiaxin/SO-Net/shrec16/checkpoints/0_0.748621_net_classifier.pth'))
    output_folder = '/ssd/tmp/retrieval'

    model.encoder.eval()
    model.classifier.eval()

    visualizer = Visualizer(opt)
    softmax_layer = nn.Softmax2d()

    batch_amount = 0
    feature_map = torch.FloatTensor(len(testset), 55).cuda().zero_()  # Nx55
    predicted_labels = torch.LongTensor(len(testset)).cuda().zero_()  # N
    model_name_ids = torch.LongTensor(len(testset)).cuda().zero_()  # N
    for i, data in enumerate(testloader):
        input_pc, input_sn, input_label, input_node, input_node_knn_I, input_model_name_id = data
        # input_pc, input_sn, input_model_name_id, input_node = data
        model.set_input(input_pc, input_sn, input_model_name_id, input_node,input_node_knn_I)
        model.forward()

        batch_size = input_model_name_id.size()[0]

        # feature_map[batch_amount:batch_amount+batch_size] = softmax_layer(model.score.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2).data
        feature_map[batch_amount:batch_amount+batch_size] = model.score.data

        _, predicted_idx = torch.max(model.score.data, dim=1, keepdim=False)
        predicted_labels[batch_amount:batch_amount+batch_size] = predicted_idx

        model_name_ids[batch_amount:batch_amount+batch_size] = input_model_name_id

        batch_amount += batch_size

    print(feature_map.size())
    print(predicted_labels.size())
    print(model_name_ids.size())
    print(feature_map)

    # calculate neighbors
    for i in range(len(testset)):
        # find instance that has the same label
        feature = feature_map[i]  # 55
        label = predicted_labels[i]  # N

        mask = torch.eq(predicted_labels, label)  # N
        same_label_indices = torch.nonzero(mask).squeeze(1)  # K

        # print(same_label_indices)

        feature_selected = feature_map[same_label_indices]  # Kx55
        model_name_id_selected = model_name_ids[same_label_indices]  # K

        distance = torch.norm(feature.unsqueeze(0)-feature_selected, p=2, dim=1)  # Kx55 -> K
        sorted, indices = torch.sort(distance)

        nn_model_name_id = model_name_id_selected[indices].cpu().numpy()
        nn_distance = sorted.cpu().numpy()

        # print(indices)
        # print(nn_model_name_id)
        # print(nn_distance)

        # write to file
        model_name = '%06d' % model_name_ids[i]
        nn_result = np.transpose(np.vstack((nn_model_name_id, nn_distance)))

        if nn_result.shape[0]<=1000:
            np.savetxt(os.path.join(output_folder, model_name), nn_result, fmt='%06d %f', delimiter=' ')
        else:
            np.savetxt(os.path.join(output_folder, model_name), nn_result[0:1000,:], fmt='%06d %f', delimiter=' ')
