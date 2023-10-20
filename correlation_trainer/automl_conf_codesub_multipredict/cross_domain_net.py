

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import math
from utils_transfer import *


class CrossDomainMultiNet(nn.Module):
    def __init__(self,
                 n_zcproxies=12,
                 feat_layersize=256,                
                 train_device_list='titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32', 
                 hw_emb_dim=20,
                 feat_depth = 5):
        super(CrossDomainMultiNet, self).__init__()
        device = 'cuda:0'
        
        ############### EmbTable ###############
        self.num_devices = 2*len(train_device_list.split(','))
        self.dev_emb = nn.Embedding(
            self.num_devices, 
            hw_emb_dim
            )

        #######################################
        self.n_zcproxies = n_zcproxies
        self.hw_emb_dim = hw_emb_dim
        self.feat_layersize = feat_layersize
        self.feat_depth = feat_depth

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        for i in range(feat_depth):
            if i==0:
                n = n_zcproxies + hw_emb_dim
            else:
                n = feat_layersize
            if i==feat_depth-1:
                m = 1
            else:
                m = feat_layersize

            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i != feat_depth - 1:
                layers.append(nn.ReLU())
        self.feat_int_net = nn.Sequential(*layers)

    def forward(self, zcprox, device_input):
        device_emb = self.dev_emb(device_input).squeeze(1)
        out = self.sigmoid(self.feat_int_net(torch.cat([zcprox.float(), device_emb], dim=1)))
        return out



class CrossDomainNet(nn.Module):
    def __init__(self,
                 n_zcproxies=12,
                 feat_layersize=256,
                 feat_depth = 5):
        super(CrossDomainNet, self).__init__()
        device = 'cuda:0'
        self.n_zcproxies = n_zcproxies
        self.feat_layersize = feat_layersize
        self.feat_depth = feat_depth
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        for i in range(feat_depth):
            if i==0:
                n = n_zcproxies
            else:
                n = feat_layersize
            if i==feat_depth-1:
                m = 1
            else:
                m = feat_layersize

            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i != feat_depth - 1:
                layers.append(nn.ReLU())
        self.feat_int_net = nn.Sequential(*layers)

    def forward(self, zcprox):
        out = self.sigmoid(self.feat_int_net(zcprox))
        return out


class FewShotPredictor(nn.Module):
    def __init__(self,                          
                 nfeat=8,                       
                 gcn_layer_size=100,            
                 nn_emb_dim=20,                 
                 feat_layer_size=100,           
                 feat_depth = 5,                
                 train_device_list='titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32', 
                 hw_emb_dim=20,
                 search_space='nasbench201'
                 ):
        super(FewShotPredictor, self).__init__()

        self.search_space = search_space
        ############### EmbTable ###############
        self.num_devices = len(train_device_list.split(','))
        self.dev_emb = nn.Embedding(
            self.num_devices, 
            hw_emb_dim
            )
        self.sigmoid = nn.Sigmoid()
        if search_space=='nasbench201':
            ############### GCN ###############
            self.gcn_layer_size = gcn_layer_size
            for i in range(1, 5):
                if i == 1:
                    input_dim = nfeat
                else:
                    input_dim = gcn_layer_size
                self.add_module(f'gc{i}', GraphConvolution(input_dim, gcn_layer_size))        
            hfeat = self.gcn_layer_size
            self.add_module('fc3', nn.Linear(hfeat, hfeat))
            self.add_module('fc4', nn.Linear(hfeat, hfeat))
            self.add_module('fc5', nn.Linear(hfeat, nn_emb_dim))
            self.relu = nn.ReLU(inplace=True)
            self.init_weights()
        elif search_space=='fbnet':
            self.add_module('fc1', nn.Linear(nfeat, 100))
            self.add_module('fc2', nn.Linear(100, 100))
            self.add_module('fc3', nn.Linear(100, nn_emb_dim))
            self.relu = nn.ReLU(inplace=True)
            
        ############### Feature Interaction ###############
        layers = nn.ModuleList()
        for i in range(feat_depth):
            if i==0:
                n = nn_emb_dim + hw_emb_dim
            else:
                n = feat_layer_size
            if i==feat_depth-1:
                m = 1
            else:
                m = feat_layer_size

            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)
            if i != feat_depth - 1:
            #     layers.append(nn.Sigmoid())
            # else:
                layers.append(nn.ReLU())
        self.feat_int_net = nn.Sequential(*layers)

        # ############### EmbeddingMLP ###############
        # emb_mlp_layers = nn.ModuleList()
        # for i in range(2):
        #     if i==0:
        #         n = hw_emb_dim
        #     else:
        #         n = 100
        #     if i==feat_depth-1:
        #         m = hw_emb_dim
        #     else:
        #         m = 100

        #     LL = nn.Linear(int(n), int(m), bias=True)
        #     mean = 0.0  # std_dev = np.sqrt(variance)
        #     std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        #     W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        #     std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
        #     bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
        #     LL.weight.data = torch.tensor(W, requires_grad=True)
        #     LL.bias.data = torch.tensor(bt, requires_grad=True)
        #     emb_mlp_layers.append(LL)
        #     if i == feat_depth - 1:
        #         emb_mlp_layers.append(nn.Sigmoid())
        #     else:
        #         emb_mlp_layers.append(nn.ReLU())
        # self.emb_mlp_net = nn.Sequential(*emb_mlp_layers)

    def init_weights(self):
        ############### GCN ###############
        init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, arch, device_input, device_offset=None):
        ############### EmbTable ###############
        # (batch_size, hw_emb_dim)
        device_emb = self.dev_emb(device_input).squeeze(1)
        # device_emb = self.emb_mlp_net(device_emb)
        if self.search_space=='nasbench201':
            ############### GCN ###############
            (feat, adj) = arch
            assert len(feat) == len(adj)
            gcn_out = self.relu(self.gc1(feat, adj).transpose(2,1))
            gcn_out = gcn_out.transpose(1, 2)
            gcn_out = self.relu(self.gc2(gcn_out, adj).transpose(2,1))
            gcn_out = gcn_out.transpose(1, 2)
            gcn_out = self.relu(self.gc3(gcn_out, adj).transpose(2,1))
            gcn_out = gcn_out.transpose(1, 2)
            gcn_out = self.relu(self.gc4(gcn_out, adj).transpose(2,1))
            gcn_out = gcn_out.transpose(1, 2)
            gcn_out = gcn_out[:, gcn_out.size()[1] - 1, :]
            gcn_out = self.relu(self.fc3(gcn_out))
            gcn_out = self.relu(self.fc4(gcn_out))
            # (batch_size, nn_emb_dim)
            gcn_out = self.fc5(gcn_out)
        elif self.search_space=='fbnet':
            gcn_out = self.relu(self.fc1(arch))
            gcn_out = self.relu(self.fc2(gcn_out))
            gcn_out = self.relu(self.fc3(gcn_out))
        ############### Feature Interaction ###############
        out = self.sigmoid(self.feat_int_net(torch.cat([gcn_out, device_emb], dim=1)))
        return out

    def cloned_params(self):
        params = OrderedDict()
        for (key, val) in self.named_parameters():
            params[key] = val.clone()
        return params
