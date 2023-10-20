

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import math
from utils import *


class FewShotReprPredictor(nn.Module):
    def __init__(self,                          
                 nfeat=8,    
                 nzcp=12,                 
                 nzcp_dim=16,
                 nzcp_hid_dim=64,
                 gcn_layer_size=100,            
                 nn_emb_dim=20,                 
                 feat_layer_size=100,           
                 feat_depth = 5,                
                 train_device_list='titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32', 
                 hw_emb_dim=20,
                 search_space='nasbench201'
                 ):
        super(FewShotReprPredictor, self).__init__()

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
            # self.add_module('zcpfc1', nn.Linear(nzcp, 64))
            # self.add_module('zcpfc2', nn.Linear(64, nzcp_dim))
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
                if search_space=='nasbench201':
                    n = nn_emb_dim + hw_emb_dim + nzcp
                else:
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
            (feat, adj, metric) = arch
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
            ############### Feature Interaction ###############
            out = self.sigmoid(self.feat_int_net(torch.cat([gcn_out, device_emb, metric], dim=1)))
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


# class FewShotReprPredictor(nn.Module):
#     def __init__(self,                          
#                  nfeat=8,    
#                  nzcp=12,                 
#                  nzcp_dim=16,
#                  nzcp_hid_dim=64,
#                  gcn_layer_size=100,            
#                  nn_emb_dim=20,                 
#                  feat_layer_size=100,           
#                  feat_depth = 5,                
#                  train_device_list='titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32', 
#                  hw_emb_dim=20,
#                  search_space='nasbench201'
#                  ):
#         super(FewShotReprPredictor, self).__init__()

#         self.search_space = search_space
#         ############### EmbTable ###############
#         self.num_devices = len(train_device_list.split(','))
#         self.dev_emb = nn.Embedding(
#             self.num_devices, 
#             hw_emb_dim
#             )
#         self.sigmoid = nn.Sigmoid()
#         if search_space=='nasbench201':
#             ############### GCN ###############
#             self.gcn_layer_size = gcn_layer_size
#             for i in range(1, 5):
#                 if i == 1:
#                     input_dim = nfeat
#                 else:
#                     input_dim = gcn_layer_size
#                 self.add_module(f'gc{i}', GraphConvolution(input_dim, gcn_layer_size))        
#             hfeat = self.gcn_layer_size
#             self.add_module('fc3', nn.Linear(hfeat, hfeat))
#             self.add_module('fc4', nn.Linear(hfeat, hfeat))
#             self.add_module('fc5', nn.Linear(hfeat, nn_emb_dim))
#             self.add_module('zcpfc1', nn.Linear(nzcp, 64))
#             self.add_module('zcpfc2', nn.Linear(64, nzcp_dim))
#             self.relu = nn.ReLU(inplace=True)
#             self.init_weights()
#         elif search_space=='fbnet':
#             self.add_module('fc1', nn.Linear(nfeat, 100))
#             self.add_module('fc2', nn.Linear(100, 100))
#             self.add_module('fc3', nn.Linear(100, nn_emb_dim))
#             self.relu = nn.ReLU(inplace=True)
            
#         ############### Feature Interaction ###############
#         layers = nn.ModuleList()
#         for i in range(feat_depth):
#             if i==0:
#                 if search_space=='nasbench201':
#                     n = nn_emb_dim + hw_emb_dim + nzcp_dim
#                 else:
#                     n = nn_emb_dim + hw_emb_dim
#             else:
#                 n = feat_layer_size
#             if i==feat_depth-1:
#                 m = 1
#             else:
#                 m = feat_layer_size

#             LL = nn.Linear(int(n), int(m), bias=True)
#             mean = 0.0  # std_dev = np.sqrt(variance)
#             std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
#             W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
#             std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
#             bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
#             LL.weight.data = torch.tensor(W, requires_grad=True)
#             LL.bias.data = torch.tensor(bt, requires_grad=True)
#             layers.append(LL)
#             if i != feat_depth - 1:
#             #     layers.append(nn.Sigmoid())
#             # else:
#                 layers.append(nn.ReLU())
#         self.feat_int_net = nn.Sequential(*layers)

#         # ############### EmbeddingMLP ###############
#         # emb_mlp_layers = nn.ModuleList()
#         # for i in range(2):
#         #     if i==0:
#         #         n = hw_emb_dim
#         #     else:
#         #         n = 100
#         #     if i==feat_depth-1:
#         #         m = hw_emb_dim
#         #     else:
#         #         m = 100

#         #     LL = nn.Linear(int(n), int(m), bias=True)
#         #     mean = 0.0  # std_dev = np.sqrt(variance)
#         #     std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
#         #     W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
#         #     std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
#         #     bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
#         #     LL.weight.data = torch.tensor(W, requires_grad=True)
#         #     LL.bias.data = torch.tensor(bt, requires_grad=True)
#         #     emb_mlp_layers.append(LL)
#         #     if i == feat_depth - 1:
#         #         emb_mlp_layers.append(nn.Sigmoid())
#         #     else:
#         #         emb_mlp_layers.append(nn.ReLU())
#         # self.emb_mlp_net = nn.Sequential(*emb_mlp_layers)

#     def init_weights(self):
#         ############### GCN ###############
#         init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
#         init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
#         init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
#         init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

#     def forward(self, arch, device_input, device_offset=None):
#         ############### EmbTable ###############
#         # (batch_size, hw_emb_dim)
#         device_emb = self.dev_emb(device_input).squeeze(1)
#         # device_emb = self.emb_mlp_net(device_emb)
#         if self.search_space=='nasbench201':
#             ############### GCN ###############
#             (feat, adj, metric) = arch
#             assert len(feat) == len(adj)
#             gcn_out = self.relu(self.gc1(feat, adj).transpose(2,1))
#             gcn_out = gcn_out.transpose(1, 2)
#             gcn_out = self.relu(self.gc2(gcn_out, adj).transpose(2,1))
#             gcn_out = gcn_out.transpose(1, 2)
#             gcn_out = self.relu(self.gc3(gcn_out, adj).transpose(2,1))
#             gcn_out = gcn_out.transpose(1, 2)
#             gcn_out = self.relu(self.gc4(gcn_out, adj).transpose(2,1))
#             gcn_out = gcn_out.transpose(1, 2)
#             gcn_out = gcn_out[:, gcn_out.size()[1] - 1, :]
#             gcn_out = self.relu(self.fc3(gcn_out))
#             gcn_out = self.relu(self.fc4(gcn_out))
#             # (batch_size, nn_emb_dim)
#             gcn_out = self.fc5(gcn_out)
#             metric_out = self.relu(self.zcpfc1(metric))
#             metric_out = self.relu(self.zcpfc2(metric_out))
#             ############### Feature Interaction ###############
#             out = self.sigmoid(self.feat_int_net(torch.cat([gcn_out, device_emb, metric_out], dim=1)))
#         elif self.search_space=='fbnet':
#             gcn_out = self.relu(self.fc1(arch))
#             gcn_out = self.relu(self.fc2(gcn_out))
#             gcn_out = self.relu(self.fc3(gcn_out))
#             ############### Feature Interaction ###############
#             out = self.sigmoid(self.feat_int_net(torch.cat([gcn_out, device_emb], dim=1)))
#         return out

#     def cloned_params(self):
#         params = OrderedDict()
#         for (key, val) in self.named_parameters():
#             params[key] = val.clone()
#         return params


class FewShotMetaPredictor(nn.Module):
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
        super(FewShotMetaPredictor, self).__init__()
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
                layers.append(nn.ReLU())
                # pass
            # else:
        self.feat_int_net = nn.Sequential(*layers)
        
    def init_weights(self):
        ############### GCN ###############
        init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, arch, device_input, device_offset=None, params=None):
        if params==None:
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
        else:
            ############### EmbTable ###############
            # (batch_size, hw_emb_dim)
            # device_emb = self.dev_emb(device_input).squeeze(1)
            device_emb = F.embedding(device_input, params['dev_emb.weight']).squeeze(1)
            # device_emb = self.emb_mlp_net(device_emb)
            if self.search_space=='nasbench201':
                ############### GCN ###############
                (feat, adj) = arch
                assert len(feat) == len(adj)
                gcn_out = F.relu(self.gc1(feat, adj, weight=params['gc1.weight'], bias=params['gc1.bias']).transpose(2,1))
                gcn_out = gcn_out.transpose(1, 2)
                gcn_out = F.relu(self.gc2(gcn_out, adj, weight=params['gc2.weight'], bias=params['gc2.bias']).transpose(2,1))
                gcn_out = gcn_out.transpose(1, 2)
                gcn_out = F.relu(self.gc3(gcn_out, adj, weight=params['gc3.weight'], bias=params['gc3.bias']).transpose(2,1))
                gcn_out = gcn_out.transpose(1, 2)
                gcn_out = F.relu(self.gc4(gcn_out, adj, weight=params['gc4.weight'], bias=params['gc4.bias']).transpose(2,1))
                gcn_out = gcn_out.transpose(1, 2)
                gcn_out = gcn_out[:, gcn_out.size()[1] - 1, :]
                gcn_out = F.relu(F.linear(gcn_out, params['fc3.weight'], params['fc3.bias']))
                gcn_out = F.relu(F.linear(gcn_out, params['fc4.weight'], params['fc4.bias']))
                # (batch_size, nn_emb_dim)
                gcn_out = F.linear(gcn_out, params['fc5.weight'], params['fc5.bias'])
            elif self.search_space=='fbnet':
                gcn_out = F.relu(F.linear(x, params['fc1.weight'], ['fc1.bias']))
                gcn_out = F.relu(F.linear(x, params['fc2.weight'], ['fc2.bias']))
                gcn_out = F.relu(F.linear(x, params['fc3.weight'], ['fc3.bias']))
            ############### Feature Interaction ###############
            # out = self.feat_int_net(torch.cat([gcn_out, device_emb], dim=1))
            out = torch.cat([gcn_out, device_emb], dim=1)
            out = F.relu(F.linear(out, params['feat_int_net.0.weight'], params['feat_int_net.0.bias']))
            out = F.relu(F.linear(out, params['feat_int_net.2.weight'], params['feat_int_net.2.bias']))
            out = F.sigmoid(F.linear(out, params['feat_int_net.4.weight'], params['feat_int_net.4.bias']))
            return out

    def cloned_params(self):
        params = OrderedDict()
        for (key, val) in self.named_parameters():
            params[key] = val.clone()
        return params



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


class FewShotNoEmbeddingPredictor(nn.Module):
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
        super(FewShotNoEmbeddingPredictor, self).__init__()
        self.search_space = search_space
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
            if i == feat_depth - 1:
                # layers.append(nn.Sigmoid())
                pass
            else:
                layers.append(nn.ReLU())
        self.feat_int_net = nn.Sequential(*layers)


        ############### EmbeddingMLP ###############
        emb_mlp_layers = nn.ModuleList()
        for i in range(2):
            if i==0:
                n = hw_emb_dim
            else:
                n = 100
            if i==2-1:
                m = hw_emb_dim
            else:
                m = 100

            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            emb_mlp_layers.append(LL)
            # if i == feat_depth - 1:
            #     emb_mlp_layers.append(nn.Sigmoid())
            # else:
            emb_mlp_layers.append(nn.ReLU())
        self.emb_mlp_net = nn.Sequential(*emb_mlp_layers)

    def init_weights(self):
        ############### GCN ###############
        init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, arch, hw_emb):
        hw_emb = self.emb_mlp_net(hw_emb)
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
        out = self.sigmoid(self.feat_int_net(torch.cat([gcn_out, hw_emb], dim=1)))
        return out
    def cloned_params(self):
        params = OrderedDict()
        for (key, val) in self.named_parameters():
            params[key] = val.clone()
        return params



# class FewShotZModulatedPredictor(nn.Module):
#     def __init__(self,                          
#                  nfeat                  =   8,                       
#                  gcn_layer_size         =   100,            
#                  nn_emb_dim             =   20,                 
#                  feat_layer_size        =   100,           
#                  feat_depth             =   5,                
#                  train_device_list      =   'titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32', 
#                  transfer_device_list   =   'eyeriss,pixel3,raspi4',
#                  z_mod_depth            =   2,
#                  z_mod_width            =   100,
#                  hw_emb_dim             =   20
#                  ):
#         super(FewShotZModulatedPredictor, self).__init__()
#         self.num_train_devices = len(train_device_list.split(','))
#         self.num_transfer_devices = len(transfer_device_list.split(','))
#         self.tot_dev = self.num_train_devices + self.num_transfer_devices
#         ############### EmbTable ###############
#         self.num_devices = len(train_device_list.split(','))
#         self.dev_emb = nn.Embedding(
#             self.num_devices, 
#             hw_emb_dim
#             )
            
#         ############### Z Modulator ###############
#         for z_mod_net_idx in range(self.num_train_devices, self.tot_dev, 1):
#             layers = nn.ModuleList()
#             for i in range(z_mod_depth):
#                 if i==0:
#                     n = hw_emb_dim
#                 else:
#                     n = z_mod_width
#                 if i==z_mod_depth-1:
#                     m = hw_emb_dim
#                 else:
#                     m = z_mod_width

#                 LL = nn.Linear(int(n), int(m), bias=True)
#                 mean = 0.0  # std_dev = np.sqrt(variance)
#                 std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
#                 W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
#                 std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
#                 bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
#                 LL.weight.data = torch.tensor(W, requires_grad=True)
#                 LL.bias.data = torch.tensor(bt, requires_grad=True)
#                 layers.append(LL)
#                 if i != feat_depth - 1:
#                     layers.append(nn.ReLU(inplace=False))
#             exec('self.z_modulator_net_' + str(z_mod_net_idx) + ' = nn.Sequential(*layers)')

#         ############### GCN ###############
#         self.gcn_layer_size = gcn_layer_size
#         for i in range(1, 5):
#             if i == 1:
#                 input_dim = nfeat
#             else:
#                 input_dim = gcn_layer_size
#             self.add_module(f'gc{i}', GraphConvolution(input_dim, gcn_layer_size))        
#         hfeat = self.gcn_layer_size
#         self.add_module('fc3', nn.Linear(hfeat, hfeat))
#         self.add_module('fc4', nn.Linear(hfeat, hfeat))
#         self.add_module('fc5', nn.Linear(hfeat, nn_emb_dim))
#         self.relu = nn.ReLU(inplace=True)
#         self.init_weights()
        
#         ############### Feature Interaction ###############
#         layers = nn.ModuleList()
#         for i in range(feat_depth):
#             if i==0:
#                 n = nn_emb_dim + hw_emb_dim
#             else:
#                 n = feat_layer_size
#             if i==feat_depth-1:
#                 m = 1
#             else:
#                 m = feat_layer_size

#             LL = nn.Linear(int(n), int(m), bias=True)
#             mean = 0.0  # std_dev = np.sqrt(variance)
#             std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
#             W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
#             std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
#             bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
#             LL.weight.data = torch.tensor(W, requires_grad=True)
#             LL.bias.data = torch.tensor(bt, requires_grad=True)
#             layers.append(LL)
#             if i == feat_depth - 1:
#                 layers.append(nn.Sigmoid())
#             else:
#                 layers.append(nn.ReLU())
#         self.feat_int_net = nn.Sequential(*layers)


#     def init_weights(self):
#         ############### GCN ###############
#         init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
#         init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
#         init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
#         init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

#     def forward(self, arch, device_input, device_offset=None):
#         ############### EmbTable ###############
#         # (batch_size, hw_emb_dim)
#         device_emb = self.dev_emb(device_input).squeeze(1)
#         # print(device_emb)
#         ############### Z Modulator ###############
#         # print([dev_.item() for dev_ in device_input])
#         for idx, dev_ in enumerate(device_input):
#             if dev_.item() > self.num_train_devices-1:
#                 z_mod_net = eval('self.z_modulator_net_' + str(dev_.item()))
#                 # new_emb = z_mod_net(device_emb[idx])
#                 # print(new_emb.shape)
#                 # print(device_emb[idx].shape)
#                 device_emb[idx] = z_mod_net(device_emb[idx])
#                 # exit(0)
#         ############### GCN ###############
#         (feat, adj) = arch
#         assert len(feat) == len(adj)
#         gcn_out = self.relu(self.gc1(feat, adj).transpose(2,1))
#         gcn_out = gcn_out.transpose(1, 2)
#         gcn_out = self.relu(self.gc2(gcn_out, adj).transpose(2,1))
#         gcn_out = gcn_out.transpose(1, 2)
#         gcn_out = self.relu(self.gc3(gcn_out, adj).transpose(2,1))
#         gcn_out = gcn_out.transpose(1, 2)
#         gcn_out = self.relu(self.gc4(gcn_out, adj).transpose(2,1))
#         gcn_out = gcn_out.transpose(1, 2)
#         gcn_out = gcn_out[:, gcn_out.size()[1] - 1, :]
#         gcn_out = self.relu(self.fc3(gcn_out))
#         gcn_out = self.relu(self.fc4(gcn_out))
#         # (batch_size, nn_emb_dim)
#         gcn_out = self.fc5(gcn_out)
#         ############### Feature Interaction ###############
#         out = self.feat_int_net(torch.cat([gcn_out, device_emb], dim=1))
#         return out

class FewShotZModulatedPredictor(nn.Module):
    def __init__(self,                          
                 nfeat                  =   8,                       
                 gcn_layer_size         =   100,            
                 nn_emb_dim             =   20,                 
                 feat_layer_size        =   100,           
                 feat_depth             =   5,                
                 train_device_list      =   'titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32', 
                 transfer_device_list   =   'eyeriss,pixel3,raspi4',
                 z_mod_depth            =   2,
                 z_mod_width            =   100,
                 hw_emb_dim             =   20
                 ):
        super(FewShotZModulatedPredictor, self).__init__()
        self.num_train_devices = len(train_device_list.split(','))
        self.num_transfer_devices = len(transfer_device_list.split(','))
        self.tot_dev = self.num_train_devices + self.num_transfer_devices
        ############### EmbTable ###############
        self.num_devices = len(train_device_list.split(','))
        self.dev_emb = nn.Embedding(
            self.num_devices, 
            hw_emb_dim
            )
            
        ############### Z Modulator ###############
        layers = nn.ModuleList()
        for i in range(z_mod_depth):
            if i==0:
                n = hw_emb_dim
            else:
                n = z_mod_width
            if i==z_mod_depth-1:
                m = hw_emb_dim
            else:
                m = z_mod_width

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
                layers.append(nn.ReLU(inplace=False))
        self.z_modulator = nn.Sequential(*layers)

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
            if i == feat_depth - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        self.feat_int_net = nn.Sequential(*layers)


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
        # print(device_emb)
        ############### Z Modulator ###############
        device_emb = self.z_modulator(device_emb)
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
        ############### Feature Interaction ###############
        out = self.feat_int_net(torch.cat([gcn_out, device_emb], dim=1))
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_, adj, weight=None, bias=None):
        if weight is not None:
            support = torch.matmul(input_, weight)
            output = torch.bmm(adj, support)
            if bias is not None:
                return output+bias
            else:
                return output
            
        else :
            support = torch.matmul(input_, self.weight)
            output = torch.bmm(adj, support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FewShotSinglePredictor(nn.Module):
    def __init__(self,                          
                 nfeat=8,                       
                 gcn_layer_size=100,            
                 ):
        super(FewShotSinglePredictor, self).__init__()

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
        self.add_module('fc5', nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()


    def init_weights(self):
        ############### GCN ###############
        init.uniform_(self.gc1.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc2.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc3.weight, a=-0.05, b=0.05)
        init.uniform_(self.gc4.weight, a=-0.05, b=0.05)

    def forward(self, arch, hw_emb):
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
        gcn_out = self.fc5(gcn_out)
        return out
