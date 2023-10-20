import torch
import json
from tqdm import tqdm
import types
import copy
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import pickle
import random, time

import sys, os

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'


class FBNet:
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True):
        self.primitives = [
            'k3_e1' ,
            'k3_e1_g2' ,
            'k3_e3' ,
            'k3_e6' ,
            'k5_e1' ,
            'k5_e1_g2',
            'k5_e3',
            'k5_e6',
            'skip'
        ]
        self.archs = torch.load(BASE_PATH + "fbnet_archs.pt")
        self.num_uniq_ops = len(self.primitives)
        # self.zcp_dict = json.load(open(BASE_PATH + "zc_fbnet.json"), "r")
        # self.normalize_and_process_zcp(normalize_zcp, log_synflow)
    def min_max_scaling(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def log_transform(self, data):
        return np.log1p(data)

    def standard_scaling(self, data):
        return (data - np.mean(data)) / np.std(data)

    # def normalize_and_process_zcp(self, normalize_zcp, log_synflow):
    #     if normalize_zcp:
    #         print("Normalizing ZCP dict")
    #         # Add normalization code here
    #         self.norm_zcp['epe_nas'] = self.min_max_scaling(self.norm_zcp['epe_nas'])
    #         self.norm_zcp['fisher'] = self.min_max_scaling(self.log_transform(self.norm_zcp['fisher']))
    #         self.norm_zcp['flops'] = self.min_max_scaling(self.log_transform(self.norm_zcp['flops']))
    #         self.norm_zcp['grad_norm'] = self.min_max_scaling(self.log_transform(self.norm_zcp['grad_norm']))
    #         self.norm_zcp['grasp'] = self.standard_scaling(self.norm_zcp['grasp'])
    #         self.norm_zcp['jacov'] = self.min_max_scaling(self.norm_zcp['jacov'])
    #         self.norm_zcp['l2_norm'] = self.min_max_scaling(self.norm_zcp['l2_norm'])
    #         self.norm_zcp['nwot'] = self.min_max_scaling(self.norm_zcp['nwot'])
    #         self.norm_zcp['params'] = self.min_max_scaling(self.log_transform(self.norm_zcp['params']))
    #         self.norm_zcp['plain'] = self.min_max_scaling(self.norm_zcp['plain'])
    #         self.norm_zcp['snip'] = self.min_max_scaling(self.log_transform(self.norm_zcp['snip']))
    #         if log_synflow:
    #             self.norm_zcp['synflow'] = self.min_max_scaling(self.log_transform(self.norm_zcp['synflow']))
    #         else:
    #             self.norm_zcp['synflow'] = self.min_max_scaling(self.norm_zcp['synflow'])
    #         self.norm_zcp['zen'] = self.min_max_scaling(self.norm_zcp['zen'])
    
    def get_adjmlp_zcp(self, idx):
        return np.concatenate([self.archs[idx], np.asarray(self.get_zcp(idx))]).tolist()
    
    def get_adj_op(self, idx):
        adj = np.eye(len(self.archs[idx])).tolist()
        op = self.archs[idx]
        # Convert integer array to one-hot array
        op = np.eye(len(self.primitives))[op].tolist()
        return {"module_adjacency": adj, "module_operations": op}

    def get_numitems(self, space=None):
        return 5000
    
    def get_zcp(self, idx):
        return self.zcp_dict['cifar10']["_".join([str(czs) for czs in self.archs[idx]])]
