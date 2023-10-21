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
import pandas as pd

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
        self.arch2vec_fbnet = torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_fbnet-fbnet.pt")
        self.zcpn_list = ['synflow', 'zen', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip']
        self.archs = torch.load(BASE_PATH + "fbnet_archs.pt")
        self.num_uniq_ops = len(self.primitives)
        self.zcp_dict = json.load(open(BASE_PATH + "zc_fbnet.json", "r"))
        self.normalize_and_process_zcp(normalize_zcp, log_synflow)
        self.cate_fbnet = torch.load(BASE_PATH + "cate_embeddings/cate_fbnet.pt")
        self.devices = ["1080ti_1","1080ti_32","1080ti_64","2080ti_1","2080ti_32","2080ti_64","essential_ph_1","eyeriss","fpga",\
                        "gold_6226","gold_6240","pixel2","pixel3","raspi4","samsung_a50","samsung_s7","silver_4114","silver_4210r",\
                        "titan_rtx_1","titan_rtx_32","titan_rtx_64","titanx_1","titanx_32","titanx_64","titanxp_1","titanxp_32","titanxp_64"]
        latency_data = {}
        for dev_ in self.devices:
            latency_data[dev_] = torch.load(BASE_PATH + "/fbnet_latency/" + dev_ + ".pt")

        # Normalize each key in latency_data
        for k, v in latency_data.items():
            latency_data[k] = self.min_max_scaling(np.asarray(v)).tolist()
        self.latency_data = latency_data
        self.device_key = {device: idx for idx, device in enumerate(self.devices)}

    def min_max_scaling(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def log_transform(self, data):
        return np.log1p(data)

    def standard_scaling(self, data):
        return (data - np.mean(data)) / np.std(data)

    def normalize_and_process_zcp(self, normalize_zcp, log_synflow):
        if normalize_zcp:
            print("Normalizing ZCP dict")
            self.norm_zcp = pd.DataFrame(self.zcp_dict).T
            self.unnorm_zcp_dict = self.norm_zcp.T.to_dict()
            # Add normalization code here
            self.norm_zcp['epe_nas'] = self.min_max_scaling(self.norm_zcp['epe_nas'])
            self.norm_zcp['fisher'] = self.min_max_scaling(self.log_transform(self.norm_zcp['fisher']))
            self.norm_zcp['flops'] = self.min_max_scaling(self.log_transform(self.norm_zcp['flops']))
            self.norm_zcp['grad_norm'] = self.min_max_scaling(self.log_transform(self.norm_zcp['grad_norm']))
            self.norm_zcp['grasp'] = self.standard_scaling(self.norm_zcp['grasp'])
            self.norm_zcp['jacov'] = self.min_max_scaling(self.norm_zcp['jacov'])
            self.norm_zcp['l2_norm'] = self.min_max_scaling(self.norm_zcp['l2_norm'])
            self.norm_zcp['nwot'] = self.min_max_scaling(self.norm_zcp['nwot'])
            self.norm_zcp['params'] = self.min_max_scaling(self.log_transform(self.norm_zcp['params']))
            self.norm_zcp['plain'] = self.min_max_scaling(self.norm_zcp['plain'])
            self.norm_zcp['snip'] = self.min_max_scaling(self.log_transform(self.norm_zcp['snip']))
            if log_synflow:
                self.norm_zcp['synflow'] = self.min_max_scaling(self.log_transform(self.norm_zcp['synflow']))
            else:
                self.norm_zcp['synflow'] = self.min_max_scaling(self.norm_zcp['synflow'])
            self.norm_zcp['zen'] = self.min_max_scaling(self.norm_zcp['zen'])
            self.zcp_dict = self.norm_zcp.T.to_dict()
    
    def get_adjmlp_zcp(self, idx, space=None, task=None):
        return np.concatenate([[xm/self.num_uniq_ops for xm in self.archs[idx]], np.asarray(self.get_zcp(idx))]).tolist()
    
    def get_adj_op(self, idx, space=None, task=None):
        adj = np.eye(len(self.archs[idx])).tolist()
        op = self.archs[idx]
        # Convert integer array to one-hot array
        op = np.eye(len(self.primitives))[op].tolist()
        return {"module_adjacency": adj, "module_operations": op}
    
    def get_valacc(self, idx, space=None, task=None):
        return 0.0

    def get_numitems(self, space=None, task=None):
        return 5000
    
    def get_zcp(self, idx, joint=None, space=None, task=None):
        return [self.zcp_dict["_".join([str(czs) for czs in self.archs[idx]])][ke] for ke in self.zcpn_list]

    def get_latency(self, idx, device="1080ti_1", space=None, task=None):
        return self.latency_data[device][idx]

    def get_device_index(self, device="1080ti_1"):
        return self.device_key[device]

    def get_arch2vec(self, idx, joint=None, space=None, task=None):
        return self.arch2vec_fbnet[idx]['feature'].tolist()
    
    def get_params(self, idx, space=None, task=None,):
        return self.unnorm_zcp_dict["_".join([str(czs) for czs in self.archs[idx]])]['params']
        
    def get_cate(self, idx, joint=None, space=None, task=None):
        return self.cate_fbnet['embeddings'][idx].tolist()
        
    def get_a2vcatezcp(self, idx, joint=None, space=None, task=None):
        a2v = self.get_arch2vec(idx, joint=joint, space=space)
        if not isinstance(a2v, list):
            a2v = a2v.tolist()
        cate = self.get_cate(idx, joint=joint, space=space)
        if not isinstance(cate, list):
            cate = cate.tolist()
        zcp = self.get_zcp(idx, joint=joint, space=space)
        if not isinstance(zcp, list):
            zcp = zcp.tolist()
        return a2v + cate + zcp
    
    def get_norm_w_d(self, idx, space=None):
        return [0, 0]
    