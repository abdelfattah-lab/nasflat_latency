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
import random, time
import sys, os

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
NDS_DPATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/NDS/nds_data/'

class NDS:
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True, embedding_list = ['adj',
                                                                    'adj_op',
                                                                    'path',
                                                                    'one_hot',
                                                                    'path_indices',
                                                                    'zcp']):
        adj_path = BASE_PATH + 'nds_adj_encoding/'
        # self.spaces = ['Amoeba.json', 'NASNet.json','DARTS.json','ENAS.json','PNAS.json']
        self.spaces = ['Amoeba.json','PNAS_fix-w-d.json','ENAS_fix-w-d.json','NASNet.json','DARTS.json','ENAS.json','PNAS.json','DARTS_lr-wd.json','DARTS_fix-w-d.json']
        self.cate_embeddings = {k.replace(".json", ""): torch.load(BASE_PATH + 'cate_embeddings/cate_nds_{}.pt'.format(k.replace(".json", ""))) for k in self.spaces}
        self.arch2vec_embeddings = {k.replace(".json", ""): torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_{}-nds.pt".format(k.replace(".json", ""))) for k in self.spaces}
        self.space_dicts = {space.replace(".json", ""): json.load(open(NDS_DPATH + space, "r")) for space in self.spaces}
        self.space_adj_mats = {space.replace(".json", ""): json.load(open(adj_path + space, "r")) for space in self.spaces}
        # for task_ in self.spaces:
        #     task_ = task_.replace(".json", "")
        #     adj_mat_list = []
        #     for kx in self.space_adj_mats[task_].keys():
        #         adj_mat_list.append(self.space_adj_mats[task_][kx])
        #     # MinMaxScaler Normalize adj_mat_list
        #     min_max_scaler = preprocessing.MinMaxScaler()
        #     adj_mat_list = np.array(adj_mat_list)
        #     adj_mat_list = min_max_scaler.fit_transform(adj_mat_list)
        #     for kx in self.spaces[task_].keys():
        #         self.spaces[task_][kx] = adj_mat_list[kx].tolist()  
        self.zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']
        self.normalize_zcp = normalize_zcp
        self.zcp_nds_norm = {}
        if self.normalize_zcp:
            for task_ in self.spaces:
                print("normalizing task: ", task_)
                self.norm_zcp = pd.read_csv(BASE_PATH + "nds_zcps/" + task_.replace(".json", "") + "_zcps.csv", index_col=0)
                self.norm_zcp = self.norm_zcp[self.zcps]
                minfinite = self.norm_zcp['zen'].replace(-np.inf, 1000).min()
                self.norm_zcp['zen'] = self.norm_zcp['zen'].replace(-np.inf, minfinite)
                if log_synflow:
                    self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(0, 1e-2)
                    self.norm_zcp['synflow'] = np.log10(self.norm_zcp['synflow'])
                else:
                    print("WARNING: Not taking log of synflow values for normalization results in very small synflow inputs")
                minfinite = self.norm_zcp['synflow'].replace(-np.inf, 1000).min()
                self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(-np.inf, minfinite)
                min_max_scaler = preprocessing.QuantileTransformer()
                self.norm_zcp = pd.DataFrame(min_max_scaler.fit_transform(self.norm_zcp), columns=self.zcps, index=self.norm_zcp.index)
                self.zcp_nds_norm[task_.replace(".json", "")] = self.norm_zcp.T.to_dict()
        for task_ in self.spaces:
            # normalize cate_embeddings[space]['embeddings']
            min_max_scaler = preprocessing.MinMaxScaler()
            self.cate_embeddings[task_.replace(".json", "")]['embeddings'] = min_max_scaler.fit_transform(self.cate_embeddings[task_.replace(".json", "")]['embeddings'])
        self.all_accs = {}
        self.unnorm_all_accs = {}
        self.minmax_sc = {}
        for space in self.spaces:
            space = space.replace(".json", "")
            self.all_accs[space] = []
            for idx in range(len(self.space_dicts[space])):
                self.all_accs[space].append(float(100.-self.space_dicts[space][idx]['test_ep_top1'][-1]))
            # RobustScaler normalize this space
            min_max_scaler = preprocessing.QuantileTransformer()
            self.all_accs[space] = min_max_scaler.fit_transform(np.array(self.all_accs[space]).reshape(-1, 1)).flatten()
            self.unnorm_all_accs[space] = np.array(self.all_accs[space]).reshape(-1, 1).flatten().tolist()
            self.all_accs[space] = self.all_accs[space].tolist()
            self.minmax_sc[space] = min_max_scaler
        self.unnorm_accs = self.all_accs # need to comment out this line.
        self.all_accs = self.all_accs

    #################### Key Functions Begin ###################
    def get_adjmlp_zcp(self, idx, space="Amoeba"):
        adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = self.get_adj_op(idx, space=space).values()
        adj_mat_norm = np.asarray(adj_mat_norm).flatten()
        adj_mat_red = np.asarray(adj_mat_red).flatten()
        op_mat_norm = torch.Tensor(np.asarray(op_mat_norm)).argmax(dim=1).numpy().flatten() # Careful here.
        op_mat_red = torch.Tensor(np.asarray(op_mat_red)).argmax(dim=1).numpy().flatten() # Careful here.
        op_mat_norm = op_mat_norm/np.max(op_mat_norm)
        op_mat_red = op_mat_red/np.max(op_mat_red)
        return np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red, np.asarray(self.get_zcp(idx, space)).flatten())).tolist()

    def get_a2vcatezcp(self, idx, space="Amoeba", joint=None):
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
    
    def get_zcp(self, idx, space="Amoeba", joint=None):
        return list(self.zcp_nds_norm[space][idx].values())
    
    def get_adj_op(self, idx, space="Amoeba", bin_space=None):
        return self.space_adj_mats[space][str(idx)]
    
    def get_cate(self, idx, space="Amoeba", joint=None):
        return self.cate_embeddings[space]['embeddings'][idx].tolist()
    
    def get_arch2vec(self, idx, space="Amoeba", joint=None):
        return self.arch2vec_embeddings[space][idx]['feature'].tolist()
    
    def get_valacc(self, idx, space="Amoeba"):
        return self.unnorm_all_accs[space][idx]
    
    def get_numitems(self, space="Amoeba"):
        return len(self.space_dicts[space])

    def get_norm_w_d(self, idx, space="Amoeba"):
        try:
            return [self.space_dicts[space][idx]['net']['width']/32., \
                    self.space_dicts[space][idx]['net']['depth']/20.]
        except:
            print("WARNING: No width/depth information found for idx: ", idx, ",", space)
            exit(0)
    ##################### Key Functions End #####################

    def get_flops(self, idx, space="Amoeba"):
        return self.space_dicts[space][idx]["flops"]

    def get_params(self, idx, space="Amoeba"):
        return self.space_dicts[space][idx]["params"]
    