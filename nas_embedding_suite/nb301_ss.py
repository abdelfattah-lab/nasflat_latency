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
from nb123.nas_bench_301.cell_301 import Cell301
from nb123.nb301_arch_tuple_to_nb101 import convert_arch_tuple_to_idx
import nasbench301 as nb3_api

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'

class NASBench301:
    def __init__(self, use_nb3_performance_model=False, zcp_dict=False, normalize_zcp=True, log_synflow=True,
                 path=None, zcp=False, embedding_list = [ 'adj', 
                                                        'adj_op',
                                                        'paths', 
                                                        'path_indices', 
                                                        'genotypes', 
                                                        'cate', 
                                                        'zcp', 
                                                        'arch2vec',
                                                        'valacc']):
        if path==None:
            path = ''
        print("Loading Files for NASBench301...")
        a = time.time()
        self.noisy_nb3 = False

        self.use_nb3_performance_model = use_nb3_performance_model
        self.ensemble_dir_performance = BASE_PATH + "nb_models_0.9/xgb_v0.9"
        self.cate_nb301 = torch.load(BASE_PATH + "cate_embeddings/cate_nasbench301.pt")
        self.nb301_proxy_cate = None
        self.arch2vec_nb301 = torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_nasbench301-nasbench301.pt")

        self.nb3_api = nb3_api
        if self.use_nb3_performance_model:
            self.performance_model = nb3_api.load_ensemble(self.ensemble_dir_performance)

        print("Loaded files in: ", time.time() - a, " seconds")

        
        self.op_dict = {
            0: 'max_pool_3x3',
            1: 'avg_pool_3x3',
            2: 'skip_connect',
            3: 'sep_conv_3x3',
            4: 'sep_conv_5x5',
            5: 'dil_conv_3x3',
            6: 'dil_conv_5x5'
            }
        
        self.op_dict_rev = {v: k for k, v in self.op_dict.items()}
    #################### Key Functions Begin ###################
    def get_adjmlp_zcp(self, idx):
        adj_mat, op_mat = self.get_adj_op(idx)
        adj_mat = np.asarray(adj_mat).flatten()
        op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten()
        op_mat = op_mat/np.max(op_mat)
        return np.concatenate([adj_mat, op_mat, np.asarray(self.get_zcp(idx))]).tolist()

    def get_adj_op(self, idx, space=None, bin_space=None):
        cate_nb301_arch = self.cate_nb301['genotypes'][idx]
        arch_desc = {'arch': ([(int(x[0]), self.op_dict_rev[x[1]]) for x in cate_nb301_arch],[(int(x[0]), self.op_dict_rev[x[1]]) for x in cate_nb301_arch])}
        return convert_arch_tuple_to_idx([(str(x[0]), self.op_dict[x[1]]) for x in arch_desc['arch'][0]])
    
    def get_arch2vec(self, idx, joint=None, space=None):
        return self.arch2vec_nb301[idx]['feature'].tolist()
    
    def get_cate(self, idx, joint=None, space=None):
        return self.cate_nb301['embeddings'][idx].tolist()
    
    def get_valacc(self, idx, space=None, use_nb3_performance_model=False):
        if use_nb3_performance_model:
            cate_nb301_arch = self.cate_nb301['genotypes'][idx]
            arch_desc = {'arch': ([(int(x[0]), self.op_dict_rev[x[1]]) for x in cate_nb301_arch],[(int(x[0]), self.op_dict_rev[x[1]]) for x in cate_nb301_arch])}
            cate_nb301_arch = Cell301(None).convert_to_genotype(**arch_desc)
            return self.performance_model.predict(config=cate_nb301_arch, representation="genotype", with_noise=self.noisy_nb3)
        else:
            return self.cate_nb301['predicted_accs'][idx]
        
    def get_norm_w_d(self, idx, space=None):
        return [0, 0]
    
    def get_numitems(self, space=None):
        return 1000000
    ##################### Key Functions End #####################
    
    def get_genotype(self, idx):
        return self.index_to_embedding(idx)['genotypes']
    
    def get_params(self, idx):
        if self.nb301_proxy_cate is None:
            with open(BASE_PATH + '/nasbench301_proxy.json', 'r') as f: # load 
                self.nb301_proxy_cate = json.load(f)
        return self.nb301_proxy_cate[str(idx)]["params"]