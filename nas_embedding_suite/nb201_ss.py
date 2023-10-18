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
from nb123.nas_bench_201.cell_201 import Cell201
from nas_201_api import NASBench201API as NB2API

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'


class NASBench201:
    CACHE_FILE_PATH = BASE_PATH + "/nb201_cellobj_cache.pkl"  # Adjust this to your preferred path
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True, embedding_list = [ 'adj',
                                                                    'adj_op',
                                                                    'path',
                                                                    'trunc_path',
                                                                    'gcn',
                                                                    'path_indices',
                                                                    'one_hot',
                                                                    'zcp',
                                                                    'arch2vec',
                                                                    'valacc']):
        if path==None:
            path = ''
        print("Loading files for NASBench201...")
        a = time.time()
        self.zcp_dict = zcp_dict
        self.arch2vec_nb201 = torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_nasbench201-nasbench201.pt")
        self.cate_nb201 = torch.load(BASE_PATH + "cate_embeddings/cate_nasbench201.pt")
        self.zcp_nb201 = json.load(open(BASE_PATH + "zc_nasbench201.json", "r"))
        self.zcp_nb201_valacc = json.load(open(BASE_PATH + "zc_nasbench201.json", "r"))
        self.zcp_nb201_valacc = {k: v['val_accuracy'] for k,v in self.zcp_nb201_valacc['cifar10'].items()}
        valacc_frame = pd.DataFrame(self.zcp_nb201_valacc, index=[0]).T
        self.valacc_frame = valacc_frame
        self.zcp_unnorm_nb201_valacc = pd.DataFrame(valacc_frame, columns=valacc_frame.columns, index=valacc_frame.index).to_dict()[0]
        # MinMax normalize the valacc_frame using sklearn preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        self.zcp_nb201_valacc = pd.DataFrame(min_max_scaler.fit_transform(valacc_frame), columns=valacc_frame.columns, index=valacc_frame.index).to_dict()[0]
        self.normalize_zcp = normalize_zcp
        self.normalize_and_process_zcp(normalize_zcp, log_synflow)
        self._opname_to_index = {
                    'none': 0,
                    'skip_connect': 1,
                    'nor_conv_1x1': 2,
                    'nor_conv_3x3': 3,
                    'avg_pool_3x3': 4,
                    'input': 5,
                    'output': 6,
                    'global': 7
                }
        self._index_to_opname = {v: k for k, v in self._opname_to_index.items()}
        self.nb2_api  = NB2API(BASE_PATH + "NAS-Bench-201-v1_1-096897.pth")
        print("Loaded files in: ", time.time() - a, " seconds")
        self.zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']    
        self.cache = {}
        self.cready = False
        # Check if the cache file exists
        if os.path.exists(NASBench201.CACHE_FILE_PATH):
            print("Loading cache for NASBench-201 speedup!!...")
            self.cready = True
            with open(NASBench201.CACHE_FILE_PATH, 'rb') as cache_file:
                self.cache = pickle.load(cache_file)
        else:
            # If the cache file doesn't exist, populate the cache for all idx values
            self.cache = {}
            print("Populating cache for NASBench-201 speedup!!...")
            for idx in tqdm(range(15625)):  # Assuming your range is [0, 15625]
                self.cache[idx] = {
                    'adjmlp_zcp': self.get_adjmlp_zcp(idx),
                    'adj_op': self.get_adj_op(idx),
                    'zcp': self.get_zcp(idx),
                    'acc': self.get_valacc(idx),
                }
            
            # Now save the populated cache
            with open(NASBench201.CACHE_FILE_PATH, 'wb') as cache_file:
                pickle.dump(self.cache, cache_file)
            self.cready = True
        
        latency_data = {}
        devices = ['1080ti_1', '1080ti_256', '1080ti_32', '2080ti_1', '2080ti_256', '2080ti_32', 'desktop_cpu_core_i7_7820x_fp32', 'desktop_gpu_gtx_1080ti_fp32',      \
                   'embedded_gpu_jetson_nano_fp16', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'essential_ph_1', 'eyeriss', 'flops_nb201_cifar10', \
                   'fpga', 'gold_6226', 'gold_6240', 'mobile_cpu_snapdragon_450_cortex_a53_int8', 'mobile_cpu_snapdragon_675_kryo_460_int8', 'mobile_cpu_snapdragon_855_kryo_485_int8', \
                   'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_450_adreno_506_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', \
                   'mobile_gpu_snapdragon_855_adreno_640_int8', 'nwot_nb201_cifar10', 'params_nb201_cifar10', 'pixel2', 'pixel3', 'raspi4', 'samsung_a50', 'samsung_s7', 'silver_4114', \
                   'silver_4210r', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'titanx_1', 'titanx_256', 'titanx_32', 'titanxp_1', 'titanxp_256', 'titanxp_32']
        for dev_ in devices:
            latency_data[dev_] = torch.load(BASE_PATH + "/nb201_latency/" + dev_ + ".pt")
        
        # Normalize each key in latency_data
        for k, v in latency_data.items():
            latency_data[k] = self.min_max_scaling(np.asarray(v)).tolist()
        self.latency_data = latency_data
        self.zready = False
        self.zcp_cache = {}
        
        if os.path.exists(NASBench201.CACHE_FILE_PATH.replace("cellobj", "zcp")):
            print("Loading cache for NASBench-201 speedup!!...")
            self.zready = True
            with open(NASBench201.CACHE_FILE_PATH.replace("cellobj", "zcp"), 'rb') as cache_file:
                self.zcp_cache = pickle.load(cache_file)
        if not self.zready:
            for idx in range(15625):
                self.zcp_cache[idx] = self.get_zcp(idx)
            with open(NASBench201.CACHE_FILE_PATH.replace("cellobj", "zcp"), 'wb') as cache_file:
                pickle.dump(self.zcp_cache, cache_file)
            self.zready = True
                
    def min_max_scaling(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def log_transform(self, data):
        return np.log1p(data)

    def standard_scaling(self, data):
        return (data - np.mean(data)) / np.std(data)

    def normalize_and_process_zcp(self, normalize_zcp, log_synflow):
        if normalize_zcp:
            print("Normalizing ZCP dict")
            self.norm_zcp = pd.DataFrame({k0: {k1: v1["score"] for k1, v1 in v0.items() if v1.__class__() == {}} 
                                          for k0, v0 in self.zcp_nb201['cifar10'].items()}).T

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
            # self.norm_zcp['val_accuracy'] = self.min_max_scaling(self.norm_zcp['val_accuracy'])

            self.zcp_nb201 = {'cifar10': self.norm_zcp.T.to_dict()}
    #################### Key Functions Begin ###################
    
    def get_adjmlp_zcp(self, idx):
        # Check if result exists in cache
        if self.cready:
            return self.cache[idx]['adjmlp_zcp']
        else:
            arch_str = self.nb2_api.query_by_index(idx).arch_str
            cellobj = Cell201(arch_str)
            gcn_encoding = cellobj.gcn_encoding(self.nb2_api, deterministic=True)
            arch_vector = self.get_arch_vector_from_arch_str(arch_str)
            matrix = self.get_matrix_and_ops(arch_vector)[0]
            op_mat = gcn_encoding['operations'].tolist()
            adj_mat = np.asarray(matrix).flatten()
            op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten()
            op_mat = op_mat/np.max(op_mat)
            return np.concatenate([adj_mat, op_mat, np.asarray(self.get_zcp(idx))]).tolist()


    def get_a2vcatezcp(self, idx, joint=None, space=None):
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
    
    def get_adj_op(self, idx, space=None, bin_space=None):
        if self.cready:
            return self.cache[idx]['adj_op']
        else:
            arch_str = self.nb2_api.query_by_index(idx).arch_str
            cellobj = Cell201(arch_str)
            gcn_encoding = cellobj.gcn_encoding(self.nb2_api, deterministic=True)
            arch_vector = self.get_arch_vector_from_arch_str(arch_str)
            return {'module_adjacency': self.get_matrix_and_ops(arch_vector)[0], 'module_operations': gcn_encoding['operations'].tolist()}


    def get_zcp(self, idx, joint=None, space=None):
        # Check if result exists in cache
        if self.zready:
            return self.zcp_cache[idx]
        else:
            arch_str = self.nb2_api.query_by_index(idx).arch_str
            cellobj = Cell201(arch_str)
            zcp_key = str(tuple(cellobj.encode(predictor_encoding='adj')))
            return [self.zcp_nb201['cifar10'][zcp_key][nn] for nn in self.zcps]

    def get_numitems(self, space=None):
        return 15625
    
    def get_norm_w_d(self, idx, space=None):
        return [0, 0]
    
    def get_valacc(self, idx, space=None):
        if self.cready:
            return self.cache[idx]['acc']
        else:
            arch_str = self.nb2_api.query_by_index(idx).arch_str
            arch_index = self.nb2_api.query_index_by_arch(arch_str)
            # acc_results = self.nb2_api.query_by_index(arch_index, 'cifar10-valid', use_12epochs_result=False)
            try:
                acc_results = sum([self.nb2_api.get_more_info(arch_index, 'cifar10-valid', None,
                                                        use_12epochs_result=False,
                                                        is_random=seed)['valid-accuracy'] for seed in [777, 888, 999]])/3.
                val_acc = acc_results['valid-accuracy'] / 100.
            except:
                # some architectures only contain 1 seed result
                acc_results = self.nb2_api.get_more_info(arch_index, 'cifar10-valid', None,
                                                        use_12epochs_result=False,
                                                        is_random=False)['valid-accuracy'] 
                val_acc = acc_results / 100.
            return val_acc
        
    def get_latency(self, idx, space=None, device="1080ti_1"):
        return self.latency_data[device][idx]

    def get_arch2vec(self, idx, joint=None, space=None):
        return self.arch2vec_nb201[idx]['feature'].tolist()

    def get_cate(self, idx, joint=None, space=None):
        return self.cate_nb201['embeddings'][idx].tolist()
    ##################### Key Functions End #####################
    
    def get_matrix_and_ops(self, g, prune=True, keep_dims=True):
        ''' Return the adjacency matrix and label vector.

            Args:
                g : should be a point from Nasbench201 search space
                prune : remove dangling nodes that only connected to zero ops
                keep_dims : keep the original matrix size after pruning
        '''
        n_nodes = 8
        zero_id = 0
        skip_id = 1
        matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        labels = [None for _ in range(n_nodes)]
        labels[0] = 'input'
        labels[-1] = 'output'
        matrix[0][1] = matrix[0][2] = matrix[0][4] = 1
        matrix[1][3] = matrix[1][5] = 1
        matrix[2][6] = 1
        matrix[3][6] = 1
        matrix[4][7] = 1
        matrix[5][7] = 1
        matrix[6][7] = 1
        for idx, op in enumerate(g):
            if op == zero_id: # zero
                for other in range(n_nodes):
                    if matrix[other][idx+1]:
                        matrix[other][idx+1] = 0
                    if matrix[idx+1][other]:
                        matrix[idx+1][other] = 0
            elif op == skip_id: # skip-connection:
                to_del = []
                for other in range(n_nodes):
                    if matrix[other][idx+1]:
                        for other2 in range(n_nodes):
                            if matrix[idx+1][other2]:
                                matrix[other][other2] = 1
                                matrix[other][idx+1] = 0
                                to_del.append(other2)
                for d in to_del:
                    matrix[idx+1][d] = 0
            else:
                labels[idx+1] = self._index_to_opname[op]
        if prune:
            visited_fw = [False for _ in range(n_nodes)]
            visited_bw = copy.copy(visited_fw)
            def bfs(beg, vis, con_f):
                q = [beg]
                vis[beg] = True
                while q:
                    v = q.pop()
                    for other in range(n_nodes):
                        if not vis[other] and con_f(v, other):
                            q.append(other)
                            vis[other] = True
            bfs(0, visited_fw, lambda src, dst: matrix[src][dst]) # forward
            bfs(n_nodes-1, visited_bw, lambda src, dst: matrix[dst][src]) # backward
            for v in range(n_nodes-1, -1, -1):
                if not visited_fw[v] or not visited_bw[v]:
                    labels[v] = None
                    if keep_dims:
                        matrix[v] = [0] * n_nodes
                    else:
                        del matrix[v]
                    for other in range(len(matrix)):
                        if keep_dims:
                            matrix[other][v] = 0
                        else:
                            del matrix[other][v]
            if not keep_dims:
                labels = list(filter(lambda l: l is not None, labels))
            assert visited_fw[-1] == visited_bw[0]
            assert visited_fw[-1] == False or matrix
            verts = len(matrix)
            assert verts == len(labels)
            for row in matrix:
                assert len(row) == verts
        return matrix, labels
    
    def get_arch_vector_from_arch_str(self, arch_str):
        ''' Args:
                arch_str : a string representation of a cell architecture,
                    for example '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
        '''

        nodes = arch_str.split('+')
        nodes = [node[1:-1].split('|') for node in nodes]
        nodes = [[op_and_input.split('~')[0]  for op_and_input in node] for node in nodes]
        _opname_to_index = {
            'none': 0,
            'skip_connect': 1,
            'nor_conv_1x1': 2,
            'nor_conv_3x3': 3,
            'avg_pool_3x3': 4,
            'input': 5,
            'output': 6,
            'global': 7
        }
        # arch_vector is equivalent to a decision vector produced by autocaml when using Nasbench201 backend
        arch_vector = [_opname_to_index[op] for node in nodes for op in node]
        return arch_vector
    
    def get_params(self, idx):
        return self.nb2_api.get_cost_info(idx, dataset='cifar10-valid')['params']*10000000