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

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'

class TransNASBench101Micro:
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True, embedding_list = ['adj',
                                                                    'adj_op',
                                                                    'path',
                                                                    'one_hot',
                                                                    'path_indices',
                                                                    'zcp']):
        if path is None:
            path = ""
        sspaces = ['autoencoder','class_object','class_scene','jigsaw','room_layout','segmentsemantic']
        self.cate_embeddings = {k: torch.load(BASE_PATH + 'cate_embeddings/cate_transnasbench101_{}.pt'.format(k)) for k in sspaces}
        self.arch2vec_embeddings = {k: torch.load(BASE_PATH + 'arch2vec_embeddings/arch2vec-model-dim_32_search_space_transnasbench101_task_{}-tb101.pt'.format(k)) for k in sspaces}
        # self.cate_embeddings = {}
        self.zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']
        self.zcp_tb101 = json.load(open(BASE_PATH + "zc_transbench101_micro.json", "r"))
        self.unnorm_zcp_tb101 = json.load(open(BASE_PATH + "zc_transbench101_micro.json", "r"))
        self.valaccs = {}
        self.unnorm_valaccs = {}
        for task_ in self.zcp_tb101.keys():
            valacc_frame = pd.DataFrame({vec: self.zcp_tb101[task_][vec]['val_accuracy'] for vec in self.zcp_tb101[task_].keys()}, index=[0]).T
            # MinMax normalize the valacc_frame using sklearn preprocessing
            self.unnorm_valaccs[task_] = valacc_frame.to_dict()[0]
            valacc_frame_norm = pd.DataFrame(preprocessing.minmax_scale(valacc_frame), index=valacc_frame.index, columns=valacc_frame.columns)
            valacc_frame_norm = valacc_frame_norm.to_dict()[0]
            self.valaccs[task_] = valacc_frame_norm
        self.normalize_zcp = normalize_zcp
        self.zcp_tb101_norm = {}
        if normalize_zcp:
            print("Normalizing ZCP dict")
            for task_ in self.zcp_tb101.keys():
                print("normalizing task: ", task_)
                self.norm_zcp = pd.DataFrame({k0: {k1: v1["score"] for k1,v1 in v0.items() if v1.__class__()=={}} for k0, v0 in self.zcp_tb101[task_].items()}).T
                if log_synflow:
                    self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(0, 1e-2)
                    self.norm_zcp['synflow'] = np.log10(self.norm_zcp['synflow'])
                else:
                    print("WARNING: Not taking log of synflow values for normalization results in very small synflow inputs")
                minfinite = self.norm_zcp['synflow'].replace(-np.inf, 1000).min()
                self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(-np.inf, minfinite + 1e-2)
                # Normalize each column of self.norm_zcp
                min_max_scaler = preprocessing.MinMaxScaler()
                self.norm_zcp = pd.DataFrame(min_max_scaler.fit_transform(self.norm_zcp), columns=self.norm_zcp.columns, index=self.norm_zcp.index)
                self.zcp_tb101_norm[task_] = self.norm_zcp.T.to_dict()
            self.zcp_tb101 = self.zcp_tb101_norm
        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.CONV3X3 = 'conv3x3'
        self.CONV1X1 = 'conv1x1'
        self.ZEROIZE = 'zeroize'
        self.SKIP = 'skip_connect'
        self._opname_to_index = {
            'none': 0,
            'skip_connect': 1,
            'conv1x1': 2,
            'conv3x3': 3,
            'input': 4,
            'output': 5
        }
        self._index_to_opname = {v: k for k, v in self._opname_to_index.items()}
        self.OPS = [self.ZEROIZE, self.SKIP, self.CONV1X1, self.CONV3X3]
        self.init_op_map = {0: self.ZEROIZE, 1: self.SKIP, 2: self.CONV1X1, 3: self.CONV3X3}
        self.hash_iterator_list = list(self.zcp_tb101['normal'].keys())
        
    #################### Key Functions Begin ###################
    def get_adjmlp_zcp(self, idx, task=None):
        adj_mat, op_mat = self.get_adj_op(idx, task).values()
        adj_mat = np.asarray(adj_mat).flatten()
        op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten()
        op_mat = op_mat/np.max(op_mat)
        return np.concatenate([adj_mat, op_mat, np.asarray(self.get_zcp(idx, task))]).tolist()

    def get_adj_op(self, idx, task=None, space=None, bin_space=None):
        task = 'class_scene' if task==None else task
        hash = self.hash_iterator_list[idx]
        ops = self.opslist_onehot(eval(hash))
        return {'module_adjacency': self.get_matrix_and_ops(eval(self.hash_iterator_list[idx]))[0], 'module_operations': ops.tolist()}
    

    def get_a2vcatezcp(self, idx, task=None, joint=None, space=None):
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

    def get_zcp(self, idx, task=None, joint=None, space=None):
        task = 'class_scene' if task==None else task
        hash = self.hash_iterator_list[idx]
        return list(self.zcp_tb101[task][hash].values())
    
    def get_valacc(self, idx, task=None, space=None):
        task = 'class_scene' if task==None else task
        hash = self.hash_iterator_list[idx]
        # return self.valaccs[task][hash]
        return self.unnorm_valaccs[task][hash]
    
    def get_arch2vec(self, idx, task=None, joint=None, space=None):
        task = 'class_scene' if task==None else task
        return self.arch2vec_embeddings[task][idx]['feature'].tolist()
    
    def get_cate(self, idx, task=None, joint=None, space=None):
        task = 'class_scene' if task==None else task
        return self.cate_embeddings[task]['embeddings'][idx].tolist()
    
    def get_numitems(self, space=None):
        return len(self.hash_iterator_list) 
    
    def get_norm_w_d(self, idx, space=None):
        return [0, 0]
    ##################### Key Functions End #####################
    def opslist_onehot(self, op_list):
        op_map = [self.OUTPUT, self.INPUT, *self.OPS]
        ops_ = [self.init_op_map[x] for x in op_list]
        ops_ = [self.INPUT, *ops_, self.OUTPUT]
        ops_onehot = np.array([[i == op_map.index(op) for i in range(len(op_map))] for op in ops_], dtype=np.float32)
        return ops_onehot
    
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
    
    
    def get_params(self, idx, task=None):
        if task == None:
            task = 'class_scene'
        return self.unnorm_zcp_tb101[task][self.hash_iterator_list[idx]]['params']['score']*1e5
        