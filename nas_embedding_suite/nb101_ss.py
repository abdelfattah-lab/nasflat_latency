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
from nb123.nas_bench_101.cell_101 import Cell101
from nasbench import api as NB1API

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'

class NASBench101:
    def __init__(self, path=None, zcp_dict=False, normalize_zcp=True, log_synflow=True, 
                 embedding_list=['adj', 'adj_op', 'cat_adj', 'cont_adj', 'path', 'cat_path', 'trunc_path',
                                 'trunc_cat_path', 'zcp', 'gcn', 'vae', 'arch2vec', 'cate']):
        self.zcps = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']
        self.nb1_op_idx = {'input': 0, 'output': 1, 'conv3x3-bn-relu': 2, 'conv1x1-bn-relu': 3, 'maxpool3x3': 4}
        self.zcp_dict = zcp_dict

        if path is None:
            path = ''

        self.load_files(path)
        self.normalize_and_process_zcp(normalize_zcp, log_synflow)
        self.create_hash_to_idx()
        # check if saved_valacc.pkl exists, if not, create it
        if not os.path.exists(BASE_PATH + 'saved_valacc.pkl'):
            valacc_list = [
                sum([
                    self.nb1_api.get_metrics_from_hash(hash_)[1][108][ixe]['final_validation_accuracy'] for ixe in range(3)
                    ])/3 for hash_ in tqdm(self.nb1_api.hash_iterator())
                ]
            torch.save(valacc_list, BASE_PATH + 'saved_valacc.pkl')
        else:
            valacc_list = torch.load(BASE_PATH + 'saved_valacc.pkl')
        self.unnorm_valacc_list = valacc_list
        # MinMax normalize valacc_list
        self.min_max_scaler = preprocessing.QuantileTransformer()
        self.valacc_list = self.min_max_scaler.fit_transform(np.array(valacc_list).reshape(-1, 1)).reshape(-1)
        
    #################### Key Functions Begin ###################

    def get_adjmlp_zcp(self, idx):
        hash = self.hash_iterator_list[idx]
        metrics_hashed = self.nb1_api.get_metrics_from_hash(hash)
        matrix = metrics_hashed[0]['module_adjacency']
        ops = metrics_hashed[0]['module_operations']
        matrix, ops = self.pad_size_6(matrix, ops)
        module_operations = self.transform_nb101_operations(ops)
        adj_mat = np.asarray(matrix).flatten()
        op_mat = torch.Tensor(np.asarray(module_operations)).argmax(dim=1).numpy().flatten()
        op_mat = op_mat/np.max(op_mat)
        return np.concatenate((adj_mat, op_mat, np.asarray(self.get_zcp(idx)))).tolist()

    def get_adj_op(self, idx, space=None, bin_space=None):
        hash = self.hash_iterator_list[idx]
        metrics_hashed = self.nb1_api.get_metrics_from_hash(hash)
        matrix = metrics_hashed[0]['module_adjacency']
        ops = metrics_hashed[0]['module_operations']
        matrix, ops = self.pad_size_6(matrix, ops)
        module_operations = self.transform_nb101_operations(ops)
        return {'module_adjacency': matrix.tolist(), 'module_operations': module_operations.tolist()}

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
    
    def get_arch2vec(self, idx, joint=None, space=None):
        return self.arch2vec_nb101[idx]['feature'].tolist()

    def get_cate(self, idx, joint=None, space=None):
        return self.cate_nb101['embeddings'][idx].tolist()

    def get_zcp(self, idx, joint=None, space=None):
        hash = self.hash_iterator_list[idx]
        metrics = self.nb1_api.get_metrics_from_hash(hash)
        mat_adj = np.asarray(metrics[0]['module_adjacency']).flatten().tolist()
        mat_op = [self.nb1_op_idx[x] for x in metrics[0]['module_operations']]
        mat = mat_adj + mat_op
        mat_repr = str(tuple(mat))
        return [self.zcp_nb101['cifar10'][mat_repr][nn] for nn in self.zcps]

    def get_valacc(self, idx, space=None, normalized=True):
        valacc = self.valacc_list[idx]
        if not normalized:
            valacc = self.unnorm_valacc_list[idx]
        return valacc
    
    def get_norm_w_d(self, idx, space=None):
        return [0, 0]
    
    def get_numitems(self, space=None):
        return len(self.hash_iterator_list)
    
    ##################### Key Functions End #####################
    def load_files(self, path):
        print("Loading Files for NASBench101...")
        start_time = time.time()

        self.cate_nb101 = torch.load(BASE_PATH + "cate_embeddings/cate_nasbench101.pt")
        self.arch2vec_nb101 = torch.load(BASE_PATH + "arch2vec_embeddings/arch2vec-model-dim_32_search_space_nasbench101-nasbench101.pt")
        self.zcp_nb101 = json.load(open(BASE_PATH + "zc_nasbench101_full.json", "r"))
        self.nb1_api = NB1API.NASBench(BASE_PATH + 'nasbench_only108_caterec.tfrecord')

        print("Loaded files in: ", time.time() - start_time, " seconds")

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
                                          for k0, v0 in self.zcp_nb101['cifar10'].items()}).T

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
            self.norm_zcp['val_accuracy'] = self.min_max_scaling(self.norm_zcp['val_accuracy'])

            self.zcp_nb101 = {'cifar10': self.norm_zcp.T.to_dict()}
        
    # def normalize_and_process_zcp(self, normalize_zcp, log_synflow):
    #     if normalize_zcp:
    #         print("Normalizing ZCP dict")
    #         self.norm_zcp = pd.DataFrame({k0: {k1: v1["score"] for k1, v1 in v0.items() if v1.__class__() == {}} 
    #                                       for k0, v0 in self.zcp_nb101['cifar10'].items()}).T

    #         if log_synflow:
    #             # loglist = ['synflow', 'fisher', 'flops', 'grad_norm', ]
    #             self.norm_zcp['synflow'] = np.log1p(self.norm_zcp['synflow'])
    #             # self.norm_zcp['fisher'] = np.log1p(self.norm_zcp['fisher'])
    #             # self.norm_zcp['flops'] = np.log1p(self.norm_zcp['flops'])
    #             # self.norm_zcp
    #         else:
    #             print("WARNING: Not taking log of synflow values for normalization results in very small synflow inputs")

    #         minfinite = self.norm_zcp['synflow'].replace(-np.inf, 1000).min()
    #         self.norm_zcp['synflow'] = self.norm_zcp['synflow'].replace(-np.inf, minfinite + 1e-2)

    #         # Normalize each column of self.norm_zcp
    #         min_max_scaler = preprocessing.MinMaxScaler()
    #         self.norm_zcp = pd.DataFrame(min_max_scaler.fit_transform(self.norm_zcp), columns=self.norm_zcp.columns, index=self.norm_zcp.index)
    #         self.zcp_nb101 = {'cifar10': self.norm_zcp.T.to_dict()}

    def create_hash_to_idx(self):
        self.hash_iterator_list = list(self.nb1_api.hash_iterator())
        self.hash_to_idx = {hash: idx for idx, hash in enumerate(self.hash_iterator_list)}

    def get_params(self, idx):
        return self.nb1_api.get_metrics_from_hash(self.hash_iterator_list[idx])[0]['trainable_parameters']
    
    def transform_nb101_operations(self, ops):
        transform_dict = {'input': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'output': 4}
        ops_array = np.zeros([7, 5], dtype='int8')
        for row, op in enumerate(ops):
            col = transform_dict[op]
            ops_array[row, col] = 1
        return ops_array

    def pad_size_6(self, matrix, ops):
        if len(matrix) < 7:
            new_matrix, new_ops = self.create_padded_matrix_and_ops(matrix, ops)
            return new_matrix, new_ops
        else:
            return matrix, ops

    def create_padded_matrix_and_ops(self, matrix, ops):
        new_matrix = np.zeros((7, 7), dtype='int8')
        new_ops = []
        n = matrix.shape[0]

        for i in range(7):
            for j in range(7):
                if j < n - 1 and i < n:
                    new_matrix[i][j] = matrix[i][j]
                elif j == n - 1 and i < n:
                    new_matrix[i][-1] = matrix[i][j]

        for i in range(7):
            if i < n - 1:
                new_ops.append(ops[i])
            elif i < 6:
                new_ops.append('conv3x3-bn-relu')
            else:
                new_ops.append('output')

        return new_matrix, new_ops
    
    def index_to_embedding(self, idx):
        hash = self.hash_iterator_list[idx]
        metrics_hashed = self.nb1_api.get_metrics_from_hash(hash)
        matrix = metrics_hashed[0]['module_adjacency']
        ops = metrics_hashed[0]['module_operations']
        matrix, ops = self.pad_size_6(matrix, ops)
        module_operations = self.transform_nb101_operations(ops)
        zcp_op = [self.nb1_op_idx[m] for m in ops]
        zcp_stringarg = str(tuple(matrix.flatten().tolist() + zcp_op))
        cellobj = Cell101(matrix, ops)

        output_dict = self.create_embedding_output_dict(cellobj, matrix, module_operations, idx, zcp_stringarg)
        return output_dict

    def create_embedding_output_dict(self, cellobj, matrix, module_operations, idx, zcp_stringarg):
        predictor_encodings = ['adj', 'adj_op', 'cat_adj', 'cont_adj', 'path', 'cat_path', 'trunc_path', 'trunc_cat_path', 'gcn', 'vae', 'arch2vec', 'cate']
        output_dict = {}

        for encoding in predictor_encodings:
            try:
                getter_function = getattr(self, f'get_{encoding}_value')
                output_dict[encoding] = getter_function(cellobj, matrix, module_operations, idx)
            except Exception as e:
                print(f"Probably {encoding} matrix mismatch: ", e)
                output_dict[encoding] = None

        output_dict['zcp'] = self.get_zcp(zcp_stringarg)
        output_dict['valacc'] = self.get_valacc(idx)
        return output_dict
