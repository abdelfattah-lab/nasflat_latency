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
# import pickle
import time
import dill as pickle
import sys

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

from nb101_ss import NASBench101
from nb201_ss import NASBench201
from nb301_ss import NASBench301
from nds_ss import NDS
from tb101_micro_ss import TransNASBench101Micro
import os
import pickle

CACHE_DIR = '/scratch/ya255/emb_cache'
FILE_MAPPINGS = {
    'nb101': (CACHE_DIR + '/nb101.pkl', NASBench101),
    'nb201': (CACHE_DIR + '/nb201.pkl', NASBench201),
    # 'nb301': ('./cache/nb301.pkl', NASBench301),
    'tb101': (CACHE_DIR + '/tb101.pkl', TransNASBench101Micro),
    'nds': (CACHE_DIR + '/nds.pkl', NDS),
}

class AllSS:
    def __init__(self):
        self.ss_mapper = {"nb101": 0, "nb201": 1, "nb301": 2, "Amoeba": 3, "PNAS_fix-w-d": 4, 
                     "ENAS_fix-w-d": 5, "NASNet": 6, "DARTS": 7, "ENAS": 8, "PNAS": 9, 
                     "DARTS_lr-wd": 10, "DARTS_fix-w-d": 11, "tb101": 12}
        self.arch2vec_ranges = {
                        0: "nb101",
                        423624: "nb201",
                        439249: "nb301",
                        1439249: "Amoeba",
                        1444232: "PNAS_fix-w-d",
                        1448791: "ENAS_fix-w-d",
                        1453791: "NASNet",
                        1458637: "DARTS",
                        1463637: "ENAS",
                        1468636: "PNAS",
                        1473635: "DARTS_lr-wd",
                        1478635: "DARTS_fix-w-d",
                        1483635: "tb101",
                    }
        start_ = time.time()
        print("[WARNING]: ALL SS has a cache store at {}, which needs to be changed if reproducing in all_ss.py!!!!".format(CACHE_DIR))
        # self._ensure_cache_exists()
        self.nb301 = NASBench301()
        self._load_classes()
        # check if os.environ["PROJ_BPATH"] + "/embedding_dataset/arch2vec_f_ss.csv", exists
        # if it does not exist, create it
        if not os.path.exists(os.environ["PROJ_BPATH"] + "/nas_embedding_suite/embedding_datasets/arch2vec_f_ss.csv"):
            print("Creating arch2vec_f_ss.csv")
            self.arch2vec_data_dict = torch.load(
                                            os.environ["PROJ_BPATH"]
                                            + "/nas_embedding_suite/embedding_datasets/model-dim_32_search_space_all_ss-all_ss.pt"
                                        )
            self.prep_arch2vec_joint()
        else: # load it
            self.arch2vec_f_ss = pd.read_csv(os.environ["PROJ_BPATH"] + "/nas_embedding_suite/embedding_datasets/arch2vec_f_ss.csv")
        
        if not os.path.exists(os.environ["PROJ_BPATH"] + "/nas_embedding_suite/embedding_datasets/cate_f_ss.csv"):
            self.cate_data_dict = torch.load(
                                            os.environ["PROJ_BPATH"]
                                            + "/nas_embedding_suite/embedding_datasets/cate_all_ss.pt"
                                        )
            self.prep_cate_joint()
        else:
            self.cate_f_ss = pd.read_csv(os.environ["PROJ_BPATH"] + "/nas_embedding_suite/embedding_datasets/cate_f_ss.csv")
        
        self.joint_arch2vec_idxer = {}
        for space in self.arch2vec_ranges.values():
            class_map = self.ss_mapper[space]
            self.joint_arch2vec_idxer[space] = self.arch2vec_f_ss[self.arch2vec_f_ss["label"] == class_map].values[:, :-1]
        self.joint_cate_idxer = {}
        for space in self.arch2vec_ranges.values(): 
            class_map = self.ss_mapper[space]
            self.joint_cate_idxer[space] = self.cate_f_ss[self.cate_f_ss["label"] == class_map].values[:, :-1]
        self.max_oplen = self.get_max_oplen()
        print("Time taken to load all_ss: {}".format(time.time() - start_))
        self.ss_mapper_oprange = {}
        for space, space_idx in self.ss_mapper.items():
            if space in ["nb101", "nb201", "nb301", "tb101"]:
                exec(f'self.ss_mapper_oprange[space] = (np.asarray(self.{space}.get_adj_op(0)["module_operations"]).shape[-1], space_idx)')
            else:
                exec(f'self.ss_mapper_oprange[space] = (np.asarray(self.nds.get_adj_op(0, space="{space}")["normal_ops"]).shape[-1], space_idx)')

    def prep_cate_joint(self):
        features = []
        labels = []
        for key in range(self.cate_data_dict["embeddings"].shape[0]):
            class_idx = None
            for r in sorted(self.arch2vec_ranges):
                if key >= r:
                    class_idx = list(self.arch2vec_ranges.values()).index(self.arch2vec_ranges[r])
            labels.append(class_idx)
        features = np.asarray(self.cate_data_dict["embeddings"])
        labels = np.array(labels)
        self.cate_f_ss = pd.DataFrame(features)
        # add a column for labels
        self.cate_f_ss["label"] = labels
        # Save this dataframe
        self.cate_f_ss.to_csv(os.environ["PROJ_BPATH"] + "/nas_embedding_suite/embedding_datasets/cate_f_ss.csv", index=False)


    def prep_arch2vec_joint(self):
        features = []
        labels = []
        for key, val in self.arch2vec_data_dict.items():
            feature_val = val["feature"]
            class_idx = None
            for r in sorted(self.arch2vec_ranges):
                if key >= r:
                    class_idx = list(self.arch2vec_ranges.values()).index(self.arch2vec_ranges[r])
            labels.append(class_idx)
            features.append(feature_val.tolist())
        features = np.array(features)
        labels = np.array(labels)
        self.arch2vec_f_ss = pd.DataFrame(features)
        # add a column for labels
        self.arch2vec_f_ss["label"] = labels
        # Save this dataframe
        self.arch2vec_f_ss.to_csv(os.environ["PROJ_BPATH"] + "/nas_embedding_suite/embedding_datasets/arch2vec_f_ss.csv", index=False)
    
    def get_numitems(self, space=None, task=None):
        # assert self.arch2vec_f_ss.shape[0] == self.cate_f_ss.shape[0], "Number of Archs embedded in Arch2Vec and CATE should be the same!"
        return self.arch2vec_f_ss.shape[0]
    
    def get_ss_idxrange(self, space):
        class_idx = self.ss_mapper[space]
        # from arch2vec_f_ss, get the indices where label == class_idx
        idxs = self.arch2vec_f_ss[self.arch2vec_f_ss["label"] == class_idx].index.tolist()
        return idxs

    def pad_operation_matrix(self, op_matrix, space_name):
        # Calculate the total number of operations across all search spaces
        total_ops = sum([x[0] for x in self.ss_mapper_oprange.values()])
        
        # Create a zero matrix of size (num_edges, total_number_of_operations)
        padded_matrix = np.zeros((op_matrix.shape[0], total_ops))
        
        # Get the number of operations and space index for the given search space
        num_ops, space_idx = self.ss_mapper_oprange[space_name]
        
        # Calculate the start and end column indices for the operation matrix
        start_idx = sum([x[0] for _, x in sorted(self.ss_mapper_oprange.items(), key=lambda y: y[1]) if x[1] < space_idx])
        end_idx = start_idx + num_ops
        
        # Fill in the appropriate columns of the zero matrix
        padded_matrix[:, start_idx:end_idx] = op_matrix
        
        return padded_matrix


    def get_adj_op(self, idx, space, bin_space=False):
        if bin_space:
            if space in ["nb101", "nb201", "nb301", "tb101"]:
                adj_op_mat = eval("self." + space).get_adj_op(idx)
                opmat = np.asarray(adj_op_mat["module_operations"])
                # opmat will have a shape like 7 x 5
                # Convert it into 7 x max_oplen with leading zero padding
                padded_opmat = np.zeros((opmat.shape[0], self.max_oplen))
                for i in range(opmat.shape[0]):
                    padded_opmat[i, -opmat.shape[1]:] = opmat[i]
                # ss_pad will have a 1 x 4 opmat will have a shape 7 x max_oplen
                # replicate ss_pad on each row of opmat to make it 7 x (max_oplen + 4)
                ss_pad = self.ss_to_binary(space) 
                final_mat = np.hstack([padded_opmat, np.tile(ss_pad, (padded_opmat.shape[0], 1))])
                new_adj_op_mat = copy.deepcopy(adj_op_mat)
                new_adj_op_mat["module_operations"] = final_mat.tolist()
            else:
                adj_op_mat = self.nds.get_adj_op(idx, space=space)
                new_adj_op_mat = copy.deepcopy(adj_op_mat)
                for matkey in ["reduce_ops", "normal_ops"]:
                    ropmat = np.asarray(new_adj_op_mat[matkey])
                    # opmat will have a shape like 7 x 5
                    # Convert it into 7 x max_oplen with leading zero padding
                    padded_ropmat = np.zeros((ropmat.shape[0], self.max_oplen))
                    for i in range(ropmat.shape[0]):
                        padded_ropmat[i, -ropmat.shape[1]:] = ropmat[i]
                    # ss_pad will have a 1 x 4 opmat will have a shape 7 x max_oplen
                    # replicate ss_pad on each row of opmat to make it 7 x (max_oplen + 4)
                    ss_pad = self.ss_to_binary(space) 
                    final_mat = np.hstack([padded_ropmat, np.tile(ss_pad, (padded_ropmat.shape[0], 1))])
                    new_adj_op_mat[matkey] = final_mat.tolist()
            return new_adj_op_mat
        else:
            if space in ["nb101", "nb201", "nb301", "tb101"]:
                adj_op_mat = eval("self." + space).get_adj_op(idx)
                opmat = np.asarray(adj_op_mat["module_operations"])
                final_mat = self.pad_operation_matrix(opmat, space)
                new_adj_op_mat = copy.deepcopy(adj_op_mat)
                new_adj_op_mat["module_operations"] = final_mat.tolist()
            else:
                adj_op_mat = self.nds.get_adj_op(idx, space=space)
                new_adj_op_mat = copy.deepcopy(adj_op_mat)
                for matkey in ["reduce_ops", "normal_ops"]:
                    ropmat = np.asarray(new_adj_op_mat[matkey])
                    final_mat = self.pad_operation_matrix(ropmat, space)
                    new_adj_op_mat[matkey] = final_mat.tolist()
            return new_adj_op_mat
        
    def get_a2vcatezcp(self, idx, space, task="class_scene", joint=None):
        a2v = self.get_arch2vec(idx, space=space, joint=joint)
        if not isinstance(a2v, list):
            a2v = a2v.tolist()
        cate = self.get_cate(idx, space=space, joint=joint)
        if not isinstance(cate, list):
            cate = cate.tolist()
        zcp = self.get_zcp(idx, space=space, joint=joint, task=task)
        if not isinstance(zcp, list):
            zcp = zcp.tolist()
        return a2v + cate + zcp

    def get_zcp(self, idx, space, task="class_scene", joint=None):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            if space == "tb101":
                zcp = eval("self." + space).get_zcp(idx, task=task)
            else:
                zcp = eval("self." + space).get_zcp(idx)
        else:
            zcp = self.nds.get_zcp(idx, space=space)
        return zcp
        
    def get_norm_w_d(self, idx, space="Amoeba"):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            return [0, 0]
        else:
            return self.nds.get_norm_w_d(idx, space=space)
    
    def get_params(self, idx, space, task="class_scene"):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            if space == "tb101":
                params = eval("self." + space).get_params(idx, task=task)
            else:

                params = eval("self." + space).get_params(idx)
        else:
            params = self.nds.get_params(idx, space=space)
        return params
    
    def get_arch2vec(self, idx, space, joint=False):
        if joint:
            return self.joint_arch2vec_idxer[space][idx]
        else:
            if space in ["nb101", "nb201", "nb301", "tb101"]:
                arch2vec = eval("self." + space).get_arch2vec(idx)
            else:
                arch2vec = self.nds.get_arch2vec(idx, space=space)
            return arch2vec

    def get_cate(self, idx, space, joint=False):
        if joint:
            return self.joint_cate_idxer[space][idx]
        else:
            if space in ["nb101", "nb201", "nb301", "tb101"]:
                cate = eval("self." + space).get_cate(idx)
            else:
                cate = self.nds.get_cate(idx, space=space)
        return cate
    
    def get_valacc(self, idx, space):
        if space in ["nb101", "nb201", "nb301", "tb101"]:
            valacc = eval("self." + space).get_valacc(idx)
        else:
            valacc = self.nds.get_valacc(idx, space=space)
        return valacc

    def get_max_oplen(self):
        self.sskeys = list(self.ss_mapper.keys())
        oplens = {}
        for ssk in self.sskeys:
            if ssk in ["nb101", "nb201", "nb301", "tb101"]:
                oplens[ssk] = len(eval("self." + ssk).get_adj_op(0)["module_operations"][0])
            else:
                oplens[ssk + "_n"] = len(self.nds.get_adj_op(0, space=ssk)["normal_ops"][0])
                oplens[ssk + "_r"] = len(self.nds.get_adj_op(0, space=ssk)["reduce_ops"][0])
        self.oplens = oplens
        return max(list(oplens.values()))

    def _ensure_cache_exists(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
    def _load_classes(self):
        for key, (path, cls) in FILE_MAPPINGS.items():
            start_read_time = time.time()
            self._load_class_from_source_and_save_to_cache(key, path, cls)
            # if os.path.exists(path):
            #     print("Loading {} from cache".format(key))
            #     try:
            #         self._load_class_from_cache(key, path)
            #     except:
            #         print("Loading {} from source".format(key))
            #         self._load_class_from_source_and_save_to_cache(key, path, cls)
            # else:
            #     print("Loading {} from source".format(key))
            #     self._load_class_from_source_and_save_to_cache(key, path, cls)
            print("Load Time: {}".format(time.time() - start_read_time))

    def _load_class_from_cache(self, key, path):
        with open(path, 'rb') as inp:
            setattr(self, key, pickle.load(inp))

    def _load_class_from_source_and_save_to_cache(self, key, path, cls):
        instance = cls()
        setattr(self, key, instance)
        # with open(path, 'wb') as outp:
        #     pickle.dump(instance, outp, pickle.HIGHEST_PROTOCOL)

    def ss_to_binary(self, space):
        return [int(x) for x in f"{self.ss_mapper[space]:04b}"]

# all_ss = AllSS()