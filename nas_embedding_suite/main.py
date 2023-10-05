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



if __name__ == '__main__':
    test_nb1, test_nb2, test_nb3, test_tb1mic, test_nds = True, True, True, True, True
    if test_nb1:
        print("="*53)
        print("="*20, "NASBench101", "="*20)
        print("="*53)
        from nb101_ss import NASBench101
        nb101 = NASBench101()
        print(nb101.index_to_embedding(0))
        print(nb101.index_to_embedding(52))
        print("="*53)
        print("="*17, "API Distance Test", "="*17)
        print("="*53)
        for idx in range(10):
            # print(nb101.arch2vec_nb101[idx]['valid_accuracy'], nb101.cate_nb101['valid_accs'][idx], nb101.nb1_api.get_metrics_from_hash(nb101.hash_iterator_list[idx])[1][108][0]['final_validation_accuracy'])
            print(nb101.arch2vec_nb101[idx]['valid_accuracy'] - nb101.nb1_api.get_metrics_from_hash(nb101.hash_iterator_list[idx])[1][108][0]['final_validation_accuracy'])
    if test_nb2:
        print("="*53)
        print("="*20, "NASBench201", "="*20)
        print("="*53)
        from nb201_ss import NASBench201
        nb201 = NASBench201()
        print(nb201.index_to_embedding(0))
        print(nb201.index_to_embedding(1522))
    if test_nb3:
        print("="*53)
        print("="*20, "NASBench301", "="*20)
        print("="*53)
        from nb301_ss import NASBench301
        nb301 = NASBench301(use_nb3_performance_model=False)
        print(nb301.index_to_embedding(0))
        print(nb301.index_to_embedding(1000000))
        print(nb301.index_to_embedding(1000010))
        if nb301.use_nb3_performance_model:
            print("="*53)
            print("="*17, "API Distance Test", "="*17)
            print("="*53)
            for idx in range(10):
                print(nb301.cate_nb301['predicted_accs'][idx] - nb301.get_valacc(idx))
            for idx in range(1000000, 1000040, 1):
                print(nb301.cate_nb301_zcp['cifar10'][list(nb301.cate_nb301_zcp['cifar10'].keys())[idx-1000000]]['val_accuracy'] - nb301.get_valacc(idx))
            print("="*53)
    if test_tb1mic:
        print("="*53)
        print("="*20, "TB101MICRO", "="*20)
        print("="*53)
        from tb101_micro_ss import TransNASBench101Micro
        tb1mic = TransNASBench101Micro()
        print(tb1mic.index_to_embedding(0, task='normal'))
        print(tb1mic.index_to_embedding(52, task='normal'))
        print(tb1mic.index_to_embedding(520, task='normal'))
        print(tb1mic.index_to_embedding(4095, task='normal'))
        
    if test_nds:
        print("="*53)
        print("="*20, "NDS", "="*20)
        print("="*53)
        from nds_ss import NDS
        nds = NDS()
        print(nds.get_adj_op(0))
        print(nds.get_adj_op(10))
        print(nds.get_adj_op(100))


    if test_nb1:
        print("NB101: ", nb101.index_to_embedding(0).keys())
    if test_nb2:
        print("NB201: ", nb201.index_to_embedding(0).keys())
    if test_nb3:
        print("NB301: ", nb301.index_to_embedding(0).keys())
    if test_tb1mic:
        print("TB101Micro: ", tb1mic.index_to_embedding(0).keys())
    