"""API source: https://github.com/D-X-Y/NAS-Bench-201/blob/v1.1/nas_201_api/api.py"""
import numpy as np
import json, sys, os
from collections import OrderedDict
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from fbnet_ss import FBNet

# Create a argparser for 2 integers
import argparse
from tqdm import tqdm
# If data/nb3_sets doesnt exist, make it
import os

# if not os.path.exists('data/nb3_sets'):
#     os.makedirs('data/nb3_sets')

parser = argparse.ArgumentParser()
args = parser.parse_args()

def gen_data_point(nasbench):
    for unique_hash in tqdm(range(fbnet.get_numitems())):
        adj_op_desc = fbnet.get_adj_op(unique_hash)
        valacc = fbnet.get_valacc(unique_hash)
        yield {unique_hash:
                {'test_accuracy': valacc,
                 'validation_accuracy': valacc,
                 'module_adjacency': adj_op_desc['module_adjacency'],
                 'module_operations': adj_op_desc['module_operations'],
                 'training_time': 0}}
    # for unique_hash in tqdm(range(len(nasbench.space_adj_mats[args.search_space]))):
    #     adj_op_desc = nasbench.get_adj_op(unique_hash, args.search_space)
    #     valacc = nasbench.get_valacc(unique_hash, args.search_space)
    #     yield {unique_hash:
    #             {'test_accuracy': valacc,
    #              'validation_accuracy': valacc,
    #              'module_adjacency': adj_op_desc[args.type + '_adj'],
    #              'module_operations': adj_op_desc[args.type + '_ops'],
    #              'training_time': 0}}


fbnet = FBNet()
nas_gen = gen_data_point(fbnet)
data_dict = OrderedDict()
for data_point in nas_gen:
    data_dict.update(data_point)

# if fbnet directory doesnt exist, make it.
if not os.path.exists('data/fbnet'):
    os.makedirs('data/fbnet')

with open('data/fbnet/fbnet.json', 'w') as f:
    json.dump(str(data_dict), f)
