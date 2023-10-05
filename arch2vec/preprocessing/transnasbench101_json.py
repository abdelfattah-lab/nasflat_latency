from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import sys
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from tb101_micro_ss import TransNASBench101Micro

# Create a argparser for 2 integers
import argparse
# If data/nb3_sets doesnt exist, make it
import os

# if not os.path.exists('data/nb3_sets'):
#     os.makedirs('data/nb3_sets')

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='class_scene')
args = parser.parse_args()

def gen_data_point(nasbench, args):
    for unique_hash in tqdm(range(len(nasbench.zcp_tb101[args.task]))):
        adj_op_desc = nasbench.get_adj_op(unique_hash, task=args.task)
        valacc = nasbench.get_valacc(unique_hash, task=args.task)
        yield {unique_hash:
                {'test_accuracy': valacc,
                 'validation_accuracy': valacc,
                 'module_adjacency': adj_op_desc['module_adjacency'],
                 'module_operations': adj_op_desc['module_operations'],
                 'training_time': 0}}

# def gen_json_file():
tb101mic = TransNASBench101Micro(path=os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/')
nas_gen = gen_data_point(tb101mic, args)
data_dict = OrderedDict()
for data_point in nas_gen:
    data_dict.update(data_point)

with open('data/tb101micro_%s.json' % (str(args.task)), 'w') as f:
    json.dump(str(data_dict), f)

# if __name__=='__main__':
#     gen_json_file()