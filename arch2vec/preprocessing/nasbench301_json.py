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
from nb301_ss import NASBench301

# Create a argparser for 2 integers
import argparse
# If data/nb3_sets doesnt exist, make it
import os

# if not os.path.exists('data/nb3_sets'):
#     os.makedirs('data/nb3_sets')

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100000)
args = parser.parse_args()

def gen_data_point(nasbench, args):
    for unique_hash in tqdm(range(args.start, args.end, 1)):
        if unique_hash % 50000 == 0:
            print('Current unique_hash: {}'.format(unique_hash), flush=True)
        adj_op_desc = nasbench.get_adj_op(unique_hash)
        valacc = nasbench.get_valacc(unique_hash)
        yield {unique_hash:
                {'test_accuracy': valacc,
                 'validation_accuracy': valacc,
                 'module_adjacency': adj_op_desc['module_adjacency'],
                 'module_operations': adj_op_desc['module_operations'],
                 'training_time': 0}}

# def gen_json_file():
nb301 = NASBench301(path=os.environ['PROJ_BPATH'] + "/", use_nb3_performance_model=True)
nas_gen = gen_data_point(nb301, args)
data_dict = OrderedDict()
for data_point in nas_gen:
    data_dict.update(data_point)

partition = int(args.end/50000)
with open('data/nb3_sets/nasbench301_%s.json' % (str(partition)), 'w') as f:
    json.dump(str(data_dict), f)

# if __name__=='__main__':
#     gen_json_file()