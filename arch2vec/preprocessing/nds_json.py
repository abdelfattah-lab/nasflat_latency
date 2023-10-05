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
from nds_ss import NDS

# Create a argparser for 2 integers
import argparse
# If data/nb3_sets doesnt exist, make it
import os

# if not os.path.exists('data/nb3_sets'):
#     os.makedirs('data/nb3_sets')

parser = argparse.ArgumentParser()
parser.add_argument('--search_space', type=str, default='Amoeba')
parser.add_argument('--type', type=str, default='normal')
args = parser.parse_args()

def gen_data_point(nasbench, args):
    for unique_hash in tqdm(range(len(nasbench.space_adj_mats[args.search_space]))):
        adj_op_desc = nasbench.get_adj_op(unique_hash, args.search_space)
        valacc = nasbench.get_valacc(unique_hash, args.search_space)
        yield {unique_hash:
                {'test_accuracy': valacc,
                 'validation_accuracy': valacc,
                 'module_adjacency': adj_op_desc[args.type + '_adj'],
                 'module_operations': adj_op_desc[args.type + '_ops'],
                 'training_time': 0}}


nds = NDS()
nas_gen = gen_data_point(nds, args)
data_dict = OrderedDict()
for data_point in nas_gen:
    data_dict.update(data_point)

# if nds_%s directory doesnt exist, make it.
if not os.path.exists('data/nds_%s' % (str(args.search_space))):
    os.makedirs('data/nds_%s' % (str(args.search_space)))

with open('data/nds_%s/nds_%s_%s.json' % (str(args.search_space), str(args.search_space), str(args.type)), 'w') as f:
    json.dump(str(data_dict), f)
