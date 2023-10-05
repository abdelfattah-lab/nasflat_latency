from nasbench import api
from random import randint
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import sys, os
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from nds_ss import NDS
# Amoeba :  (13, 11)
# PNAS_fix-w-d :  (13, 11)
# ENAS_fix-w-d :  (13, 8)
# NASNet :  (13, 16)
# DARTS :  (11, 11)
# ENAS :  (13, 8)
# PNAS :  (13, 11)
# DARTS_lr-wd :  (11, 11)
# DARTS_fix-w-d :  (11, 11)
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--search_space', type=str, default='Amoeba')
parser.add_argument('--type', type=str, default='normal')


def gen_data_point(nasbench, args):
    for unique_hash in tqdm(range(len(nasbench.space_adj_mats[args.search_space]))):
        adj_op_desc = nasbench.get_adj_op(unique_hash, args.search_space)
        valacc = nasbench.get_valacc(unique_hash, args.search_space)
        yield {unique_hash: # unique_hash
                   {'test_accuracy': valacc,
                    'validation_accuracy': valacc,
                    'module_adjacency': adj_op_desc[args.type + '_adj'],
                    'module_operations': adj_op_desc[args.type + '_ops'],
                    'parameters': nasbench.get_params(unique_hash, args.search_space),
                    'training_time': 0}}

def gen_json_file(args):
    nasbench = NDS()
    nas_gen = gen_data_point(nasbench, args)
    data_dict = OrderedDict()
    for data_point in nas_gen:
        data_dict.update(data_point)
    with open('data/nds_%s_%s.json' % (str(args.search_space), str(args.type)), 'w') as outfile:
        json.dump(data_dict, outfile)


# if __name__ == '__main__':
args = parser.parse_args()
gen_json_file(args)