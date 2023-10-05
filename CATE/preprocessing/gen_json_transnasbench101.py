from nasbench import api
from random import randint
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import sys, os
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from tb101_micro_ss import TransNASBench101Micro

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--task', type=str, default='class_scene')


def gen_data_point(nasbench, args):
    for unique_hash in tqdm(range(len(nasbench.zcp_tb101[args.task]))):
        adj_op_desc = nasbench.get_adj_op(unique_hash, args.task)
        valacc = nasbench.get_valacc(unique_hash, args.task)
        yield {unique_hash: # unique_hash
                   {'test_accuracy': valacc,
                    'validation_accuracy': valacc,
                    'module_adjacency': adj_op_desc['module_adjacency'],
                    'module_operations': adj_op_desc['module_operations'],
                    'parameters': nasbench.get_params(unique_hash, args.task),
                    'training_time': 0}}

def gen_json_file(args):
    nasbench = TransNASBench101Micro(path=os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite', normalize_zcp=False, log_synflow=False)
    nas_gen = gen_data_point(nasbench, args)
    data_dict = OrderedDict()
    for data_point in nas_gen:
        data_dict.update(data_point)
    with open('data/transnasbench101_%s.json' % (str(args.task)), 'w') as outfile:
        json.dump(data_dict, outfile)


# if __name__ == '__main__':
args = parser.parse_args()
gen_json_file(args)