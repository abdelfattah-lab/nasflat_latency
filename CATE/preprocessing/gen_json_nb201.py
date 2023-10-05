from nasbench import api
from random import randint
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import sys, os
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from nb201_ss import NASBench201


def gen_data_point(nasbench):
    for unique_hash in tqdm(range(len(nasbench.nb2_api))):
        adj_op_desc = nasbench.get_adj_op(unique_hash)
        valacc = nasbench.get_valacc(unique_hash)
        yield {unique_hash: # unique_hash
                   {'test_accuracy': valacc,
                    'validation_accuracy': valacc,
                    'module_adjacency': adj_op_desc['module_adjacency'],
                    'module_operations': adj_op_desc['module_operations'],
                    'parameters': nasbench.get_params(unique_hash),
                    'training_time': 0}}

def gen_json_file():
    nasbench = NASBench201(path=os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
    nas_gen = gen_data_point(nasbench)
    data_dict = OrderedDict()
    for data_point in nas_gen:
        data_dict.update(data_point)
    with open('data/nasbench201.json', 'w') as outfile:
        json.dump(data_dict, outfile)


if __name__ == '__main__':
    gen_json_file()