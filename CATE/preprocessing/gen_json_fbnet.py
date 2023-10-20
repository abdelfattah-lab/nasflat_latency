from random import randint
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import sys, os
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from fbnet_ss import FBNet



def gen_data_point(fbnet):
    for unique_hash in tqdm(range(fbnet.get_numitems())):
        adj_op_desc = fbnet.get_adj_op(unique_hash)
        valacc = fbnet.get_valacc(unique_hash)
        yield {unique_hash: # unique_hash
                   {'test_accuracy': valacc,
                    'validation_accuracy': valacc,
                    'module_adjacency': adj_op_desc['module_adjacency'],
                    'module_operations': adj_op_desc['module_operations'],
                    'parameters': fbnet.get_params(unique_hash),
                    'training_time': 0}}

def gen_json_file():
    fbnet = FBNet()
    nas_gen = gen_data_point(fbnet)
    data_dict = OrderedDict()
    for data_point in nas_gen:
        data_dict.update(data_point)
    with open('data/fbnet.json', 'w') as outfile:
        json.dump(data_dict, outfile)


# if __name__ == '__main__':
gen_json_file()