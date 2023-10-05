from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.environ['PROJ_BPATH'] + "/")
from random import randint
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from tqdm import tqdm
import time


from nas_embedding_suite.all_ss import AllSS
all_ss = AllSS()
ss_mapper = {"nb101": 0, "nb201": 1, "nb301": 2, "Amoeba": 3, "PNAS_fix-w-d": 4, 
                     "ENAS_fix-w-d": 5, "NASNet": 6, "DARTS": 7, "ENAS": 8, "PNAS": 9, 
                     "DARTS_lr-wd": 10, "DARTS_fix-w-d": 11, "tb101": 12}

spaces_m = {}
for ssk in ss_mapper.keys():
    if ssk=='nb101':
        skl = list(range(len(all_ss.nb101.valacc_list)))
    elif ssk=='nb201':
        skl = list(range(len(all_ss.nb201.arch2vec_nb201.keys())))
    elif ssk=='nb301':
        skl = list(range(all_ss.nb301.cate_nb301['embeddings'].shape[0]))
    elif ssk=='tb101':
        skl = list(range(all_ss.tb101.cate_embeddings['jigsaw']['embeddings'].shape[0]))
    else:
        skl = list(range(all_ss.nds.cate_embeddings[ssk]['embeddings'].shape[0]))
    spaces_m[ssk] = skl

true_l = []
lagval = 0
space_halts = {}
for spm_k, spm in spaces_m.items():
    spm_n = np.asarray(spm) + lagval
    space_halts[spm_k] = lagval     
    lagval = spm_n[-1] + 1
    true_l.append(spm_n)

true_l = np.concatenate(np.asarray(true_l)).flatten()
space_halts_inv = {v: k for k, v in space_halts.items()}
# exit(0)
# def gen_json_file():
    # nas_gen = gen_data_point()
data_dict = OrderedDict()
curr_space = 'nb101'
for i in tqdm(true_l):
    if i in space_halts_inv.keys():
        a = time.time()
        curr_space = space_halts_inv[i]
        print("New Curr Space: ", curr_space)
    if curr_space in ['nb101', 'nb201', 'nb301', 'tb101']:
        vacc = eval('all_ss.' + curr_space).get_valacc(i - space_halts[curr_space])
        madj = all_ss.get_adj_op(i - space_halts[curr_space], curr_space)["module_adjacency"]
        mop = all_ss.get_adj_op(i - space_halts[curr_space], curr_space)["module_operations"]
    else:
        vacc = all_ss.nds.get_valacc(i - space_halts[curr_space], curr_space)
        madj = all_ss.get_adj_op(i - space_halts[curr_space], curr_space)["normal_adj"]
        mop = all_ss.get_adj_op(i - space_halts[curr_space], curr_space)["normal_ops"]

    data_dict.update({int(i): # unique_hash
                    {'test_accuracy': vacc,
                        'validation_accuracy': vacc,
                        'module_adjacency': madj,
                        'module_operations': mop,
                        'training_time': 0}})
    if i - 100 in space_halts_inv.keys():
        b = time.time()
        print("Avg Time Per Query for %s: ".format(curr_space), (b-a)/100)

# for data_point in nas_gen:
    # data_dict.update(data_point)
with open('data/all_ss.json', 'w') as outfile:
    json.dump(data_dict, outfile)




# if __name__ == '__main__':
#     gen_json_file()
