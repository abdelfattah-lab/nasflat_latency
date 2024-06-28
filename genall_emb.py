
import os
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
import argparse, sys, time, random, os
import numpy as np
from tqdm import tqdm
import pickle

import blosc

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

all_embedding_dict = {}

sys.path.append("..")
for space in tqdm(['nb201', 'fbnet']):
    if space in ['nb201']:
        exec("from nas_embedding_suite.nb{}_ss import NASBench{} as EmbGenClass_local".format(space[-3:], space[-3:]))
    elif space in ['fbnet']:
        from nas_embedding_suite.fbnet_ss import FBNet as EmbGenClass_local
    embedding_gen = EmbGenClass_local(normalize_zcp=True, log_synflow=True)
    all_embedding_dict[space] = {}
    numitems = embedding_gen.get_numitems(space=space)
    all_embedding_dict[space]['arch2vec'] = []
    all_embedding_dict[space]['cate'] = []
    all_embedding_dict[space]['adj_op'] = []
    all_embedding_dict[space]['norm_w_d'] = []
    all_embedding_dict[space]['zcp'] = []
    all_embedding_dict[space]['latencies'] = []
    for idx in tqdm(range(numitems)):
        all_embedding_dict[space]['arch2vec'].append(embedding_gen.get_arch2vec(idx, space=space))
        all_embedding_dict[space]['cate'].append(embedding_gen.get_cate(idx, space=space))
        all_embedding_dict[space]['adj_op'].append(embedding_gen.get_adj_op(idx, space=space))
        all_embedding_dict[space]['norm_w_d'].append(embedding_gen.get_norm_w_d(idx, space=space))
        all_embedding_dict[space]['zcp'].append(embedding_gen.get_zcp(idx, space=space))
        # get_latency(idx, device=device_name) for all device_names
        devices_names = embedding_gen.devices
        all_embedding_dict[space]['latencies'].append({device_name: embedding_gen.get_latency(idx, device=device_name) for device_name in devices_names})
        # all_embedding_dict[space]['acc'].append(embedding_gen.get_valacc(idx, space=space))
        # Here, take all available devices and put each of them into a separate list


# Save all_embedding_dict as a pickle file
with open('NASFLATBench_v1.pkl', 'wb') as f: pickle.dump(all_embedding_dict, f)