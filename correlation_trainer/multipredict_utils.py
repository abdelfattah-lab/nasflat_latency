####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import logging
import os
import numpy as np

import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr
import copy
from multipredict_net import FewShotPredictor, FewShotNoEmbeddingPredictor, FewShotMetaPredictor
#import wandb
import random
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
import pickle
import numpy as np
import math
from tqdm import tqdm

# FSL
# MAML
# MAML Second Order Approx
# HELP
# BRPNAS


def train_single_task(net, specific_transfer_train_loader, criterion, device, embedding_type, lr=1e-3, num_updates=2, search_space='nasbench201'):
    net.train()
    data_item = next(iter(specific_transfer_train_loader))
    adapted_params = net.cloned_params()
    for n in range(num_updates):
        nn_arch, hw_idx, latencies = data_item
        if search_space=='nasbench201':
            nn_arch = (nn_arch[0].to(device), nn_arch[1].to(device))
        else:
            nn_arch = nn_arch.to(device)
        if embedding_type=='learnable':
            outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device), params=adapted_params)
        else:
            raise NotImplementedError
        loss = criterion(outputs, torch.Tensor(np.asarray(latencies)).unsqueeze(1).to(device))
        grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, allow_unused=True)
        for (key, val), grad in zip(adapted_params.items(), grads):
            adapted_params[key] = val - lr * grad
    return adapted_params



def test_net(net, embedding_type, epochs, test_loader, criterion, device, search_space='nasbench201'):
    net.eval()
    test_loss = 0.0
    for i, data in enumerate(test_loader):
        nn_arch, hw_idx, latencies, zcp = data
        if search_space=='nasbench201':
            nn_arch = (nn_arch[0].to(device), nn_arch[1].to(device))
        else:
            nn_arch = nn_arch.to(device)
        if embedding_type == 'learnable':
            outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device), zcp.to(device))
        else:
            outputs = net(nn_arch, hw_idx.to(device), zcp.to(device))
        loss = criterion(outputs, torch.Tensor(np.asarray(latencies)).unsqueeze(1).to(device))
        test_loss += loss.item()
    return test_loss/len(test_loader)

def train_net(net, embedding_type, epochs, train_loader, criterion, optimizer, device, search_space='nasbench201'):
    net.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            nn_arch, hw_idx, latencies, zcp = data
            optimizer.zero_grad()
            if search_space=='nasbench201':
                nn_arch = (nn_arch[0].to(device), nn_arch[1].to(device))
            else:
                nn_arch = nn_arch.to(device)
            if embedding_type=='learnable':
                outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device), zcp.to(device))
            else:
                outputs = net(nn_arch, hw_idx.to(device))
            loss = criterion(outputs, torch.Tensor(np.asarray(latencies)).unsqueeze(1).to(device), zcp.to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
    return net

def train_net_brp_params(net, embedding_type, epochs, train_loader, criterion, optimizer, scheduler, val_loader, device, search_space='nasbench201'):
    net.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            nn_arch, hw_idx, latencies, zcp_ = data
            optimizer.zero_grad()
            if search_space=='nasbench201':
                zcp_ = zcp_.to(device)
                nn_arch = (nn_arch[0].to(device), nn_arch[1].to(device))
            else:
                nn_arch = nn_arch.to(device)
            if embedding_type=='learnable':
                # import pdb; pdb.set_trace()
                outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device), zcp_)
            else:
                outputs = net(nn_arch, hw_idx.to(device), zcp_)
            loss = criterion(outputs, torch.Tensor(np.asarray(latencies)).unsqueeze(1).to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # val_running_loss = 0.0
        # for i, data in enumerate(train_loader):
        #     if i > 20:
        #         break
        #     nn_arch, hw_idx, latencies = data
        #     if search_space=='nasbench201':
        #         nn_arch = (nn_arch[0].to(device), nn_arch[1].to(device))
        #     else:
        #         nn_arch = nn_arch.to(device)
        #     if embedding_type=='learnable':
        #         outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device))
        #     else:
        #         outputs = net(nn_arch, hw_idx.to(device))
        #     loss = criterion(outputs, torch.Tensor(np.asarray(latencies)).unsqueeze(1).to(device))
        #     val_running_loss += loss.item()
        # scheduler.step(val_running_loss)
        scheduler.step()
    return net


def emb_generator(embedding_type, hw_emb_dim, full_latency_dict, archs):
    if embedding_type == 'learnable':
        return None
    elif embedding_type == 'sample':
        arch_index_to_embedding = random.sample(range(len(archs)), hw_emb_dim)
        return {idx: torch.FloatTensor(full_latency_dict[x][arch_index_to_embedding]) for idx, x in enumerate(full_latency_dict.keys())}
    elif embedding_type == 'index':
        device_idx_to_emb  = {idx: [int(b) for b in [*bin(x)[2:]]] for idx, x in enumerate(full_latency_dict.keys())}
        return {k: torch.FloatTensor((hw_emb_dim-len(v))*[0] + v) for k,v in device_idx_to_emb.items()}


def get_latency_dicts(latency_data_path, device_name_idx, train_devices, transfer_devices, device_list):
    train_latency_dict = {device_name_idx[device_name]: normalization(np.asarray(torch.load(os.path.join(latency_data_path, device_name))), portion=1) for device_name in device_list if device_name.replace(".pt", "") in train_devices}
    transfer_latency_dict = {device_name_idx[device_name]: normalization(np.asarray(torch.load(os.path.join(latency_data_path, device_name))), portion=1) for device_name in device_list if device_name.replace(".pt", "") in transfer_devices}
    full_latency_dict = {device_name_idx[device_name]: normalization(np.asarray(torch.load(os.path.join(latency_data_path, device_name))), portion=1) for device_name in device_list}
    return (train_latency_dict, transfer_latency_dict, full_latency_dict)

def get_full_device_list(search_space='nasbench201'):
    if search_space=='nasbench201':
        return ["1080ti_1.pt","2080ti_1.pt","desktop_cpu_core_i7_7820x_fp32.pt","embedded_gpu_jetson_nano_fp32.pt",\
                "eyeriss.pt","gold_6240.pt","mobile_cpu_snapdragon_855_kryo_485_int8.pt","mobile_gpu_snapdragon_450_adreno_506_int8.pt",\
                "pixel2.pt","samsung_a50.pt","silver_4210r.pt","titan_rtx_32.pt","titanx_32.pt","titanxp_32.pt","1080ti_256.pt",\
                "2080ti_256.pt","desktop_gpu_gtx_1080ti_fp32.pt","embedded_tpu_edge_tpu_int8.pt","fpga.pt","mobile_cpu_snapdragon_450_cortex_a53_int8.pt",\
                "mobile_dsp_snapdragon_675_hexagon_685_int8.pt","mobile_gpu_snapdragon_675_adreno_612_int8.pt","pixel3.pt","samsung_s7.pt",\
                "titan_rtx_1.pt","titanx_1.pt","titanxp_1.pt","1080ti_32.pt","2080ti_32.pt","embedded_gpu_jetson_nano_fp16.pt","essential_ph_1.pt",\
                "gold_6226.pt","mobile_cpu_snapdragon_675_kryo_460_int8.pt","mobile_dsp_snapdragon_855_hexagon_690_int8.pt",\
                "mobile_gpu_snapdragon_855_adreno_640_int8.pt","raspi4.pt","silver_4114.pt","titan_rtx_256.pt","titanx_256.pt","titanxp_256.pt"]
    elif search_space=='fbnet':
        return ["1080ti_1.pt","1080ti_64.pt","2080ti_32.pt","essential_ph_1.pt","fpga.pt","gold_6240.pt","pixel3.pt","samsung_a50.pt","silver_4114.pt",\
                "titan_rtx_1.pt","titan_rtx_64.pt","titanx_32.pt","titanxp_1.pt","titanxp_64.pt","1080ti_32.pt","2080ti_1.pt","2080ti_64.pt","eyeriss.pt",\
                "gold_6226.pt","pixel2.pt","raspi4.pt","samsung_s7.pt","silver_4210r.pt","titan_rtx_32.pt","titanx_1.pt","titanx_64.pt","titanxp_32.pt"]

def get_alt_full_device_list(search_space='nasbench201'):
    if search_space=='nasbench201':
        return ["fisher_nb201_cifar10.pt","jacov_nb201_cifar10.pt","synflow_nb201_cifar10.pt","1080ti_1.pt","2080ti_1.pt","desktop_cpu_core_i7_7820x_fp32.pt","embedded_gpu_jetson_nano_fp32.pt",\
                "eyeriss.pt","gold_6240.pt","mobile_cpu_snapdragon_855_kryo_485_int8.pt","mobile_gpu_snapdragon_450_adreno_506_int8.pt",\
                "pixel2.pt","samsung_a50.pt","silver_4210r.pt","titan_rtx_32.pt","titanx_32.pt","titanxp_32.pt","1080ti_256.pt",\
                "2080ti_256.pt","desktop_gpu_gtx_1080ti_fp32.pt","embedded_tpu_edge_tpu_int8.pt","fpga.pt","mobile_cpu_snapdragon_450_cortex_a53_int8.pt",\
                "mobile_dsp_snapdragon_675_hexagon_685_int8.pt","mobile_gpu_snapdragon_675_adreno_612_int8.pt","pixel3.pt","samsung_s7.pt",\
                "titan_rtx_1.pt","titanx_1.pt","titanxp_1.pt","1080ti_32.pt","2080ti_32.pt","embedded_gpu_jetson_nano_fp16.pt","essential_ph_1.pt",\
                "gold_6226.pt","mobile_cpu_snapdragon_675_kryo_460_int8.pt","mobile_dsp_snapdragon_855_hexagon_690_int8.pt",\
                "mobile_gpu_snapdragon_855_adreno_640_int8.pt","raspi4.pt","silver_4114.pt","titan_rtx_256.pt","titanx_256.pt","titanxp_256.pt"]
    elif search_space=='fbnet':
        return ["1080ti_1.pt","1080ti_64.pt","2080ti_32.pt","essential_ph_1.pt","fpga.pt","gold_6240.pt","pixel3.pt","samsung_a50.pt","silver_4114.pt",\
                "titan_rtx_1.pt","titan_rtx_64.pt","titanx_32.pt","titanxp_1.pt","titanxp_64.pt","1080ti_32.pt","2080ti_1.pt","2080ti_64.pt","eyeriss.pt",\
                "gold_6226.pt","pixel2.pt","raspi4.pt","samsung_s7.pt","silver_4210r.pt","titan_rtx_32.pt","titanx_1.pt","titanx_64.pt","titanxp_32.pt"]

def load_archs(arch_data_path, search_space='nasbench201'):
    if search_space=='nasbench201':
        return [[add_global_node(_['operation'], ifAdj=False), 
                                add_global_node(_['adjacency_matrix'],ifAdj=True)]
                    for _ in torch.load(os.path.join(arch_data_path, 'architecture.pt'))]
    elif search_space=='fbnet':
        return [arch_enc(_['op_idx_list']) for _ in 
                torch.load(os.path.join(arch_data_path, 'metainfo.pt'))['arch']]


def load_net(test_idx, search_space, num_trials, report, s, emb_transfer_samples, fsh_sampling_strat, fsh_mc_sampling, dev_train_samples, train_batchsize, test_batchsize, epochs, transfer_epochs, \
            mixed_training, mixed_train_weight, hw_emb_dim, gcn_layer_size, nn_emb_dim, feat_layer_size, \
            feat_depth, loss_function, train_device_list, transfer_device_list, use_specific_lr, \
            transfer_specific_lr, embedding_type, closest_correlator, embedding_transfer, freeze_non_embedding, \
            adapt_new_embedding, pre_train_transferset, device, cpu_map):
    ### Initialize Network
    nfeat = 132 if search_space=='fbnet' else 8
    if embedding_type == 'learnable' and report != 'adaptive_learned_meta':
        net = FewShotPredictor( 
                        nfeat                   =   nfeat             , \
                        gcn_layer_size          =   gcn_layer_size    , \
                        nn_emb_dim              =   nn_emb_dim        , \
                        feat_layer_size         =   feat_layer_size   , \
                        feat_depth              =   feat_depth        , \
                        train_device_list       =   train_device_list , \
                        hw_emb_dim              =   hw_emb_dim        , \
                        search_space            =   search_space       
                            ).to(device)
    elif embedding_type == 'sample' or embedding_type == 'index':
        net = FewShotNoEmbeddingPredictor( 
                        nfeat                   =   nfeat             , \
                        gcn_layer_size          =   gcn_layer_size    , \
                        nn_emb_dim              =   nn_emb_dim        , \
                        feat_layer_size         =   feat_layer_size   , \
                        feat_depth              =   feat_depth        , \
                        train_device_list       =   train_device_list , \
                        hw_emb_dim              =   hw_emb_dim        , \
                        search_space            =   search_space          
                            ).to(device)
    elif embedding_type == 'learnable' and report == 'adaptive_learned_meta':
        net = FewShotMetaPredictor(
                        nfeat                   =   nfeat             , \
                        gcn_layer_size          =   gcn_layer_size    , \
                        nn_emb_dim              =   nn_emb_dim        , \
                        feat_layer_size         =   feat_layer_size   , \
                        feat_depth              =   feat_depth        , \
                        train_device_list       =   train_device_list , \
                        hw_emb_dim              =   hw_emb_dim        , \
                        search_space            =   search_space       
                            ).to(device)
    return net

def read_config(path):
    with open(path, 'r') as f:
        base_config = json.load(f)
    test_idx                        = base_config['test_idx']
    search_space                    = base_config['search_space']
    num_trials                      = base_config['num_trials']
    report                          = base_config['report']
    s                               = base_config['s']
    emb_transfer_samples            = base_config['emb_transfer_samples']
    fsh_sampling_strat              = base_config['fsh_sampling_strat']
    fsh_mc_sampling                 = base_config['fsh_mc_sampling']
    dev_train_samples               = base_config['dev_train_samples']  
    train_batchsize                 = base_config['train_batchsize']
    test_batchsize                  = base_config['test_batchsize']
    epochs                          = base_config['epochs']
    transfer_epochs                 = base_config['transfer_epochs']
    mixed_training                  = base_config['mixed_training']
    mixed_train_weight              = base_config['mixed_train_weight']
    hw_emb_dim                      = base_config['hw_emb_dim']
    gcn_layer_size                  = base_config['gcn_layer_size']
    nn_emb_dim                      = base_config['nn_emb_dim']
    feat_layer_size                 = base_config['feat_layer_size']
    feat_depth                      = base_config['feat_depth']
    loss_function                   = base_config['loss_function']
    train_device_list               = base_config['train_device_list']
    transfer_device_list            = base_config['transfer_device_list']
    use_specific_lr                 = base_config['use_specific_lr']
    transfer_specific_lr            = base_config['transfer_specific_lr']
    embedding_type                  = base_config['embedding_type']
    closest_correlator              = base_config['closest_correlator']
    embedding_transfer              = base_config['embedding_transfer']
    freeze_non_embedding            = base_config['freeze_non_embedding']
    adapt_new_embedding             = base_config['adapt_new_embedding']
    pre_train_transferset           = base_config['pre_train_transferset']
    device                          = base_config['device']
    cpu_map                         = base_config['cpu_map']
    return (test_idx, search_space, num_trials, report, s, emb_transfer_samples, fsh_sampling_strat, fsh_mc_sampling, dev_train_samples, train_batchsize, test_batchsize, epochs, transfer_epochs, \
            mixed_training, mixed_train_weight, hw_emb_dim, gcn_layer_size, nn_emb_dim, feat_layer_size, \
            feat_depth, loss_function, train_device_list, transfer_device_list, use_specific_lr, \
            transfer_specific_lr, embedding_type, closest_correlator, embedding_transfer, freeze_non_embedding, \
            adapt_new_embedding, pre_train_transferset, device, cpu_map)

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_dataloader(  
                    archs,
                    device_idx_mapper,
                    device_dataset,
                    sample_indexes,
                    latency_dict,
                    device_name_idx,
                    device_idx_to_emb=None,
                    per_device_num_samples=900,
                    embedding_type="learnable", 
                    embedding_dim=10, 
                    train_batchsize=128, 
                    test_batchsize=32, 
                    mixed_training=False,
                    mixed_train_weight=1,
                    transfer_set=False,
                    search_space='nasbench201',
                    fsh_sampling_strat='random',
                    ref_uncorr_dev_latency_dict=None,
                    reference_device_name=None,
                    embedding_gen=None
                    ):
    '''
    device_idx_mapper           =  {device_name: idx for idx, device_name in enumerate(train_devices + transfer_devices)}
    device_dataset              =  [x for x in train_devices] (Or whatever devices we need a data-set for.)
    sample_indexes              =  Explicitly provide a list of device samples to get.
    device_idx_to_sample_emb    =  Dictionary of device index to sample embeddings
    device_idx_to_idx_emb       =  Dictionary of device index to index embeddings
    per_device_num_samples      =  if 15625, do the whole dataset. 
    embedding_type              =  {"learnable", "sample", "index"}
    ******************* MIXED WEIGHT TRAINING NOT SUPPORTED YET *******************
    '''
    num_nets = len(archs)
    if mixed_training==True:
        raise ValueError("Mixed weight training not supported yet.")
    train_dataset = []
    test_dataset  = []

    train_split   = per_device_num_samples/len(archs)
    if transfer_set == False:
        # train_idxs    = len(device_dataset)*list(range(len(archs)))[:int(len(archs)*train_split)]
        train_idxs = random.sample(list(range(len(archs))), int(train_split*len(archs)))*len(device_dataset)
    else:
        if fsh_sampling_strat=='random':
            train_idxs    = random.sample(list(range(num_nets)), int(len(device_dataset)*train_split*len(archs)))
            print(len(train_idxs))
        elif fsh_sampling_strat=='uncorrelated':
            # Get full list of (idx, latencies) on train device
            ref_lat_dict = ref_uncorr_dev_latency_dict[device_name_idx[reference_device_name]]
            ref_lat_list = [(idx, lat) for idx, lat in enumerate(ref_lat_dict)]
            # Sort by latency
            ref_lat_list.sort(key=lambda x: x[1])
            # Sample uniformly spaced 'n' elements
            train_idxs = [x[0][0] for x in divide_chunks(ref_lat_list, len(archs)//per_device_num_samples)][:per_device_num_samples]
            print(len(train_idxs))
            print(train_idxs)

    test_idxs     = list(set(range(len(archs))) - set(train_idxs))
    device_idx_name = {v: k.replace(".pt", "") for k, v in device_name_idx.items()}

    for unif_iter, idx in enumerate(train_idxs):
        # if transfer_set==False:
        device_train = list(latency_dict.keys())[unif_iter % len(latency_dict)]
        zcp = torch.Tensor(embedding_gen.get_zcp(idx))
        # else:
        if embedding_type == 'learnable':
            train_dataset.append([archs[idx], [device_idx_mapper[device_idx_name[device_train]]], latency_dict[device_train][idx], zcp])
        else:
            train_dataset.append([archs[idx], device_idx_to_emb[device_train], latency_dict[device_train][idx], zcp])

    for idx in test_idxs:
        # if transfer_set==False:
        device_test = list(latency_dict.keys())[random.randint(0, len(latency_dict)-1)]
        zcp = torch.Tensor(embedding_gen.get_zcp(idx))
        if embedding_type == 'learnable':
            test_dataset.append([archs[idx], [device_idx_mapper[device_idx_name[device_train]]], latency_dict[device_test][idx], zcp])
        else:
            test_dataset.append([archs[idx], device_idx_to_emb[device_test], latency_dict[device_test][idx], zcp])

    train_loader = DataLoader(train_dataset, batch_size=train_batchsize, num_workers=1, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=test_batchsize, num_workers=1, shuffle=False)

    return train_loader, test_loader


def get_allarch_data(archs, device_name, latency_data_path, device_idx_mapper, embedding_type, device_name_idx, device_idx_to_emb, embedding_gen=None):
    latency_arr = normalization(np.array(torch.load(os.path.join(latency_data_path, device_name + '.pt'))), portion=1)
    data_list = []
    for i in range(len(archs)):
        if embedding_type=='learnable':
            zcp = torch.Tensor(embedding_gen.get_zcp(i))
            data_list.append((archs[i], [device_idx_mapper[device_name]], zcp))
        else:
            zcp = torch.Tensor(embedding_gen.get_zcp(i))
            data_list.append((archs[i], device_idx_to_emb[device_idx_mapper[device_name]], zcp))
    return DataLoader(data_list, batch_size=128, shuffle=False)

def get_latency(net, device_dataloader, embedding_type, device, adapted_params=None, search_space='nasbench201', embedding_gen=None):
    latency_list = []
    for _, data in enumerate(device_dataloader):
        nn_arch, hw_idx, zcp = data
        if search_space=='nasbench201':
            nn_arch = (nn_arch[0].to(device), nn_arch[1].to(device))
        else:
            nn_arch = nn_arch.to(device)
        if embedding_type == 'learnable':
            if adapted_params == None:
                outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device), zcp.to(device))
            else:
                outputs = net(nn_arch, torch.LongTensor([[zz] for zz in hw_idx[0].tolist()]).to(device), zcp.to(device), params=adapted_params)
        else:
            outputs = net(nn_arch, hw_idx.to(device), zcp.to(device))
        latency_list.append([x[0] for x in outputs.detach().tolist()])
    return np.asarray([item for sublist in latency_list for item in sublist])


def get_closest_correlator(s, archs, test_device_name, train_devices, latency_data_path):
    dev_corr_list = []
    for idx, train_device_name in enumerate(train_devices):
        latency_arr_test = normalization(np.asarray(torch.load(os.path.join(latency_data_path, test_device_name + ".pt"))), portion=1)
        latency_arr_train = normalization(np.asarray(torch.load(os.path.join(latency_data_path, train_device_name + ".pt"))), portion=1)
        sample_idxs = random.sample(range(len(archs)), s)
        corr = spearmanr(latency_arr_train[sample_idxs], latency_arr_test[sample_idxs])[0]
        dev_corr_list.append(corr)
        if corr >= max(dev_corr_list):
            dev_corr_idx = train_device_name
    return dev_corr_idx


def transfer_embedding(net, train_devices, transfer_devices, hw_emb_dim, device_idx_mapper, dev_corr_idx, adapt_new_embedding, device):
    embedding_to_transfer = net.dev_emb.weight.data
    transfer_device_embedding_init = {}
    for idx, transfer_device_name in enumerate(transfer_devices):
        closest_correlated_device = dev_corr_idx[transfer_device_name]
        if adapt_new_embedding:
            transfer_device_embedding_init[transfer_device_name] = net.dev_emb.weight.data[device_idx_mapper[closest_correlated_device]]
        else:
            adapt_shape = net.dev_emb.weight.data[device_idx_mapper[closest_correlated_device]].shape
            transfer_device_embedding_init[transfer_device_name] = torch.nn.Embedding(shape).weight.data
    net.dev_emb = torch.nn.Embedding(len(train_devices) + len(transfer_devices), hw_emb_dim).to(device)
    net.dev_emb.weight.data = torch.cat((embedding_to_transfer, torch.stack(list(transfer_device_embedding_init.values())).to(device)), 0)
    net.to(device)
    return net

def get_minmax_latency_index(meta_train_devices, train_idx, latency):
    rank = {}
    cnt = {}
    for device in meta_train_devices:
        lat, rank_idx = torch.sort(latency[device][train_idx[device]])
        for r, t in zip(rank_idx, train_idx[device]):
            t = t.item()
            if not t in rank.keys():
                rank[t] = 0
                cnt[t] = 0
            rank[t] += r
            cnt[t] += 1

    max_lat_rank = -10000000
    max_lat_idx = None
    min_lat_rank = 100000000
    min_lat_idx = None
    for (t, r), c in zip(rank.items(), cnt.values()):
        if c < len(meta_train_devices):
            continue
        if r > max_lat_rank:
            max_lat_rank = r
            max_lat_idx = t 
        if r < min_lat_rank:
            min_lat_rank = r 
            min_lat_idx = t
    return max_lat_idx, min_lat_idx


def log_prob(dist, groundtruth):
    log_p = dist.log_prob(groundtruth)
    return -log_p.mean()

loss_fn = {
            'mse': lambda yq_hat, yq,: F.mse_loss(yq_hat, yq),
            'logprob': lambda yq_hat, yq: log_prob(dist, yq)
            }

def flat(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().reshape(-1)
    else:
        return v.reshape(-1)

metrics_fn = {
            'spearman': lambda yq_hat, yq: spearmanr(flat(yq_hat), flat(yq)),
            'pearsonr': lambda yq_hat, yq: pearsonr(flat(yq_hat), flat(yq)),
            'kendalltau': lambda yq_hat, yq: kendalltau(flat(yq_hat), flat(yq))
            }


class Log():
    def __init__(self, save_path, summary_steps, metrics, devices, split, writer=None, use_wandb=False):
        self.save_path = save_path
        self.metrics = metrics
        self.devices = devices
        self.summary_steps = summary_steps
        self.split = split
        self.writer = writer

        self.epi = []
        self.elems = {}
        for metric in metrics:  
            self.elems[metric] = { device: [] for device in devices }
        self.elems['loss'] = { device: [] for device in devices }
        self.elems['mse_loss'] = { device: [] for device in devices }
        self.elems['kl_loss'] = { device: [] for device in devices }
        # self.elems['denorm_mse'] = { device: [] for device in devices }

        self.use_wandb = use_wandb

    def update_epi(self, i_epi):
        self.epi.append(i_epi)

    def update(self, i_epi, metric, device, val):
        self.elems[metric][device].append(val)
        if self.use_wandb:
            log_dict = {f'{self.split}_{metric}/{device}': val}
            wandb.log(log_dict, step=i_epi)
        if self.writer is not None:
            self.writer.add_scalar(f'{self.split}_{metric}/{device}', val, i_epi)  

    def avg(self, i_epi, metric, is_print=True):
        v = 0.0
        cnt = 0
        for device in self.devices:
            v += self.get(metric, device, i_epi)
            cnt += 1     
        if self.use_wandb:
            log_dict = {f'mean/{self.split}_{metric}': v / cnt}
            wandb.log(log_dict, step=i_epi)
        if self.writer is not None and is_print:
            self.writer.add_scalar(f'mean/{self.split}_{metric}', v / cnt, i_epi)
        return v / cnt
    
    # def last(self, metric, device):
    #     return self.elems[metric][device][-1]

    def get(self, metric, device, i_epi):
        idx = self.epi.index(i_epi)
        return self.elems[metric][device][idx]

    def save(self):
        torch.save({
                    'summary_steps': self.summary_steps,
                    'episode': self.epi,
                    'elems': self.elems
                    }, 
                    os.path.join(self.save_path, f'{self.split}_log_data.pt'))


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

def denorm(lat, maxv, minv):
    return lat * (maxv-minv) + minv

def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency

def arch_enc(arch):
    feature=[]
    for i in arch:
        onehot = np.zeros(6)
        if i == 8 :
            feature = np.hstack([feature, onehot])
        else :
            if i < 4:
                onehot[0] = 1
            elif i < 8:
                onehot[1] = 1
            k = i % 4
            onehot[2+k] = 1
            feature = np.hstack([feature, onehot])
    assert len(feature) == 132
    return torch.FloatTensor(feature)
    
    
def add_global_node( mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return torch.FloatTensor(mx)

def padzero( mx, ifAdj, maxsize=7):
    if ifAdj:
        while mx.shape[0] < maxsize:
            mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    else:
        while mx.shape[0] < maxsize:
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    return mx


def arch_encoding_ofa(arch):
    # This function converts a network config to a feature vector (128-D).
    ks_list, ex_list, d_list, r = copy.deepcopy(arch['ks']), copy.deepcopy(arch['e']), copy.deepcopy(arch['d']), arch['r']
    
    ks_map = {}
    ks_map[3]=0
    ks_map[5]=1
    ks_map[7]=2
    ex_map = {}
    ex_map[3]=0
    ex_map[4]=1
    ex_map[6]=2
    
    
    start = 0
    end = 4
    for d in d_list:
        for j in range(start+d, end):
            ks_list[j] = 0
            ex_list[j] = 0
        start += 4
        end += 4

    # convert to onehot
    ks_onehot = [0 for _ in range(60)]
    ex_onehot = [0 for _ in range(60)]
    r_onehot = [0 for _ in range(25)] #128 ~ 224

    for i in range(20):
        start = i * 3
        if ks_list[i] != 0:
            ks_onehot[start + ks_map[ks_list[i]]] = 1
        if ex_list[i] != 0:
            ex_onehot[start + ex_map[ex_list[i]]] = 1

    r_onehot[(r - 128) // 4] = 1
    return torch.Tensor(ks_onehot + ex_onehot + r_onehot)


def data_norm(v, src, des):
    min_s = min(src)
    max_s = max(src)
    min_d = min(des)
    max_d = max(des)
    nv = (v-min_s) / (max_s-min_s)
    nv = nv *(max_d-min_d) + min_d
    return nv
