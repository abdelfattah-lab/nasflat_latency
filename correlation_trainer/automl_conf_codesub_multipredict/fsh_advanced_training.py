from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import math
from utils import *
from few_shot_net import FewShotPredictor, FewShotNoEmbeddingPredictor
from tqdm import tqdm
import random
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
import pickle
import json, sys
import os
import random
import pickle
import csv


sys.path.append('./../')

from device_task_list import HardwareDataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config_idx', type=int, default=0)
parser.add_argument('--name_desc', type=str, default="multipredict_default") 
parser.add_argument('--representation', type=str, default="zcp_vec") 
parser.add_argument('--space', type=str, default="fbnet")
parser.add_argument("--task_index", type=int, default=0)
# parser.add_argument('--train_devices', nargs='+', type=str, default=['1080ti_1','1080ti_32','1080ti_256','silver_4114','silver_4210r','samsung_a50','pixel3','essential_ph_1','samsung_s7'])
# parser.add_argument('--transfer_devices', nargs='+', type=str, default=['titan_rtx_256','gold_6226','fpga','pixel2','raspi4','eyeriss'])
parser.add_argument('--emb_transfer_samples', type=int, default=10)
parser.add_argument('--fsh_mc_sampling', type=int, default=10)
parser.add_argument('--dev_train_samples', type=int, default=900)
parser.add_argument('--num_trials', type=int, default=3)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--transfer_epochs', type=int, default=50)
str_args = parser.parse_args()

if not os.path.exists('training_results'):
    os.makedirs('training_results')

seed = 10
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print("Config IDX: ", str_args.config_idx)
all_config = read_config('base_config.json')

hw_taskset = HardwareDataset()
# args.space = 'nb201' if search_space=='nasbench201' else 'fbnet'
str_args.train_devices = hw_taskset.get_data(str_args.space, str_args.task_index)["train"]
str_args.transfer_devices = hw_taskset.get_data(str_args.space, str_args.task_index)["test"]

(representation, test_idx, search_space, num_trials, report, s, emb_transfer_samples, fsh_sampling_strat, fsh_mc_sampling, dev_train_samples, train_batchsize, test_batchsize, epochs, transfer_epochs, \
mixed_training, mixed_train_weight, hw_emb_dim, gcn_layer_size, nn_emb_dim, feat_layer_size, \
feat_depth, loss_function, train_device_list, transfer_device_list, use_specific_lr, \
transfer_specific_lr, embedding_type, closest_correlator, embedding_transfer, freeze_non_embedding, \
adapt_new_embedding, pre_train_transferset, device, cpu_map) = all_config


all_config = list(all_config)
all_config[2] = 'nasbench201' if str_args.space=='nb201' else 'fbnet'
all_config[22] = ','.join(str_args.train_devices)
train_device_list = ','.join(str_args.train_devices)
all_config[23] = ','.join(str_args.transfer_devices)
transfer_device_list = ','.join(str_args.transfer_devices)
all_config[12] = str_args.epochs
all_config[13] = str_args.transfer_epochs
all_config[3] = str_args.num_trials
all_config[9] = str_args.dev_train_samples
all_config[8] = str_args.fsh_mc_sampling
all_config[5] = str_args.emb_transfer_samples
all_config = tuple(all_config)

search_space = str_args.space
emb_transfer_samples   =  str_args.emb_transfer_samples
fsh_mc_sampling = str_args.fsh_mc_sampling
dev_train_samples = str_args.dev_train_samples
num_trials = str_args.num_trials
epochs = str_args.epochs
transfer_epochs = str_args.transfer_epochs

print(all_config)
# Append config_idx and all_config to a csv
with open('./config_mapper.log', 'a') as f:
    f.write(str(str_args.config_idx) + "," + str(os.environ['SLURM_JOB_ID']) + "," + ','.join([str(x) for x in all_config]))
    f.write("\n")

mc_sampling = fsh_mc_sampling

transfer_batchsize    =  emb_transfer_samples
train_devices         =  train_device_list.split(',')
transfer_devices      =  transfer_device_list.split(',')
reference_device_name =  train_devices[0] + ".pt"

device_list           =  get_full_device_list(search_space)
device_name_idx       =  {x: idx for idx, x in enumerate(device_list)}  
device_idx_mapper     =  {device_name: idx for idx, device_name in enumerate(train_devices + transfer_devices)}

repeat_times = num_trials
repeat_dict = {}
error_dict_1 = {}
error_dict_5 = {}
error_dict_10 = {}
trial_net_lats = {}

switch = True

if search_space == 'fbnet':
    zcp_data_path = './unified_dataset/fbnet_device_zcps'
    arch_data_path = './unified_dataset/help_latency_dataset/fbnet/'
    if switch==True:
        latency_data_path = './unified_dataset/HELP/%s/latency/' % (str(search_space))
    else:
        latency_data_path = zcp_data_path
elif search_space == 'nasbench201':
    zcp_data_path = './unified_dataset/nb201_device_zcps/'
    arch_data_path = './unified_dataset/help_latency_dataset/nasbench201/'
    if switch==True:
        latency_data_path = './unified_dataset/HELP/%s/latency/' % (str(search_space))
    else:
        latency_data_path = zcp_data_path


for repeated_trial_idx in range(repeat_times):
    print("Trial Number: ", repeated_trial_idx)
    archs = load_archs(arch_data_path, search_space)
    # If representation contains 'HWL', then initialize the HWL data-set as ZCP Arch (hack)
    if representation.__contains__('hwl'):
        hwl_set = []
        for dev in train_devices:
            hwl_set.append(torch.load(arch_data_path + '/latency/' + dev + '.pt'))
        hwl_set = pd.DataFrame(hwl_set).T
        hwl_set = [torch.Tensor(x) for x in hwl_set.to_numpy().tolist()]
        zcp_archs = hwl_set
    else:
        zcp_archs = load_zcp_archs(zcp_data_path)

    (train_latency_dict, transfer_latency_dict, full_latency_dict) = get_latency_dicts(latency_data_path, device_name_idx, train_devices, transfer_devices, device_list)
    
    device_idx_to_emb = emb_generator(embedding_type, hw_emb_dim, full_latency_dict, archs)

    train_loader, test_loader, _ = get_dataloader(representation, archs, zcp_archs, device_idx_mapper, train_devices,          \
                                            None, train_latency_dict, device_name_idx, device_idx_to_emb,      \
                                            dev_train_samples, embedding_type, hw_emb_dim, \
                                            train_batchsize, test_batchsize, search_space=search_space)
    
    net = load_net(train_loader, *(all_config))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0004, weight_decay=5.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    criterion = torch.nn.L1Loss(reduction='sum').to(device)
    ########################## Training ###########################
    net = train_net_brp_params(net, representation, embedding_type, epochs, train_loader, criterion, optimizer, scheduler, train_loader, device, search_space)

    ####################### Test All Devices #######################
    for idx, device_name in enumerate(train_devices):
        if switch:
            true_latency_array = normalization(np.asarray(torch.load(os.path.join(latency_data_path, device_name + ".pt"))), portion=1)
        else:
            true_latency_array = normalization(np.asarray(pd.read_csv(os.path.join(latency_data_path, device_name.replace(".pt", "") + ".csv"),header=None).iloc[:,-1].values), portion=1)
        device_dataloader = get_allarch_data(archs, zcp_archs, search_space, representation, device_name, latency_data_path, device_idx_mapper, embedding_type, device_name_idx, device_idx_to_emb)
        latency_array = get_latency(net, representation, device_dataloader, embedding_type, device, search_space=search_space)

    #################  Find Closest Correlator  ####################SS
    specific_train_dict = {}
    specific_test_dict = {}
    specific_train_idxs = {}
    for idx, transfer_device_name in enumerate(transfer_devices):
        device_transfer_specific_dict = {device_name_idx[transfer_device_name+'.pt']:  transfer_latency_dict[device_name_idx[transfer_device_name+'.pt']]}
        reference_device_name =  random.sample(train_devices, 1)[0] + ".pt"
        ref_uncorr_dev_latency_dict = {device_name_idx[reference_device_name]:  train_latency_dict[device_name_idx[reference_device_name]]}
        specific_train_dict[transfer_device_name], \
        specific_test_dict[transfer_device_name], specific_train_idxs[transfer_device_name] = get_dataloader(representation, archs, zcp_archs, device_idx_mapper, [transfer_device_name], \
                                                        None, device_transfer_specific_dict, device_name_idx, device_idx_to_emb, \
                                                        emb_transfer_samples, embedding_type, hw_emb_dim, \
                                                        emb_transfer_samples//2, test_batchsize, transfer_set=True, \
                                                        fsh_sampling_strat=fsh_sampling_strat, ref_uncorr_dev_latency_dict=ref_uncorr_dev_latency_dict, \
                                                        reference_device_name=reference_device_name, search_space=search_space)
    dev_corr_idx = {}
    for idx, test_device_name in enumerate(transfer_devices):
        dev_corr_idx[test_device_name] = get_closest_correlator(s, archs, test_device_name, train_devices, latency_data_path, specific_train_idxs[transfer_device_name])

    #############  Get Closest Correlator Accuracy  ################
    for idx, test_device_name in enumerate(transfer_devices):
        if switch:
            true_latency_array = normalization(np.asarray(torch.load(os.path.join(latency_data_path, test_device_name + ".pt"))), portion=1)
        else:
            true_latency_array = normalization(np.asarray(pd.read_csv(os.path.join(latency_data_path, test_device_name.replace(".pt", "") + ".csv"),header=None).iloc[:,-1].values), portion=1)
        device_to_use = dev_corr_idx[test_device_name]
        device_dataloader = get_allarch_data(archs, zcp_archs, search_space, representation, device_to_use, latency_data_path, device_idx_mapper, embedding_type, device_name_idx, device_idx_to_emb)
        latency_array = get_latency(net, representation, device_dataloader, embedding_type, device, search_space=search_space)
        if report=='closest_correlator':
            repeat_dict.setdefault(test_device_name, []).append(spearmanr(true_latency_array, latency_array).correlation)

    if report=='closest_correlator':
        continue
    ################### Modify Net Embedding  ######################
    if embedding_type == 'learnable':
        net = transfer_embedding(net, train_devices, transfer_devices, hw_emb_dim, device_idx_mapper, dev_corr_idx, adapt_new_embedding, device)
        if freeze_non_embedding==True:
            for param in net.parameters():
                param.requires_grad = False

            net.dev_emb.weight.requires_grad = True 
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=4.0e-4, weight_decay=5.0e-4)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
    if report=='joint_learned':
        continue
    ################# Specialize Net For Devices  ###################
    pre_transfer_net_dict = net.state_dict()
    
    for transfer_device_idx, transfer_device_name in enumerate(transfer_devices):
        mc_sampled_truth = []
        mc_sampled_preds = []
        device_transfer_specific_dict = {device_name_idx[transfer_device_name+'.pt']:  transfer_latency_dict[device_name_idx[transfer_device_name+'.pt']]}
        ref_uncorr_dev_latency_dict = {device_name_idx[reference_device_name]:  train_latency_dict[device_name_idx[reference_device_name]]}
        specific_transfer_train_loader, specific_transfer_test_loader = specific_train_dict[transfer_device_name], specific_test_dict[transfer_device_name]
        print("SFTLen: ", specific_transfer_train_loader.__len__())
        spec_dev_lat = [x[-1] for x in specific_transfer_train_loader.dataset]
        trial_net_lats.setdefault(transfer_device_name, []).append(spec_dev_lat)
        for _ in range(mc_sampling):
            net.load_state_dict(pre_transfer_net_dict)
            if use_specific_lr:
                transfer_specific_lr_list = [float(x) for x in transfer_specific_lr.split(",")]
                optimizer = torch.optim.AdamW(net.parameters(), lr=transfer_specific_lr_list[transfer_device_idx], weight_decay=5.0e-4)
            else:
                if search_space=='fbnet':
                    if embedding_type=='learnable':
                        net.fc1.weight.requires_grad = False
                        net.fc1.bias.requires_grad = False
                        net.fc2.weight.requires_grad = False
                        net.fc2.bias.requires_grad = False
                        net.fc3.weight.requires_grad = False
                        net.fc3.bias.requires_grad = False
                elif search_space=='nasbench201':
                    if embedding_type=='learnable':
                        net.gc1.weight.requires_grad = False
                        net.gc1.bias.requires_grad = False
                        net.gc2.weight.requires_grad = False
                        net.gc2.bias.requires_grad = False
                        net.gc3.weight.requires_grad = False
                        net.gc3.bias.requires_grad = False
                        net.gc4.weight.requires_grad = False
                        net.gc4.bias.requires_grad = False
                        net.fc3.weight.requires_grad = False
                        net.fc3.bias.requires_grad = False
                        net.fc4.weight.requires_grad = False
                        net.fc4.bias.requires_grad = False
                        net.fc5.weight.requires_grad = False
                        net.fc5.bias.requires_grad = False
                    else:
                        net.gc1.weight.requires_grad = False
                        net.gc1.bias.requires_grad = False
                        net.gc2.weight.requires_grad = False
                        net.gc2.bias.requires_grad = False
                        net.gc3.weight.requires_grad = False
                        net.gc3.bias.requires_grad = False
                        net.gc4.weight.requires_grad = False
                        net.gc4.bias.requires_grad = False
                        net.fc3.weight.requires_grad = False
                        net.fc3.bias.requires_grad = False
                        net.fc4.weight.requires_grad = False
                        net.fc4.bias.requires_grad = False
                        net.fc5.weight.requires_grad = False
                        net.fc5.bias.requires_grad = False
                    
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, weight_decay=5.0e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=transfer_epochs, eta_min=0.0)
            criterion = torch.nn.L1Loss(reduction='sum').to(device)
            net = train_net_brp_params(net, representation, embedding_type, transfer_epochs, specific_transfer_train_loader, criterion, optimizer, scheduler, train_loader, device, search_space)
            #################  Test Specific Device  ####################
            if switch:
                true_latency_array = normalization(np.asarray(torch.load(os.path.join(latency_data_path, transfer_device_name + ".pt"))), portion=1)
            else:
                true_latency_array = normalization(np.asarray(pd.read_csv(os.path.join(latency_data_path, transfer_device_name.replace(".pt", "")+".csv"),header=None).iloc[:,-1].values), portion=1)
            device_dataloader = get_allarch_data(archs, zcp_archs, search_space, representation, transfer_device_name, latency_data_path, device_idx_mapper, embedding_type, device_name_idx, device_idx_to_emb)
            latency_array = get_latency(net, representation, device_dataloader, embedding_type, device, search_space=search_space)
            mc_sampled_preds.append(latency_array)
            mc_sampled_truth.append(true_latency_array)
        if report=='adaptive_learned':

            # repeat_dict.setdefault(transfer_device_name, []).append(spearmanr(torch.tensor(mc_sampled_preds).sum(dim=0), torch.tensor(mc_sampled_truth).sum(dim=0)).correlation)
            repeat_dict.setdefault(transfer_device_name, []).append({'spr': spearmanr(torch.tensor(mc_sampled_preds).sum(dim=0), torch.tensor(mc_sampled_truth).sum(dim=0)).correlation, 'kdt': kendalltau(torch.tensor(mc_sampled_preds).sum(dim=0), torch.tensor(mc_sampled_truth).sum(dim=0)).correlation})

    
nrd = {}
for dev in repeat_dict.keys():
    spr_, kdt_ = [], []
    for tr_ in repeat_dict[dev]:
        spr_.append(tr_['spr'])
        kdt_.append(tr_['kdt'])
    nrd[dev] = {'spr': spr_, 'kdt': kdt_}

repeat_dict = nrd

uid = random.randint(0, 1000000000)

# Check if truecorrs directory exists, if not, create it.
if not os.path.exists('truecorrs'):
    os.makedirs('truecorrs')

# Check that uid does not exist as a .pkl file. If it does, change it
while os.path.exists('truecorrs/{}/{}.pkl'.format(str_args.name_desc, uid)):
    uid = random.randint(0, 1000000000)

# Save repeat_dict as a pickle file in a folder called "truecorrs"
if not os.path.exists('truecorrs/{}'.format(str_args.name_desc)):
    os.makedirs('truecorrs/{}'.format(str_args.name_desc))
with open('truecorrs/{}/{}.pkl'.format(str_args.name_desc, uid), 'wb') as f:
    pickle.dump(repeat_dict, f, pickle.HIGHEST_PROTOCOL)

if not os.path.exists('./../correlation_results/{}'.format(str_args.name_desc)):
    os.makedirs('./../correlation_results/{}'.format(str_args.name_desc))

filename = f'./../correlation_results/{str_args.name_desc}/nb201_samp_eff.csv'
header = "uid,name_desc,task_index,seed,source_devices,target_device,dev_train_samples,emb_transfer_samples,num_trials,spr,kdt,spr_std,kdt_std"
if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")

source_devices = "|".join(train_devices)

with open(filename, 'a') as f:
    for target_device in repeat_dict.keys():
        spr = np.mean(repeat_dict[target_device]['spr'])
        kdt = np.mean(repeat_dict[target_device]['kdt'])
        spr_std = np.std(repeat_dict[target_device]['spr'])
        kdt_std = np.std(repeat_dict[target_device]['kdt'])
        vals = [
            str(uid),
            str(str_args.name_desc),
            str(str_args.task_index),
            str(seed),
            source_devices,
            target_device,
            str(dev_train_samples),
            str(emb_transfer_samples),
            str(num_trials),
            str(spr),
            str(kdt),
            str(spr_std),
            str(kdt_std)
        ]
        f.write("%s\n" % ','.join(vals))

# Print device, spr, kdt, spr_std, kdt_std 
dev_spr_means, dev_kdt_means= [], []
for target_device in repeat_dict.keys():
    print(target_device, "\t\t", np.mean(repeat_dict[target_device]['spr']), np.mean(repeat_dict[target_device]['kdt']), np.std(repeat_dict[target_device]['spr']), np.std(repeat_dict[target_device]['kdt']))
    dev_spr_means.append(np.mean(repeat_dict[target_device]['spr']))
    dev_kdt_means.append(np.mean(repeat_dict[target_device]['kdt']))

print("Mean\t\t", np.mean(dev_spr_means), np.mean(dev_kdt_means), np.std(dev_spr_means), np.std(dev_kdt_means))

if not os.path.exists('./../correlation_results/aggr_{}'.format(str_args.name_desc)):
    os.makedirs('./../correlation_results/aggr_{}'.format(str_args.name_desc))

filename = f'./../correlation_results/aggr_{str_args.name_desc}/nb201_samp_eff.csv'

header = "uid,name_desc,seed,source_devices,dev_train_samples,emb_transfer_samples,num_trials,spr,kdt,spr_std,kdt_std"
if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")

source_devices = "|".join(train_devices)

with open(filename, 'a') as f:
    avg_spr = []
    avg_kdt = []
    avg_spr_std = []
    avg_kdt_std = []
    for target_device in repeat_dict.keys():
        avg_spr.append(np.mean(repeat_dict[target_device]['spr']))
        avg_kdt.append(np.mean(repeat_dict[target_device]['kdt']))
        avg_spr_std.append(np.std(repeat_dict[target_device]['spr']))
        avg_kdt_std.append(np.std(repeat_dict[target_device]['kdt']))    
    spr = np.mean(avg_spr)
    kdt = np.mean(avg_kdt)
    spr_std = np.mean(avg_spr_std)
    kdt_std = np.mean(avg_kdt_std)
    vals = [
        str(uid),
        str(str_args.name_desc),
        str(seed),
        source_devices,
        target_device,
        str(dev_train_samples),
        str(emb_transfer_samples),
        str(num_trials),
        str(spr),
        str(kdt),
        str(spr_std),
        str(kdt_std)
    ]
    f.write("%s\n" % ','.join(vals))