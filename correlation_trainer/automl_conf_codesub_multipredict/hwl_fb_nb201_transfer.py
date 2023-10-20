from cross_domain_net import CrossDomainNet
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import math
from utils_transfer import *
from tqdm import tqdm
import random
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
import pickle
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if not os.path.exists('nb_fb_hwl_transfer_results'):
    os.makedirs('nb_fb_hwl_transfer_results')

if not os.path.exists('nb_fb_hwl_results'):
    os.makedirs('nb_fb_hwl_results')

if not os.path.exists('nb_fb_vec_results'):
    os.makedirs('nb_fb_vec_results')

if not os.path.exists('cross_arch_nb_fb_hwl_vec'):
    os.makedirs('cross_arch_nb_fb_hwl_vec')

parser = argparse.ArgumentParser()
parser.add_argument('--config_idx', type=int, default=0)
parser.add_argument('--mode', type=str, default='nb_to_fb', choices=['nb_to_fb', 'nb_to_nb', 'fb_to_fb', 'fb_to_nb'])
parser.add_argument('--training_device', type=str, default='pixel2')
parser.add_argument('--transfer_device', type=str, default='pixel2')
str_args = parser.parse_args()
samp_testing = [2, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]
devices = ['fpga', 'raspi4', 'eyeriss', 'val_accuracy'][:-1]

mode            = str_args.mode
training_device = str_args.training_device
transfer_device = str_args.transfer_device
device = 'cuda:0'
def set_seed(seed):
    # Set the random seed for reproducible experiments
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed = 10
set_seed(seed)
repeat_times = 5

master_result_dict = {}

all_devices = ['fpga', 'raspi4', 'eyeriss', 'pixel2', 'silver_4210r', 'titanx_1', 'titan_rtx_32', 'samsung_s7', 'titan_rtx_1', '1080ti_1', 'essential_ph_1', 'silver_4114', 'gold_6240', 'titanxp_32', 'gold_6226', 'titanxp_1', '2080ti_1', 'samsung_a50', '2080ti_32', 'titanx_32', 'pixel3', '1080ti_32']

# all_devices = all_devices + [str_args.training_device, str_args.transfer_device]

latency_data_path = './unified_dataset/help_latency_dataset/%s/latency/' % (str('fbnet'))
fbnet_latency_dict = {device_name: np.asarray(torch.load(os.path.join(latency_data_path, device_name + '.pt'))) for device_name in all_devices}
for k,v in fbnet_latency_dict.items():
    fbnet_latency_dict[k] = (fbnet_latency_dict[k] - fbnet_latency_dict[k].min(0)) / fbnet_latency_dict[k].ptp(0)

latency_data_path = './unified_dataset/help_latency_dataset/%s/latency/' % (str('nasbench201'))
nasbench201_latency_dict = {device_name: np.asarray(torch.load(os.path.join(latency_data_path, device_name + '.pt'))) for device_name in all_devices}
for k,v in nasbench201_latency_dict.items():
    nasbench201_latency_dict[k] = (nasbench201_latency_dict[k] - nasbench201_latency_dict[k].min(0)) / nasbench201_latency_dict[k].ptp(0)

fbnet_latency_data = pd.DataFrame(fbnet_latency_dict)
nasbench201_latency_data = pd.DataFrame(nasbench201_latency_dict)

cross_dev_corrs = pd.DataFrame.from_dict({y: {x: spearmanr(nasbench201_latency_dict[x], nasbench201_latency_dict[y]).correlation for x in all_devices} for y in all_devices})
transfer_dev_correlations = cross_dev_corrs[transfer_device].to_dict()
# Sort transfer_dev_correlations by value
transfer_dev_correlations = {k: v for k, v in sorted(transfer_dev_correlations.items(), key=lambda item: item[1])}

all_devices = list(transfer_dev_correlations.keys())[:16]  + [str_args.transfer_device]

latency_data_path = './unified_dataset/help_latency_dataset/%s/latency/' % (str('fbnet'))
fbnet_latency_dict = {device_name: np.asarray(torch.load(os.path.join(latency_data_path, device_name + '.pt'))) for device_name in all_devices}
for k,v in fbnet_latency_dict.items():
    fbnet_latency_dict[k] = (fbnet_latency_dict[k] - fbnet_latency_dict[k].min(0)) / fbnet_latency_dict[k].ptp(0)

latency_data_path = './unified_dataset/help_latency_dataset/%s/latency/' % (str('nasbench201'))
nasbench201_latency_dict = {device_name: np.asarray(torch.load(os.path.join(latency_data_path, device_name + '.pt'))) for device_name in all_devices}
for k,v in nasbench201_latency_dict.items():
    nasbench201_latency_dict[k] = (nasbench201_latency_dict[k] - nasbench201_latency_dict[k].min(0)) / nasbench201_latency_dict[k].ptp(0)

fbnet_latency_data = pd.DataFrame(fbnet_latency_dict)
nasbench201_latency_data = pd.DataFrame(nasbench201_latency_dict)

map_ss = {'nb': 'nasbench201', 'fb': 'fbnet'}
result_folder = open("nb_fb_hwl_transfer_results/" + str(mode) + "_" + str(training_device) + "_" + str(transfer_device) + ".txt", "w")
var_folder = open("nb_fb_hwl_transfer_results/var_" + str(mode) + "_" + str(training_device) + "_" + str(transfer_device) + ".txt", "w")
if mode[:2]=='nb':
    training_target = nasbench201_latency_data[training_device].to_numpy()
    training_dataset = nasbench201_latency_data.drop(training_device, axis=1).to_numpy()
    training_dataset = np.append(training_dataset, training_target.reshape(-1, 1), axis=1)
else:
    training_target = fbnet_latency_data[training_device].to_numpy()
    training_dataset = fbnet_latency_data.drop(training_device, axis=1).to_numpy()
    training_dataset = np.append(training_dataset, training_target.reshape(-1, 1), axis=1)
if mode[-2:]=='nb':
    transfer_target = nasbench201_latency_data[transfer_device].to_numpy()
    transfer_dataset = nasbench201_latency_data.drop(transfer_device, axis=1).to_numpy()
    transfer_dataset = np.append(transfer_dataset, transfer_target.reshape(-1, 1), axis=1)
else:
    transfer_target = fbnet_latency_data[transfer_device].to_numpy()
    transfer_dataset = fbnet_latency_data.drop(transfer_device, axis=1).to_numpy()
    transfer_dataset = np.append(transfer_dataset, transfer_target.reshape(-1, 1), axis=1)

training_size = 100

train_dataset_l = []

train_idxs = random.sample(range(0, len(training_dataset)), training_size)

for i in train_idxs:

    train_dataset_l.append([torch.Tensor(training_dataset[i][:-1].tolist()), training_dataset[i][-1]])

train_loader = DataLoader(train_dataset_l, batch_size=32, shuffle=True)

train_epochs = 250

net = CrossDomainNet(n_zcproxies=len(training_dataset[0][:-1].tolist()), feat_layersize=256, feat_depth=4).to(device)

optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=0.0)
criterion = torch.nn.L1Loss(reduction='sum').to(device)

net.train()
for epoch in range(train_epochs):
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = net(x)
        loss = criterion(y_pred, y.view(-1, 1))
        loss.backward()
        optimizer.step()
    scheduler.step()
sd = net.state_dict()

samp_eff_dict = {}
for samp_eff in samp_testing:
    corrs = []
    for _ in range(5):
        net.load_state_dict(sd)
        transfer_dataset_l = []
        test_dataset_l = []
        transfer_idxs = random.sample(range(0, len(transfer_dataset)), samp_eff)
        test_idxs = list(set(range(0, len(transfer_dataset))) - set(transfer_idxs))
        for i in transfer_idxs:
            transfer_dataset_l.append([torch.Tensor(transfer_dataset[i][:-1].tolist()), transfer_dataset[i][-1]])
        for i in test_idxs:
            test_dataset_l.append([torch.Tensor(transfer_dataset[i][:-1].tolist()), transfer_dataset[i][-1]])

        transfer_loader = DataLoader(transfer_dataset_l, batch_size = samp_eff, shuffle=True)
        test_loader = DataLoader(test_dataset_l, batch_size=128, shuffle=False)

        transfer_epochs = 50


        optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=0.0)
        criterion = torch.nn.L1Loss(reduction='sum').to(device)

        net.train()
        for epoch in range(transfer_epochs):
            for i, (x, y) in enumerate(transfer_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = net(x)
                loss = criterion(y_pred, y.view(-1, 1))
                loss.backward()
                optimizer.step()
            scheduler.step()


        net.eval()
        with torch.no_grad():
            preds = []
            truths = []
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_pred = net(x).flatten()
                preds.append(y_pred.cpu().numpy())
                truths.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        corrs.append(spearmanr(preds, truths).correlation)
    samp_eff_dict[samp_eff] = corrs
    print(samp_eff, ", ", np.mean(corrs), ", ", np.std(corrs))
    result_folder.write(str(samp_eff) + ", " + str(np.mean(corrs)) + ", " + str(np.std(corrs)))
    result_folder.write("\n")
    var_folder.write(str(samp_eff) +  "," +  ','.join([str(x) for x in corrs]) +  "\n")

master_result_dict["HWL Transfer"] = samp_eff_dict
var_folder.close()
result_folder.close()


training_device = str_args.training_device
transfer_device = str_args.transfer_device
result_folder = open("nb_fb_hwl_results/" + str(mode) + "_" + str(training_device) + "_" + str(transfer_device) + ".txt", "w")
var_folder = open("nb_fb_hwl_results/var_" + str(mode) + "_" + str(training_device) + "_" + str(transfer_device) + ".txt", "w")
if mode[:2]=='nb':
    training_target = nasbench201_latency_data[training_device].to_numpy()
    training_dataset = nasbench201_latency_data.drop(training_device, axis=1).to_numpy()
    training_dataset = np.append(training_dataset, training_target.reshape(-1, 1), axis=1)
else:
    training_target = fbnet_latency_data[training_device].to_numpy()
    training_dataset = fbnet_latency_data.drop(training_device, axis=1).to_numpy()
    training_dataset = np.append(training_dataset, training_target.reshape(-1, 1), axis=1)
if mode[-2:]=='nb':
    transfer_target = nasbench201_latency_data[transfer_device].to_numpy()
    transfer_dataset = nasbench201_latency_data.drop(transfer_device, axis=1).to_numpy()
    transfer_dataset = np.append(transfer_dataset, transfer_target.reshape(-1, 1), axis=1)
else:
    transfer_target = fbnet_latency_data[transfer_device].to_numpy()
    transfer_dataset = fbnet_latency_data.drop(transfer_device, axis=1).to_numpy()
    transfer_dataset = np.append(transfer_dataset, transfer_target.reshape(-1, 1), axis=1)

net = CrossDomainNet(n_zcproxies=len(training_dataset[0][:-1].tolist()), feat_layersize=256, feat_depth=4).to(device)
sd = net.state_dict()

samp_eff_dict = {}
for samp_eff in samp_testing:
    corrs = []
    for _ in range(5):
        net.load_state_dict(sd)
        transfer_dataset_l = []
        test_dataset_l = []
        transfer_idxs = random.sample(range(0, len(transfer_dataset)), samp_eff)
        test_idxs = list(set(range(0, len(transfer_dataset))) - set(transfer_idxs))
        for i in transfer_idxs:
            transfer_dataset_l.append([torch.Tensor(transfer_dataset[i][:-1].tolist()), transfer_dataset[i][-1]])
        for i in test_idxs:
            test_dataset_l.append([torch.Tensor(transfer_dataset[i][:-1].tolist()), transfer_dataset[i][-1]])

        transfer_loader = DataLoader(transfer_dataset_l, batch_size = samp_eff, shuffle=True)
        test_loader = DataLoader(test_dataset_l, batch_size=128, shuffle=False)

        transfer_epochs = 50


        optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=0.0)
        criterion = torch.nn.L1Loss(reduction='sum').to(device)

        net.train()
        for epoch in range(transfer_epochs):
            for i, (x, y) in enumerate(transfer_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = net(x)
                loss = criterion(y_pred, y.view(-1, 1))
                loss.backward()
                optimizer.step()
            scheduler.step()


        net.eval()
        with torch.no_grad():
            preds = []
            truths = []
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_pred = net(x).flatten()
                preds.append(y_pred.cpu().numpy())
                truths.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        corrs.append(spearmanr(preds, truths).correlation)
    samp_eff_dict[samp_eff] = corrs
    print(samp_eff, ", ", np.mean(corrs), ", ", np.std(corrs))
    result_folder.write(str(samp_eff) + ", " + str(np.mean(corrs)) + ", " + str(np.std(corrs)))
    result_folder.write("\n")
    var_folder.write(str(samp_eff) +  "," +  ','.join([str(x) for x in corrs]) +  "\n")

    
master_result_dict["HWL"] = samp_eff_dict
var_folder.close()
result_folder.close()


import torch
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

def load_archs(arch_data_path, search_space='fbnet'):
    return np.asarray([np.asarray(arch_enc(_['op_idx_list'])) for _ in 
            torch.load(os.path.join(arch_data_path, 'metainfo.pt'))['arch']])
search_space = 'fbnet'
fbnet_arch_data_path = './unified_dataset/help_latency_dataset/%s/' % (str('fbnet'))
fbnet_vec_data = load_archs(fbnet_arch_data_path, search_space)

# nasbench201_archs = 
if os.path.isfile("nasbench201_vec_data.csv")==False:
    from models.nasbench201 import utils
    from nas_201_api import NASBench201API as API
    arch_key_to_index = {}
    api = API('./unified_dataset/NAS-Bench-201-v1_1-096897.pth', verbose=False)
    for idx_arch in range(15625):
        arch_key = str(tuple(utils.get_arch_vector_from_arch_str(api[idx_arch])))
        arch_key_to_index[arch_key] = idx_arch
    nasbench201_vec_data = {}
    for ix in range(len(eval(arch_key))):
        metric = []
        for idx_arch in range(15625):
            arch_key = str(tuple(utils.get_arch_vector_from_arch_str(api[idx_arch])))
            metric.append(eval(arch_key)[ix])
        metric = np.asarray(metric)
        nasbench201_vec_data['vec_' + str(ix)] = (metric - metric.min(0))/metric.ptp(0)
    nasbench201_vec_data = pd.DataFrame(nasbench201_vec_data)
    nasbench201_vec_data.to_csv("nasbench201_vec_data.csv", index=False)
else:
    nasbench201_vec_data = pd.read_csv("nasbench201_vec_data.csv")

training_device = str_args.training_device
transfer_device = str_args.transfer_device
result_folder = open("nb_fb_vec_results/" + str(mode) + "_" + str(training_device) + "_" + str(transfer_device) + ".txt", "w")
var_folder = open("nb_fb_vec_results/var_" + str(mode) + "_" + str(training_device) + "_" + str(transfer_device) + ".txt", "w")
if mode[:2]=='nb':
    training_target = nasbench201_latency_data[training_device].to_numpy()
    training_dataset = nasbench201_vec_data.to_numpy()
    training_dataset = np.append(training_dataset, training_target.reshape(-1, 1), axis=1)
else:
    training_target = fbnet_latency_data[training_device].to_numpy()
    training_dataset = fbnet_vec_data
    training_dataset = np.append(training_dataset, training_target.reshape(-1, 1), axis=1)
if mode[-2:]=='nb':
    transfer_target = nasbench201_latency_data[transfer_device].to_numpy()
    transfer_dataset = nasbench201_vec_data.to_numpy()
    transfer_dataset = np.append(transfer_dataset, transfer_target.reshape(-1, 1), axis=1)
else:
    transfer_target = fbnet_latency_data[transfer_device].to_numpy()
    transfer_dataset = fbnet_vec_data
    transfer_dataset = np.append(transfer_dataset, transfer_target.reshape(-1, 1), axis=1)

net = CrossDomainNet(n_zcproxies=len(transfer_dataset[0][:-1].tolist()), feat_layersize=256, feat_depth=4).to(device)
sd = net.state_dict()

samp_eff_dict = {}
for samp_eff in samp_testing:
    corrs = []
    for _ in range(5):
        net.load_state_dict(sd)
        transfer_dataset_l = []
        test_dataset_l = []
        transfer_idxs = random.sample(range(0, len(transfer_dataset)), samp_eff)
        test_idxs = list(set(range(0, len(transfer_dataset))) - set(transfer_idxs))
        for i in transfer_idxs:
            transfer_dataset_l.append([torch.Tensor(transfer_dataset[i][:-1].tolist()), transfer_dataset[i][-1]])
        for i in test_idxs:
            test_dataset_l.append([torch.Tensor(transfer_dataset[i][:-1].tolist()), transfer_dataset[i][-1]])

        transfer_loader = DataLoader(transfer_dataset_l, batch_size = samp_eff, shuffle=True)
        test_loader = DataLoader(test_dataset_l, batch_size=128, shuffle=False)

        transfer_epochs = 50


        optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=0.0)
        criterion = torch.nn.L1Loss(reduction='sum').to(device)

        net.train()
        for epoch in range(transfer_epochs):
            for i, (x, y) in enumerate(transfer_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = net(x)
                loss = criterion(y_pred, y.view(-1, 1))
                loss.backward()
                optimizer.step()
            scheduler.step()


        net.eval()
        with torch.no_grad():
            preds = []
            truths = []
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_pred = net(x).flatten()
                preds.append(y_pred.cpu().numpy())
                truths.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        corrs.append(spearmanr(preds, truths).correlation)
    print(samp_eff, ", ", np.mean(corrs), ", ", np.std(corrs))
    samp_eff_dict[samp_eff] = corrs
    result_folder.write(str(samp_eff) + ", " + str(np.mean(corrs)) + ", " + str(np.std(corrs)))
    result_folder.write("\n")
    var_folder.write(str(samp_eff) +  "," +  ','.join([str(x) for x in corrs]) +  "\n")
var_folder.close()
result_folder.close()
master_result_dict["Vec"] = samp_eff_dict


from pylab import *
params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False
}
rcParams.update(params)


def perc(data):
   median = np.zeros(data.shape[1])
   perc_25 = np.zeros(data.shape[1])
   perc_75 = np.zeros(data.shape[1])
   for i in range(0, len(median)):
       median[i] = np.median(data[:, i])
       perc_25[i] = np.percentile(data[:, i], 25)
       perc_75[i] = np.percentile(data[:, i], 75)
   return median, perc_25, perc_75


master_var_chart = {}
for key, val in master_result_dict.items():
    median, perc25, perc75 = perc(pd.DataFrame(master_result_dict[key]).values)
    master_var_chart[key] = [median, perc25, perc75]

if True:
    x = np.array(samp_testing)
    colors = ['#006BB2', '#B22400', '#262626']
    plt.plot(x, master_var_chart['HWL Transfer'][0], marker='o', linewidth=2, color=colors[0], label='HWL Transfer')
    plt.plot(x, master_var_chart['HWL'][0], marker='o', linewidth=2, color=colors[1], label='HWL')
    plt.plot(x, master_var_chart['Vec'][0], marker='o', linewidth=2, color=colors[2], label='Vec')
    plt.fill_between(x, master_var_chart['HWL Transfer'][1], master_var_chart['HWL Transfer'][2], alpha=0.15, linewidth=0, color=colors[0])
    plt.fill_between(x, master_var_chart['HWL'][1], master_var_chart['HWL'][2], alpha=0.15, linewidth=0, color=colors[1])
    plt.fill_between(x, master_var_chart['Vec'][1], master_var_chart['Vec'][2], alpha=0.15, linewidth=0, color=colors[2])
    legend = plt.legend(["HWL Transfer", "HWL", "Vec"], loc=3)
    frame = legend.get_frame()
    frame.set_facecolor('1.0')
    frame.set_edgecolor('1.0')
    plt.ylim(master_var_chart['HWL'][1].min() - 0.2,1)
    plt.xscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Spearman Rank Correlation")
    plt.tight_layout()
    plt.savefig('./cross_arch_nb_fb_hwl_vec/' + str(str_args.mode) + "_" + str(str_args.training_device) + "_" + str(str_args.transfer_device) + '.png', dpi=400)
    plt.close()
    plt.cla()
    plt.clf()