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

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--crosstask_m', type=str, default='micro')
str_args = parser.parse_args()

device = 'cuda:0'
def set_seed(seed):
    # Set the random seed for reproducible experiments
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if not os.path.exists('fs_zcp_crosstask'):
    os.makedirs('fs_zcp_crosstask')

seed = 10
set_seed(seed)


dataset = pd.read_csv('./unified_dataset/cross_domain_superset.csv', header=None).to_numpy()

name_to_shorthand_map = {'zc_transbench101_macro_class_scene'         : 'ma_CS',    
                         'zc_transbench101_macro_class_object'        : 'ma_CO',    
                         'zc_transbench101_macro_autoencoder'         : 'ma_AE',    
                         'zc_transbench101_macro_normal'              : 'ma_N' ,
                         'zc_transbench101_macro_jigsaw'              : 'ma_J' ,
                         'zc_transbench101_macro_room_layout'         : 'ma_RL',    
                         'zc_transbench101_macro_segmentsemantic'     : 'ma_SS',        
                         'zc_transbench101_micro_class_scene'         : 'mi_CS',    
                         'zc_transbench101_micro_class_object'        : 'mi_CO',    
                         'zc_transbench101_micro_autoencoder'         : 'mi_AE',    
                         'zc_transbench101_micro_normal'              : 'mi_N' ,
                         'zc_transbench101_micro_jigsaw'              : 'mi_J' ,
                         'zc_transbench101_micro_room_layout'         : 'mi_RL',    
                         'zc_transbench101_micro_segmentsemantic'     : 'mi_SS',        
                         'zc_transbench101_micro_ninapro'             : 'mi_NP',
                         'zc_transbench101_micro_svhn'                : 'mi_SV',
                         'zc_transbench101_micro_scifar100'           : 'mi_SC'}

name_map = {
'zc_nasbench101_cifar10':     'NB1_CF10',
'zc_nasbench201_cifar10':     'NB2_CF10',
'zc_nasbench201_cifar100':    'NB2_CF100',
'zc_nasbench201_ImageNet16':  'NB2_IM16',
'zc_nasbench201_ninapro':     'NB2_NP',
'zc_nasbench201_svhn':        'NB2_SVHN',
'zc_nasbench201_scifar100':   'NB2_SC100',
'zc_nasbench301_cifar10':     'NB3_CF10',
'zc_nasbench301_ninapro':     'NB3_NP',
'zc_nasbench301_svhn':        'NB3_SVHN',
'zc_nasbench301_scifar100':   'NB3_SC100',
'zc_transbench101_macro_class_scene':'MA-SCENE',
'zc_transbench101_macro_class_object': 'MA-OBJECT',
'zc_transbench101_macro_autoencoder': 'MA-AUTOENC',
'zc_transbench101_macro_normal': 'MA-NORMAL',
'zc_transbench101_macro_jigsaw': 'MA-JIGSAW',
'zc_transbench101_macro_room_layout': 'MA-ROOM',
'zc_transbench101_macro_segmentsemantic': 'MA-SEGMENT',
'zc_transbench101_micro_class_scene': 'MI-SCENE',
'zc_transbench101_micro_class_object': 'MI-OBJECT',
'zc_transbench101_micro_autoencoder': 'MI-AUTOENC',
'zc_transbench101_micro_normal': 'MI-NORMAL',
'zc_transbench101_micro_jigsaw': 'MI-JIGSAW',
'zc_transbench101_micro_room_layout': 'MI-ROOM',
'zc_transbench101_micro_segmentsemantic': 'MI-SEGMENT',
'zc_transbench101_micro_ninapro': 'MI-NINAPRO',
'zc_transbench101_micro_svhn': 'MI-SVHN',
'zc_transbench101_micro_scifar100': 'MI-SCIFAR100'}

dataset_to_tuplemap = {
'zc_nasbench101_cifar10': (0, 0),
'zc_nasbench201_cifar10': (1, 0),
'zc_nasbench201_cifar100': (1, 1),
'zc_nasbench201_ImageNet16': (1, 2),
'zc_nasbench201_ninapro': (1, 3),
'zc_nasbench201_svhn': (1, 4),
'zc_nasbench201_scifar100': (1, 5),
'zc_nasbench301_cifar10': (2, 0),
'zc_nasbench301_ninapro': (2, 1),
'zc_nasbench301_svhn': (2, 2),
'zc_nasbench301_scifar100': (2, 3),
'zc_transbench101_macro_class_scene': (3, 0),
'zc_transbench101_macro_class_object': (3, 1),
'zc_transbench101_macro_autoencoder': (3, 2),
'zc_transbench101_macro_normal': (3, 3),
'zc_transbench101_macro_jigsaw': (3, 4),
'zc_transbench101_macro_room_layout': (3, 5),
'zc_transbench101_macro_segmentsemantic': (3, 6),
'zc_transbench101_micro_class_scene': (4, 0),
'zc_transbench101_micro_class_object': (4, 1),
'zc_transbench101_micro_autoencoder': (4, 2),
'zc_transbench101_micro_normal': (4, 3),
'zc_transbench101_micro_jigsaw': (4, 4),
'zc_transbench101_micro_room_layout': (4, 5),
'zc_transbench101_micro_segmentsemantic': (4, 6),
'zc_transbench101_micro_ninapro': (4, 7),
'zc_transbench101_micro_svhn': (4, 8),
'zc_transbench101_micro_scifar100': (4, 9)}

crosstask_m = str_args.crosstask_m
# do_transfer = False

if crosstask_m == 'micro':
    training_spaces = ['zc_transbench101_micro_class_scene', 'zc_transbench101_micro_class_object', 'zc_transbench101_micro_autoencoder', 'zc_transbench101_micro_normal', 'zc_transbench101_micro_jigsaw', 'zc_transbench101_micro_room_layout', 'zc_transbench101_micro_segmentsemantic', 'zc_transbench101_micro_ninapro', 'zc_transbench101_micro_svhn', 'zc_transbench101_micro_scifar100']
    testing_sets = training_spaces
elif crosstask_m =='micro_to_macro':
    training_spaces = ['zc_transbench101_micro_class_scene', 'zc_transbench101_micro_class_object', 'zc_transbench101_micro_autoencoder', 'zc_transbench101_micro_normal', 'zc_transbench101_micro_jigsaw', 'zc_transbench101_micro_room_layout', 'zc_transbench101_micro_segmentsemantic', 'zc_transbench101_micro_ninapro', 'zc_transbench101_micro_svhn', 'zc_transbench101_micro_scifar100']
    testing_sets = ['zc_transbench101_macro_class_scene', 'zc_transbench101_macro_class_object', 'zc_transbench101_macro_autoencoder', 'zc_transbench101_macro_normal', 'zc_transbench101_macro_jigsaw', 'zc_transbench101_macro_room_layout', 'zc_transbench101_macro_segmentsemantic']
elif crosstask_m == 'macro_to_micro':
    training_spaces = ['zc_transbench101_macro_class_scene', 'zc_transbench101_macro_class_object', 'zc_transbench101_macro_autoencoder', 'zc_transbench101_macro_normal', 'zc_transbench101_macro_jigsaw', 'zc_transbench101_macro_room_layout', 'zc_transbench101_macro_segmentsemantic']
    testing_sets = ['zc_transbench101_micro_class_scene', 'zc_transbench101_micro_class_object', 'zc_transbench101_micro_autoencoder', 'zc_transbench101_micro_normal', 'zc_transbench101_micro_jigsaw', 'zc_transbench101_micro_room_layout', 'zc_transbench101_micro_segmentsemantic', 'zc_transbench101_micro_ninapro', 'zc_transbench101_micro_svhn', 'zc_transbench101_micro_scifar100']
elif crosstask_m == 'nasbench':
    training_spaces = ['zc_nasbench101_cifar10','zc_nasbench201_cifar10','zc_nasbench301_cifar10']
    testing_sets = training_spaces
elif crosstask_m == 'all':
    training_spaces = ['zc_nasbench101_cifar10', 'zc_nasbench201_cifar10', 'zc_nasbench201_cifar100', 'zc_nasbench201_ImageNet16', 'zc_nasbench201_ninapro', 'zc_nasbench201_svhn', 'zc_nasbench201_scifar100', 'zc_nasbench301_cifar10', 'zc_nasbench301_ninapro', 'zc_nasbench301_svhn', 'zc_nasbench301_scifar100', 'zc_transbench101_macro_class_scene', 'zc_transbench101_macro_class_object', 'zc_transbench101_macro_autoencoder', 'zc_transbench101_macro_normal', 'zc_transbench101_macro_jigsaw', 'zc_transbench101_macro_room_layout', 'zc_transbench101_macro_segmentsemantic', 'zc_transbench101_micro_class_scene', 'zc_transbench101_micro_class_object', 'zc_transbench101_micro_autoencoder', 'zc_transbench101_micro_normal', 'zc_transbench101_micro_jigsaw', 'zc_transbench101_micro_room_layout', 'zc_transbench101_micro_segmentsemantic', 'zc_transbench101_micro_ninapro', 'zc_transbench101_micro_svhn', 'zc_transbench101_micro_scifar100']
    testing_sets = training_spaces
else:
    training_spaces = ['zc_transbench101_macro_class_scene', 'zc_transbench101_macro_class_object', 'zc_transbench101_macro_autoencoder', 'zc_transbench101_macro_normal', 'zc_transbench101_macro_jigsaw', 'zc_transbench101_macro_room_layout', 'zc_transbench101_macro_segmentsemantic']
    testing_sets = training_spaces

conf_mat_notransfer = {}
conf_mat_transfer = {}
repeat_times = 3
samp_eff = 20
for do_transfer in [True, False]:
    for train_space in tqdm(training_spaces):
        training_sets = [train_space]
        conf_mat_row_notransfer = {}
        conf_mat_row_transfer = {}
        for transfer_s in testing_sets:
            transfer_sets = [transfer_s]


            training_map = [dataset_to_tuplemap[x] for x in training_sets]
            transfer_map = [dataset_to_tuplemap[x] for x in transfer_sets]
            # All points with 0/0.5 in first index
            train_dataset = []

            print(train_space, ",", transfer_s, ",", samp_eff, end=",", flush=True)
            if do_transfer:
                rand_list = random.sample(range(len(dataset)), int(len(dataset)*0.3))
                for i in rand_list:
                    map_tup = (dataset[i][0], dataset[i][1])
                    if map_tup in training_map:
                        if random.randint(1, 10) >= 5:
                            train_dataset.append([torch.Tensor(dataset[i][2:-1].tolist()), dataset[i][-1]])

                train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

            corrlist = []

            train_epochs = 250
            transfer_epochs = 50
            
            db = dataset[dataset[:, 0]==transfer_map[0][0]]
            db = db[db[:,1]==transfer_map[0][1]]
            gap = db.shape[0]

            net = CrossDomainNet(n_zcproxies=12, feat_layersize=128, feat_depth=4).to(device)
            if do_transfer:
                optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-3, weight_decay=5.0e-4)
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
            for _ in range(repeat_times):
                net.load_state_dict(sd)
                sflag = 0
                # samp_eff points with 1 in first index
                transfer_dataset = []
                # Rest of the points with 1 in the first index
                test_dataset = []
                for i in range(len(dataset)):
                    map_tup = (dataset[i][0], dataset[i][1])
                    if map_tup in transfer_map:
                        if sflag==0:
                            low = i
                            sflag = 1
                        if len(transfer_dataset) < samp_eff:
                            num = random.randint(low, low+gap-1)
                            # num=i
                            transfer_dataset.append([torch.Tensor(dataset[num][2:-1].tolist()), dataset[num][-1]])
                        else:
                            test_dataset.append([torch.Tensor(dataset[i][2:-1].tolist()), dataset[i][-1]])

                transfer_loader = DataLoader(transfer_dataset, batch_size=samp_eff, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

                optimizer = torch.optim.AdamW(net.parameters(), lr=4.0e-4, weight_decay=5.0e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=transfer_epochs, eta_min=0.0)
                criterion = torch.nn.L1Loss(reduction='sum').to(device)
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
                if math.isnan(spearmanr(preds,truths).correlation):
                    pass
                else:
                    corrlist.append(spearmanr(preds,truths).correlation)
            print(np.mean(corrlist), ",", np.std(corrlist))
            if do_transfer==False:
                conf_mat_row_notransfer[transfer_s] = np.mean(corrlist)
            else:
                conf_mat_row_transfer[transfer_s] = np.mean(corrlist)
        if do_transfer==False:
            conf_mat_notransfer[train_space] = conf_mat_row_notransfer
        else:
            conf_mat_transfer[train_space] = conf_mat_row_transfer

print(conf_mat_notransfer)
print(conf_mat_transfer)
if True:
    diff_arr = pd.DataFrame(np.asarray(pd.DataFrame(conf_mat_transfer).values) - np.asarray(pd.DataFrame(conf_mat_notransfer).values))
    colnames = [name_map[x] for x in pd.DataFrame(conf_mat_transfer).columns]
    idxnames = [name_map[x] for x in pd.DataFrame(conf_mat_transfer).index]
    diff_arr.columns = colnames
    diff_arr.index = idxnames
    fig, ax = plt.subplots(figsize=(12,12))
    conf_mat_notransfer = pd.DataFrame.from_dict(conf_mat_notransfer)
    conf_mat_notransfer.columns = [name_map[x] for x in conf_mat_notransfer.columns]
    conf_mat_notransfer.index = [name_map[x] for x in conf_mat_notransfer.index]
    sns.heatmap(conf_mat_notransfer, annot=True, square=True, ax=ax, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./fs_zcp_crosstask/AUTOMLnr_' + str(repeat_times) + '_s' +str(samp_eff) + '_fig_9_notransfer_' + crosstask_m + ".png")
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,12))
    conf_mat_transfer = pd.DataFrame.from_dict(conf_mat_transfer)
    conf_mat_transfer.columns = [name_map[x] for x in conf_mat_transfer.columns]
    conf_mat_transfer.index = [name_map[x] for x in conf_mat_transfer.index]
    sns.heatmap(conf_mat_transfer, annot=True, square=True, ax=ax, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./fs_zcp_crosstask/AUTOMLnr_' + str(repeat_times) + '_s' +str(samp_eff) + '_fig_9_transfer_' + crosstask_m + ".png")
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(diff_arr, annot=True, square=True, ax=ax, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./fs_zcp_crosstask/AUTOMLnr_' + str(repeat_times) + '_s' +str(samp_eff) + '_fig_9_' + crosstask_m + ".png")
if True:
    diff_arr = pd.DataFrame((np.asarray(pd.DataFrame(conf_mat_transfer).values) - np.asarray(pd.DataFrame(conf_mat_notransfer).values))/np.asarray(pd.DataFrame(conf_mat_notransfer).values))
    diff_arr[diff_arr>=0.01] = 1
    diff_arr[diff_arr<0.01] = 0
    colnames = [x for x in pd.DataFrame(conf_mat_transfer).columns]
    idxnames = [x for x in pd.DataFrame(conf_mat_transfer).index]
    diff_arr.columns = colnames
    diff_arr.index = idxnames
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(diff_arr, annot=True, square=True, ax=ax, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./fs_zcp_crosstask/AUTOMLnr_' + str(repeat_times) + '_s' +str(samp_eff) + '_fig_9_binary_' + crosstask_m + ".png")
if True:
    diff_arr = pd.DataFrame((np.asarray(pd.DataFrame(conf_mat_transfer).values) - np.asarray(pd.DataFrame(conf_mat_notransfer).values))/np.asarray(pd.DataFrame(conf_mat_notransfer).values))
    diff_arr[diff_arr>=0.0] = 1
    diff_arr[diff_arr<0.0] = 0
    colnames = [x for x in pd.DataFrame(conf_mat_transfer).columns]
    idxnames = [x for x in pd.DataFrame(conf_mat_transfer).index]
    diff_arr.columns = colnames
    diff_arr.index = idxnames
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(diff_arr, annot=True, square=True, ax=ax, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('./fs_zcp_crosstask/AUTOMLno_thresh_nr_' + str(repeat_times) + '_s' +str(samp_eff) + '_fig_9_binary_' + crosstask_m + ".png")