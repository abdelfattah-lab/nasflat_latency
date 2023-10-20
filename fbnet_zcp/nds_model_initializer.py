import random
import numpy as np
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype
import json
import torch
import os
from tqdm import tqdm 
import time

from torch.nn import functional as F
from torch import nn
import torchvision
import torchvision.transforms as transforms

from thop import profile
from flopth import flopth

import json

from zero_cost_proxies_utils import calc_measure
from zero_cost_proxies_utils.model_stats import (
    get_model_stats,
)
from zero_cost_proxies_utils.p_utils import (
    get_some_data,
    get_some_data_grasp,
)
import types
import copy
from tqdm import tqdm
import argparse
import random
import subprocess
import os

class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x)
                

def return_feature_layer(network, prefix=''):
    #for attr_str in dir(network):
    #    target_attr = getattr(network, attr_str)
    #    if isinstance(target_attr, torch.nn.Linear):
    #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')
             
class NDS:
    def __init__(self, searchspace):
        self.searchspace = searchspace
        data = json.load(open(f'NDS/nds_data/{searchspace}.json', 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data
    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def get_network_config(self, uid):
        return self.data[uid]['net']
    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']
    def get_network(self, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        #print(config)
        if 'genotype' in config:
            #print('geno')
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(config['width'], 1000, config['depth'], config['aux'],  genotype)
            else:
                network = NetworkCIFAR(config['width'], 10, config['depth'], config['aux'],  genotype)
            network.drop_path_prob = 0.
            #print(config)
            #print('genotype')
            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'
            #"res_stem_cifar": ResStemCifar,
            #"res_stem_in": ResStemIN,
            #"simple_stem_in": SimpleStemIN,
            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
        return_feature_layer(network)
        return network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return len(self.data)
    def random_arch(self):
        return random.randint(0, len(self.data)-1)
    def get_final_accuracy(self, uid):
        return 100.-self.data[uid]['test_ep_top1'][-1]


def no_op(self, x):  # pylint: disable=unused-argument
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn is False:
        for l in net.modules():
            if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                l.forward = types.MethodType(no_op, l)
    return net

valid_zcp_spacelist = ['Amoeba','DARTS','DARTS_fix-w-d','DARTS_lr-wd','ENAS','ENAS_fix-w-d','PNAS','PNAS_fix-w-d','NASNet']

import argparse
# Create an argparser for space id
parser = argparse.ArgumentParser(description='ZCP')
parser.add_argument('--space_id', type=int, default=0, help='space id')
parser.add_argument('--output_dir', type=str, default='.', help='output directory')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--num_hashes', type=int, default=10, help='number_of_models_to_eval')
args = parser.parse_args()

print("Space ID: ", args.space_id)
print("Space Name: ", valid_zcp_spacelist[args.space_id])
print("Output Directory: ", args.output_dir)
print("Random Seed: ", args.seed)

# # Set random seed
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)

batch_size = 16
s_space = NDS(valid_zcp_spacelist[args.space_id])

if not os.path.exists(str(valid_zcp_spacelist[args.space_id]) + '_zcps'):
    os.makedirs(str(valid_zcp_spacelist[args.space_id]) + '_zcps')

if not os.path.exists(str(valid_zcp_spacelist[args.space_id]) + '_zcp_jobids'):
    os.makedirs(str(valid_zcp_spacelist[args.space_id]) + '_zcp_jobids')

if not os.path.exists(str(valid_zcp_spacelist[args.space_id]) + '_executed_hashes'):
    os.makedirs(str(valid_zcp_spacelist[args.space_id]) + '_executed_hashes')

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                        shuffle=True, num_workers=1)
inputs, targets = next(trainloader.__iter__())

device = args.device
inputs = inputs.to(device)
dummy_inputs = torch.randn(1, 3, 32, 32).to(device)
targets = targets.to(device)

hash_iterator_list = [str(x) for x in range(s_space.__len__())]
success_hash_list = [str(x) for x in os.listdir(str(valid_zcp_spacelist[args.space_id]) + '_executed_hashes')]

if len(success_hash_list) == len(hash_iterator_list):
    print("All hashes already executed. Exiting.")
    exit(0)

current_jobids = str(subprocess.check_output('squeue --nohead --format %F --name bash | uniq', shell=True)).replace("b'", "").replace("'", "").split("\\n")
current_jobids = [x for x in current_jobids if x != '']
current_hash_list = []
for jobid in tqdm(current_jobids):
    try:
        current_hash_list += [x for x in open(str(valid_zcp_spacelist[args.space_id]) + "_zcp_jobids/%s" % (jobid), "r").read().split("\n")] 
    except Exception as e:
        print(e)
        pass

hashes_to_ignore = success_hash_list + current_hash_list
hash_iterator_list = [x for x in tqdm(hash_iterator_list) if x not in hashes_to_ignore]

jobs_ids = os.listdir("./" + str(valid_zcp_spacelist[args.space_id]) + "_zcps")

hashes_execute = random.sample(hash_iterator_list, args.num_hashes)
hashes_index = [hash_iterator_list.index(x) for x in hashes_execute]
with open(str(valid_zcp_spacelist[args.space_id]) + "_zcp_jobids/%s" % (os.environ['SLURM_JOB_ID']), "w") as f:
    f.write("\n".join(hashes_execute))

print("Hashes to execute: ", hashes_execute)



metrics = ['val_accuracy', 'synflow', 'zen', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']

model_masterinfo = {}
for i in tqdm(hashes_execute):
    i = int(i)
    try:
        measurements = {}
        scalar_measurement = {}
        a = time.time()
        for m in metrics:
            model = s_space.get_network(i)
            model.to(device)
            if not hasattr(model, "get_prunable_copy"):
                model.get_prunable_copy = types.MethodType(copynet, model)
            if m not in ['flops', 'params', 'val_accuracy']:
                measurements[m] = calc_measure(m, 
                                            model,
                                            device,
                                            inputs,
                                            targets,
                                            loss_fn=F.cross_entropy)
                if m in ['epe_nas', 'jacov', 'nwot', 'zen']:
                    scalar_measurement[m] = float(measurements[m])
                else:
                    scalar_measurement[m] = float(sum([x.sum() for x in measurements[m]]).item())
            else:
                flops, params = flopth(model, inputs=(dummy_inputs,), bare_number=True)
                if m=='flops': # Flops
                    scalar_measurement[m] = float(flops)
                elif m=='params': # Params
                    scalar_measurement[m] = float(params)
                else:
                    scalar_measurement[m] = float(s_space.get_final_accuracy(i))
        print("Total time: ", time.time() - a)
        model_masterinfo[i] = scalar_measurement
    except Exception as e:
        print(e)
        pass

with open(str(valid_zcp_spacelist[args.space_id]) + '_zcps/%s.json' % (os.environ['SLURM_JOB_ID']), 'w') as f:
    json.dump(model_masterinfo, f)

# save a new file with the name of  each hash completed
for hash in hashes_execute:
    with open(str(valid_zcp_spacelist[args.space_id]) + "_executed_hashes/%s" % (str(hash)), "w") as f:
        f.write(" ")

