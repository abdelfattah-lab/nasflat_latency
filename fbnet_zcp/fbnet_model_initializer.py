import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import random
import numpy as np
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
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )

class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        self.conv = nn.Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias)
        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = nn.Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(C_in*expansion)

        self.conv2 = nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias)
        self.bn2 = nn.BatchNorm2d(C_in*expansion)


        self.conv3 = nn.Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias)
        self.bn3 = nn.BatchNorm2d(C_out)

        self.nl = nn.ReLU(inplace=True)


    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x):

        identity = x
        x = self.nl(self.bn1(self.conv1(x)))

        if self.groups > 1:
            x = self.shuffle(x)

        x = self.nl(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))

        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        return x

class Skip(nn.Module):
    def __init__(self, C_in, C_out, layer_id, stride=1):
        super(Skip, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

        self.layer_id = layer_id

        self.kernel_size = 1
        self.padding = 0

        if stride == 2 or C_in != C_out:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        if hasattr(self, 'conv'):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x

        return out


# The 9 blocks in FBNet Space
PRIMITIVES = [
    'k3_e1' ,
    'k3_e1_g2' ,
    'k3_e3' ,
    'k3_e6' ,
    'k5_e1' ,
    'k5_e1_g2',
    'k5_e3',
    'k5_e6',
    'skip'
]

OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2),
    'k3_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1),
    'k3_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1),
    'k5_e1' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2),
    'k5_e3' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1),
    'k5_e6' : lambda C_in, C_out, layer_id, stride: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1),
    'skip' : lambda C_in, C_out, layer_id, stride: Skip(C_in, C_out, layer_id, stride)
}

AUX_OPS = {
    'ConvNorm' : lambda C_in, C_out, layer_id, kernel_size, stride: ConvBlock(C_in, C_out, layer_id, kernel_size=kernel_size, stride=stride),
    'AvgP' : lambda C_in, C_out, layer_id, kernel_size, stride: nn.AdaptiveAvgPool2d(1),
    'FC' : lambda C_in, C_out: nn.Linear(C_in, C_out),
}

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, layer_id, stride=1):
        super(MixedOp, self).__init__()
        self.layer_id = layer_id
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, layer_id, stride)

    def forward(self, x):
        return self._op(x)


class FBNet_Infer(nn.Module):
    def __init__(self, config):
        super(FBNet_Infer, self).__init__()

        self.op_idx_list = config["op_idx_list"]
        assert len(self.op_idx_list) == 22, ("The length of op_idx_list should be 22, while it is {}".format(len(self.op_idx_list)))

        self.num_classes = config["num_classes"]

        self.num_layer_list = [1, 4, 4, 4, 4, 4, 1] # FBNetv1 setting
        self.num_channel_list = [16, 24, 32, 64, 112, 184, 352] # FBNetv1 setting

        if "cifar" in config["dataset"]:
            self.stride_list = [1, 1, 2, 2, 1, 2, 1] # FBNetv1 setting, unoffcial CIFAR-10/100 setting
        else:
            self.stride_list = [1, 2, 2, 2, 1, 2, 1] # FBNetv1 setting, offcial ImageNet setting


        self.stem_channel = 16 # FBNetv1 setting

        if "cifar" in config["dataset"]:
            self.header_channel = 1504 # FBNetv1 setting
        else:
            self.header_channel = 1984 # FBNetv1 setting

        if "cifar" in config["dataset"]:
            stride_init = 1
        else:
            stride_init = 2

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=stride_init, padding=1, bias=False)

        self.cells = nn.ModuleList()

        layer_id = 1
        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0: # first Conv takes the stride into consideration
                    if stage_id == 0:
                        # The first block in the first stage will use stem_channel as input channel
                        op = OPS[PRIMITIVES[self.op_idx_list[layer_id-1]]](self.stem_channel, self.num_channel_list[stage_id], layer_id, self.stride_list[stage_id])
                    else:
                        op = OPS[PRIMITIVES[self.op_idx_list[layer_id-1]]](self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], layer_id, self.stride_list[stage_id])
                else:
                    op = OPS[PRIMITIVES[self.op_idx_list[layer_id-1]]](self.num_channel_list[stage_id], self.num_channel_list[stage_id], layer_id, stride=1)

                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)
        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, input):

        out = self.stem(input)

        for i, cell in enumerate(self.cells):
            out = cell(out)

        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))

        return out

## Add argparser for start and end index
parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=10)
args = parser.parse_args()

archs = torch.load("/home/ya255/projects/flan_hardware/fbnet_zcp/arch.pt")
    
metrics = ['synflow', 'zen', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen']

stem_channel = 16 
num_layer_list = [1, 4, 4, 4, 4, 4, 1]
num_channel_list = [16, 24, 32, 64, 112, 184, 352]

ConvBlock_list = []
ConvBlock_list.append({"exp": 1, "kernel": 3, "group": 1})
ConvBlock_list.append({"exp": 1, "kernel": 3, "group": 2})
ConvBlock_list.append({"exp": 3, "kernel": 3, "group": 1})
ConvBlock_list.append({"exp": 6, "kernel": 3, "group": 1})
ConvBlock_list.append({"exp": 1, "kernel": 5, "group": 1})
ConvBlock_list.append({"exp": 1, "kernel": 5, "group": 2})
ConvBlock_list.append({"exp": 3, "kernel": 5, "group": 1})
ConvBlock_list.append({"exp": 6, "kernel": 5, "group": 1})

num_classes = 100
stride_list = [1, 1, 2, 2, 1, 2, 1]
header_channel = 1504 # FBNetv1 setting
stride_init = 1
H_W = 32

# for arch_ in archs:
#     config = {}
#     config["op_idx_list"] = arch_
#     config["num_classes"] = 100
#     config["dataset"] =  "cifar100"
#     net = FBNet_Infer(config)
#     print("Config: ", config, " Initialized")
#     # exit(0)

device = 'cpu'
batch_size = 16
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                        shuffle=True, num_workers=1)
inputs, targets = next(trainloader.__iter__())

inputs = inputs.to(device)
dummy_inputs = torch.randn(1, 3, 32, 32).to(device)
targets = targets.to(device)


def no_op(self, x):  # pylint: disable=unused-argument
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn is False:
        for l in net.modules():
            if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                l.forward = types.MethodType(no_op, l)
    return net

model_masterinfo = {}
for i in tqdm(archs[args.start_idx:args.end_idx]):
    try:
        measurements = {}
        scalar_measurement = {}
        a = time.time()
        for m in metrics:
            config = {}
            config["op_idx_list"] = i
            config["num_classes"] = 100
            config["dataset"] =  "cifar100"
            model = FBNet_Infer(config)
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
        model_masterinfo["_".join([str(kz) for kz in i])] = scalar_measurement
    except Exception as e:
        print(e)
        pass

with open('fbnet_zcps/%s.json' % (os.environ['SLURM_JOB_ID']), 'w') as f:
    json.dump(model_masterinfo, f)
