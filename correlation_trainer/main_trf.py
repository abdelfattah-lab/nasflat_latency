import os
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
import argparse, sys, time, random, os
import numpy as np
from pprint import pprint
from tqdm import tqdm
from utils import CustomDataset, get_tagates_sample_indices
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

# python -i new_main.py --space nb101 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 72 --batch_size 8
# python -i new_main.py --space nb201 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 40 --batch_size 8

parser = argparse.ArgumentParser()
####################################################### Search Space Choices #######################################################
parser.add_argument('--space', type=str, default='nb201')         # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--source_devices', nargs='+', type=str, default=['1080ti_1','1080ti_32','1080ti_256','silver_4114','silver_4210r','samsung_a50','pixel3','essential_ph_1','samsung_s7'])
    # '1080ti_1', '1080ti_256', '1080ti_32', '2080ti_1', '2080ti_256', '2080ti_32', 'desktop_cpu_core_i7_7820x_fp32', 'desktop_gpu_gtx_1080ti_fp32',      \
    #    'embedded_gpu_jetson_nano_fp16', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'essential_ph_1', 'eyeriss', 'flops_nb201_cifar10', \
    #    'fpga', 'gold_6226', 'gold_6240', 'mobile_cpu_snapdragon_450_cortex_a53_int8', 'mobile_cpu_snapdragon_675_kryo_460_int8', 'mobile_cpu_snapdragon_855_kryo_485_int8', \
    #    'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_450_adreno_506_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', \
    #    'mobile_gpu_snapdragon_855_adreno_640_int8', 'nwot_nb201_cifar10', 'params_nb201_cifar10', 'pixel2', 'pixel3', 'raspi4', 'samsung_a50', 'samsung_s7', 'silver_4114', \
    #    'silver_4210r', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'titanx_1', 'titanx_256', 'titanx_32', 'titanxp_1', 'titanxp_256', 'titanxp_32'
    # If device is NOT None, space automatically becomes 'nb201'. 
    # Sample selection, hw encoding of nodes, few shot HW only.
    # Section 1   : Architectural
    # Section 2.1 : Network Sampling
    # Section 2.2 : Minimizing Pre-Training Samples
    # Section 3.1 : Device Embedding
    # Section 3.2 : Generalizability of predictor across 'n' devices vs fine tuning 
    # Section 4   : Transfer Learning Across Hardware Devices (with accuracy as well?)
    # Section 5   : Another SS? -> Can we look at NB101, NB301, NDS, TB101 if they have even a single device latency data?
    # BRAINSTORM:
    # For network sampling: use arch2vec, cate etc. [random, flops, params, arch2vec, cate, zcp, accuracy/latency (oracle)]
    # For device embedding, each operation may require a unique device embedding.
    #            So, use an embedding for each operation. (e.g. 5 ops, 5 embeddings per device) and concatenate appropriately.
parser.add_argument('--sampling_metric', type=str, default="random")
parser.add_argument('--metric_device', type=str, default="titanx_256")
parser.add_argument('--target_devices', nargs='+', type=str, default=['titan_rtx_256','gold_6226','fpga','pixel2','raspi4','eyeriss'])
# parser.add_argument('--sample_size', type=int, default=900)
parser.add_argument('--sample_sizes', nargs='+', type=int, default=[900]) # Default NB101
parser.add_argument('--transfer_sample_sizes', nargs='+', type=int, default=[5,10,20])
parser.add_argument('--task', type=str, default='class_scene')     # all tb101 tasks supported
parser.add_argument('--representation', type=str, default='cate')  # adj_mlp, adj_gin, zcp (except nb301), cate, arch2vec, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate supported. # adj_gin_org
parser.add_argument('--loss_type', type=str, default='pwl')        # mse, pwl supported
parser.add_argument('--gnn_type', type=str, default='dense')       # dense, gat, gat_mh supported
parser.add_argument('--hwemb_to_mlp', action="store_true")         # hw embedding at MLP stage if True, at per-node embedding if False (False should be better)
parser.add_argument('--transfer_hwemb', action="store_true")       # Init emb of new HW to closest training HW if True, else use random init for HW Emb (True should be better) #TODO IMPLEMENT TODAY
parser.add_argument('--num_trials', type=int, default=3)
parser.add_argument('--op_fp_gcn_out_dims', nargs='+', type=int, default=[128, 128])
parser.add_argument('--forward_gcn_out_dims', nargs='+', type=int, default=[128, 128, 128])
parser.add_argument('--backward_gcn_out_dims', nargs='+', type=int, default=[128, 128, 128])
parser.add_argument('--replace_bgcn_mlp_dims', nargs='+', type=int, default=[128, 128, 128])
parser.add_argument('--ensemble_fuse_method', type=str, default='add')               # add, mlp (Need to test)  # <DEPRECATED, HARDCODED>
parser.add_argument('--fb_conversion_dims', nargs='+', type=int, default=[128, 128])
###################################################### Other Hyper-Parameters ######################################################
parser.add_argument('--name_desc', type=str, default=None)
parser.add_argument('--cpu_gpu_device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--test_size', type=int, default=None)
parser.add_argument('--transfer_epochs', type=int, default=30)
parser.add_argument('--transfer_lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--lr_gamma', type=float, default=0.6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eta_min', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--id', type=int, default=0)
####################################################################################################################################
args = parser.parse_args()
# device = args.device
if args.representation.__contains__("adj_gin_org"):
    args.representation = args.representation.replace("adj_gin_org", "adj_gin")
    from models_abl import GIN_Model, FullyConnectedNN # Use original model with default set arguments
else:
    from models_abl_p2 import GIN_Model, FullyConnectedNN

if args.source_devices is not None:
    assert args.space in ["nb201"], "If device is not None, space MUST be nb201."

assert args.metric_device not in args.source_devices, "Metric device cannot be in source devices."
assert args.metric_device not in args.target_devices, "Metric device cannot be in target devices."

sample_tests, transfer_sample_tests = {}, {}
sample_tests[args.space] = args.sample_sizes
transfer_sample_tests[args.space] = args.transfer_sample_sizes

assert args.name_desc is not None, "Please provide a name description for the experiment."

# Set random seeds
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
if args.seed is not None:
    seed_everything(args.seed)

nb101_train_tagates_sample_indices, \
    nb101_tagates_sample_indices = get_tagates_sample_indices(args)

def flatten_mixed_list(pred_scores):
    flattened = []
    for sublist in pred_scores:
        if isinstance(sublist, (list, tuple)):  # Check if the item is iterable
            flattened.extend(sublist)  # If it's iterable, extend the flattened list
        else:
            flattened.append(sublist)  # If it's not iterable, append it directly
    return flattened

def pwl_train(args, model, dataloader, criterion, optimizer, scheduler, test_dataloader, epoch):
    model.training = True
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]  or args.representation.__contains__("adj_mlp"):
            if inputs.shape[0] == 1 and args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
            elif inputs.shape[0] <= 2 and args.space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
        else:
            if inputs[0].shape[0] == 1 and args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
            elif inputs[0].shape[0] <= 2 and args.space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
        #### Params for PWL Loss
        accs = targets
        max_compare_ratio = 4
        compare_threshold = 0.0
        max_grad_norm = None
        compare_margin = 0.1
        margin = [compare_margin]
        n = targets.shape[0]
        ###### 
        n_max_pairs = int(max_compare_ratio * n)
        acc_diff = np.array(accs)[:, None] - np.array(accs)
        acc_abs_difF_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_difF_matrix > compare_threshold)
        ex_thresh_nums = len(ex_thresh_inds[0])
        if ex_thresh_nums > n_max_pairs:
            keep_inds = np.random.choice(np.arange(ex_thresh_nums), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]  or args.representation.__contains__("adj_mlp"):
            archs_1 = [torch.stack(list((inputs[indx] for indx in ex_thresh_inds[1])))]
            archs_2 = [torch.stack(list((inputs[indx] for indx in ex_thresh_inds[0])))]
            X_input_1 = archs_1[0].to(dtype=torch.float32, device=args.cpu_gpu_device)
            s_1 = model(X_input_1).squeeze()
            X_input_2 = archs_2[0].to(dtype=torch.float32, device=args.cpu_gpu_device)
            s_2 = model(X_input_2).squeeze()
        elif args.representation in ["adj_gin"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0])))]
                X_adj_1, X_ops_1, norm_w_d_1, hw_idx = archs_1[0].to(args.cpu_gpu_device), archs_1[1].to(args.cpu_gpu_device), archs_1[2].to(args.cpu_gpu_device), archs_1[3].to(args.cpu_gpu_device)
                s_1 = model(x_ops_1=X_ops_1, x_adj_1=X_adj_1.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_2, X_ops_2, norm_w_d_2, hw_idx = archs_2[0].to(args.cpu_gpu_device), archs_2[1].to(args.cpu_gpu_device), archs_2[2].to(args.cpu_gpu_device), archs_2[3].to(args.cpu_gpu_device)
                s_2 = model(x_ops_1=X_ops_2, x_adj_1=X_adj_2.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=norm_w_d_2, hw_idx=hw_idx).squeeze()
            else:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[5][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[5][indx] for indx in ex_thresh_inds[0])))]
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, norm_w_d_1, hw_idx = archs_1[0].to(args.cpu_gpu_device), archs_1[1].to(args.cpu_gpu_device), archs_1[2].to(args.cpu_gpu_device), archs_1[3].to(args.cpu_gpu_device), archs_1[4].to(args.cpu_gpu_device)
                s_1 = model(x_ops_1=X_ops_a_1, x_adj_1=X_adj_a_1.to(torch.long), x_ops_2=X_ops_b_1, x_adj_2=X_adj_b_1.to(torch.long), zcp=None, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, norm_w_d_2, hw_idx = archs_2[0].to(args.cpu_gpu_device), archs_2[1].to(args.cpu_gpu_device), archs_2[2].to(args.cpu_gpu_device), archs_2[3].to(args.cpu_gpu_device), archs_2[4].to(args.cpu_gpu_device)
                s_2 = model(x_ops_1=X_ops_a_2, x_adj_1=X_adj_a_2.to(torch.long), x_ops_2=X_ops_b_2, x_adj_2=X_adj_b_2.to(torch.long), zcp=None, norm_w_d=norm_w_d_2, hw_idx=hw_idx).squeeze()
        elif args.representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[0])))]
                X_adj_1, X_ops_1, zcp, norm_w_d_1, hw_idx = archs_1[0].to(args.cpu_gpu_device), archs_1[1].to(args.cpu_gpu_device), archs_1[2].to(args.cpu_gpu_device), archs_1[3].to(args.cpu_gpu_device), archs_1[4].to(args.cpu_gpu_device)
                s_1 = model(x_ops_1=X_ops_1, x_adj_1=X_adj_1.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=zcp, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_2, X_ops_2, zcp, norm_w_d_2, hw_idx = archs_2[0].to(args.cpu_gpu_device), archs_2[1].to(args.cpu_gpu_device), archs_2[2].to(args.cpu_gpu_device), archs_2[3].to(args.cpu_gpu_device), archs_2[4].to(args.cpu_gpu_device)
                s_2 = model(x_ops_1=X_ops_2, x_adj_1=X_adj_2.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=zcp, norm_w_d=norm_w_d_2, hw_idx=hw_idx).squeeze()
            else:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[5][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[6][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[4][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[5][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[6][indx] for indx in ex_thresh_inds[0])))]
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, zcp, norm_w_d_1 = archs_1[0].to(args.cpu_gpu_device), archs_1[1].to(args.cpu_gpu_device), archs_1[2].to(args.cpu_gpu_device), archs_1[3].to(args.cpu_gpu_device), archs_1[4].to(args.cpu_gpu_device), archs_1[5].to(args.cpu_gpu_device), archs_1[6].to(args.cpu_gpu_device)
                s_1 = model(x_ops_1 = X_ops_a_1, x_adj_1 = X_adj_a_1.to(torch.long), x_ops_2 = X_ops_b_1, x_adj_2 = X_adj_b_1.to(torch.long), zcp = zcp, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, zcp, norm_w_d_2 = archs_2[0].to(args.cpu_gpu_device), archs_2[1].to(args.cpu_gpu_device), archs_2[2].to(args.cpu_gpu_device), archs_2[3].to(args.cpu_gpu_device), archs_2[4].to(args.cpu_gpu_device), archs_2[5].to(args.cpu_gpu_device), archs_2[6].to(args.cpu_gpu_device)
                s_2 = model(x_ops_1 = X_ops_a_2, x_adj_1 = X_adj_a_2.to(torch.long), x_ops_2 = X_ops_b_2, x_adj_2 = X_adj_b_2.to(torch.long), zcp = zcp, norm_w_d=norm_w_d_2, hw_idx=hw_idx).squeeze()
        else:
            raise NotImplementedError
        better_lst = (acc_diff>0)[ex_thresh_inds]
        better_pm = 2 * s_1.new(np.array(better_lst, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = s_1.new(margin)
        pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        optimizer.zero_grad()
        pair_loss.backward()
        optimizer.step()
        running_loss += pair_loss.item()
    scheduler.step()

    model.training = False
    model.eval()
    pred_scores, true_scores = [], []
    repr_max = int(80/args.test_batch_size)
    for repr_idx, (reprs, scores) in enumerate(tqdm(test_dataloader)):
        if epoch < args.epochs - 5 and repr_idx > repr_max:
            break
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"] or args.representation.__contains__("adj_mlp"):
            pred_scores.append(model(reprs.to(args.cpu_gpu_device, dtype=torch.float32)).squeeze().detach().cpu().tolist())
        elif args.representation in ["adj_gin"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to(args.cpu_gpu_device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=reprs[-2].to(args.cpu_gpu_device), hw_idx=reprs[-1].to(args.cpu_gpu_device)).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to(args.cpu_gpu_device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(args.cpu_gpu_device), x_adj_2=reprs[2].to(torch.long), zcp=None, norm_w_d=reprs[-2].to(args.cpu_gpu_device), hw_idx=reprs[-1].to(args.cpu_gpu_device)).squeeze().detach().cpu().tolist())
        elif args.representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to(args.cpu_gpu_device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=reprs[2].to(args.cpu_gpu_device), norm_w_d=reprs[-2].to(args.cpu_gpu_device), hw_idx=reprs[-1].to(args.cpu_gpu_device)).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to(args.cpu_gpu_device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(args.cpu_gpu_device), x_adj_2=reprs[2].to(torch.long), zcp=reprs[4].to(args.cpu_gpu_device), norm_w_d=reprs[-2].to(args.cpu_gpu_device), hw_idx=reprs[-1].to(args.cpu_gpu_device)).squeeze().detach().cpu().tolist())
        else:
            raise NotImplementedError
        true_scores.append(scores.cpu().tolist())
    # pred_scores = [t for sublist in pred_scores for t in sublist]
    # true_scores = [t for sublist in true_scores for t in sublist]
    pred_scores = flatten_mixed_list(pred_scores)
    true_scores = flatten_mixed_list(true_scores)
    num_test_items = len(pred_scores)
    return model, num_test_items, running_loss / len(dataloader), spearmanr(true_scores, pred_scores).correlation, kendalltau(true_scores, pred_scores).correlation

sys.path.append("..")
if args.space in ['Amoeba', 'DARTS', 'DARTS_fix-w-d', 'DARTS_lr-wd', 'ENAS', 'ENAS_fix-w-d', 'NASNet', 'PNAS', 'PNAS_fix-w-d']:
    from nas_embedding_suite.nds_ss import NDS as EmbGenClass
elif args.space in ['nb101', 'nb201', 'nb301']:
    exec("from nas_embedding_suite.nb{}_ss import NASBench{} as EmbGenClass".format(args.space[-3:], args.space[-3:]))
elif args.space in ['tb101']:
    from nas_embedding_suite.tb101_micro_ss import TransNASBench101Micro as EmbGenClass

embedding_gen = EmbGenClass(normalize_zcp=True, log_synflow=True)

# get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='test', indexes=transfer_samps, devices=[tfdevice])
# def get_dataloader(args, embedding_gen, space, sample_count, representation, mode, train_indexes=None, test_size=None):
def get_dataloader(args, embedding_gen, space, representation, mode, indexes, devices, batch_specified=None):
    representations = []
    accs = []
    sample_indexes = indexes
    for device in devices:
        if representation.__contains__("gin") == False: # adj_mlp, zcp, arch2vec, cate --> FullyConnectedNN
            if representation.__contains__("adj_mlp"): # adj_mlp --> FullyConnectedNN
                for i in tqdm(sample_indexes):
                    if space not in ["nb101", "nb201", "nb301", "tb101"]:
                        adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=space).values()
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        if device == None: accs.append(embedding_gen.get_valacc(i, space=space));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        adj_mat_norm = np.asarray(adj_mat_norm).flatten()
                        adj_mat_red = np.asarray(adj_mat_red).flatten()
                        op_mat_norm = torch.Tensor(np.asarray(op_mat_norm)).argmax(dim=1).numpy().flatten() # Careful here.
                        op_mat_red = torch.Tensor(np.asarray(op_mat_red)).argmax(dim=1).numpy().flatten() # Careful here.
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat_norm)).flatten() / len(embedding_gen.devices())
                        metric_val = np.asarray(eval('embedding_gen.get_{}(i, "{}")'.format(representation.replace("adj_mlp_", ""), args.task)))
                        representations.append(np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red, norm_w_d, hw_idx, metric_val)))
                    else:
                        adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                        if space == 'tb101':
                            if device == None: accs.append(embedding_gen.get_valacc(i, task=args.task));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        else:
                            if device == None: accs.append(embedding_gen.get_valacc(i));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        adj_mat = np.asarray(adj_mat).flatten()
                        op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten() # Careful here.
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat)).flatten()
                        metric_val = np.asarray(eval('embedding_gen.get_{}(i, "{}")'.format(representation.replace("adj_mlp_", ""), args.task)))
                        representations.append(np.concatenate((adj_mat, op_mat, norm_w_d, hw_idx, metric_val)))
            else:                           # zcp, arch2vec, cate --> FullyConnectedNN
                for i in tqdm(sample_indexes):
                    hw_idx = np.asarray([embedding_gen.get_device_index(device),] * 8).flatten()
                    if space in ['nb101', 'nb201', 'nb301']:
                        exec('representations.append(np.concatenate((embedding_gen.get_{}(i), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten(), hw_idx)))'.format(representation, space))
                    elif space=='tb101':
                        exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}"), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten(), hw_idx)))'.format(representation, args.task, args.task))
                    else:
                        exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}"), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten(), hw_idx)))'.format(representation, space, space))
                    if space=='tb101':
                        if device == None: accs.append(embedding_gen.get_valacc(i, task=args.task));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                    elif space not in ['nb101', 'nb201', 'nb301']:
                        if device == None: accs.append(embedding_gen.get_valacc(i, space=space));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                    else:
                        if device == None: accs.append(embedding_gen.get_valacc(i));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
            # representations = torch.stack([torch.FloatTensor(nxx) for nxx in representations])
        else: # adj_gin, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate --> GIN_Model
            assert representation in ["adj_gin", "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"], "Representation Not Supported!"
            if args.representation == "adj_gin":
                for i in tqdm(sample_indexes):
                    if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                        adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=space).values()
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        op_mat_norm = torch.Tensor(np.array(op_mat_norm)).argmax(dim=1)
                        op_mat_red = torch.Tensor(np.array(op_mat_red)).argmax(dim=1)
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat_norm)).flatten()
                        if device == None: accs.append(embedding_gen.get_valacc(i, space=space));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red), torch.Tensor(norm_w_d), torch.Tensor(hw_idx)))
                    else:
                        adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                        op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat)).flatten()
                        if space == 'tb101':
                            if device == None: accs.append(embedding_gen.get_valacc(i, task=args.task));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        else:
                            if device == None: accs.append(embedding_gen.get_valacc(i));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat), torch.Tensor(norm_w_d), torch.Tensor(hw_idx)))
            else: # "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate"
                for i in tqdm(sample_indexes):
                    if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                        adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=space).values()
                        method_name = 'get_{}'.format(args.representation.split("_")[-1])
                        method_to_call = getattr(embedding_gen, method_name)
                        zcp_ = method_to_call(i, space=space)
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        op_mat_norm = torch.Tensor(np.array(op_mat_norm)).argmax(dim=1)
                        op_mat_red = torch.Tensor(np.array(op_mat_red)).argmax(dim=1)
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat_norm)).flatten()
                        if device == None: accs.append(embedding_gen.get_valacc(i, space=space));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red), torch.Tensor(zcp_), torch.Tensor(norm_w_d), torch.Tensor(hw_idx)))
                    else:
                        adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                        method_name = 'get_{}'.format(args.representation.split("_")[-1])
                        method_to_call = getattr(embedding_gen, method_name)
                        if space == 'tb101':
                            zcp_ = method_to_call(i, task=args.task)
                        else:
                            zcp_ = method_to_call(i)
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat)).flatten()
                        if space == 'tb101':
                            if device == None: accs.append(embedding_gen.get_valacc(i, task=args.task));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        else:
                            if device == None: accs.append(embedding_gen.get_valacc(i));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        representations.append((torch.Tensor(adj_mat), torch.LongTensor(op_mat), torch.Tensor(zcp_), torch.Tensor(norm_w_d), torch.Tensor(hw_idx)))
    dataset = CustomDataset(representations, accs)
    if batch_specified != None:
        dataloader = DataLoader(dataset, batch_size=batch_specified, shuffle=True if mode=='train' else False)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size if mode=='train' else args.test_batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes
    

def get_input_dimensions(train_dataloader, representation): 
    if representation in ["adj_gin", "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
        return next(iter(train_dataloader))[0][1].shape[1]
    elif representation in ["adj_mlp", "zcp", "arch2vec", "cate"] or representation.__contains__("adj_mlp"):
        return next(iter(train_dataloader))[0].shape[1]
    else:
        return None

def create_gin_model(input_dim, num_zcps, dual_gcn, input_zcp, args):
    none_op_ind = 130  # placeholder
    return GIN_Model(
        device=args.source_devices,
        cpu_gpu_device=args.cpu_gpu_device,
        dual_gcn=dual_gcn,
        num_zcps=num_zcps,
        hwemb_to_mlp=args.hwemb_to_mlp,
        vertices=input_dim,
        none_op_ind=none_op_ind,
        op_embedding_dim=48,
        node_embedding_dim=48,
        zcp_embedding_dim=48,
        hid_dim=96,
        forward_gcn_out_dims=args.forward_gcn_out_dims,
        op_fp_gcn_out_dims=args.op_fp_gcn_out_dims,
        mlp_dims=[200, 200, 200],
        dropout=0.0,
        replace_bgcn_mlp_dims=args.replace_bgcn_mlp_dims,
        input_zcp=input_zcp,
        zcp_embedder_dims=[128, 128],
        updateopemb_dims = [128],
        ensemble_fuse_method=args.ensemble_fuse_method,
        gtype=args.gnn_type
    )

def get_distinct_arch2vecs_kmeans(arch2vecs_, n):
    vectors = list(arch2vecs_.values())
    print("Conducting KMeans...")
    start = time.time()
    kmeans = KMeans(n_clusters=n).fit(vectors)
    # Choose the vectors closest to the centroids as the representatives
    distinct_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(vectors - center, axis=1)
        distinct_indices.append(np.argmin(distances))
    # Return the corresponding keys from the arch2vecs_ dictionary
    keys = list(arch2vecs_.keys())
    print("KMeans took {} seconds".format(time.time() - start))
    return [keys[i] for i in distinct_indices]


def get_distinct_index(args, embedding_gen, space, sample_count, metric, device): # [random, params, arc2vec, cate, zcp, a2vcatezcp, accuracy/latency (oracle)]
    if metric == 'random':
        return random.sample(list(range(embedding_gen.get_numitems(space=space))), sample_count)
    elif metric == 'params':
        params_ = {i: embedding_gen.get_params(i) for i in list(range(embedding_gen.get_numitems(space=space)))}
        params_ = {k: v for k, v in sorted(params_.items(), key=lambda item: item[1])}
        buckets = np.array_split(list(params_.keys()), sample_count)
        return [random.choice(bucket) for bucket in buckets]
    elif metric in ['arch2vec', 'cate', 'zcp', 'a2vcatezcp']:
        metricdict_ = {i: getattr(embedding_gen, f"get_{metric}")(i) for i in list(range(embedding_gen.get_numitems(space=space)))}
        return get_distinct_arch2vecs_kmeans(metricdict_, sample_count)
    elif metric == 'accuracy':
        accs_ = {i: embedding_gen.get_valacc(i, space=space) for i in list(range(embedding_gen.get_numitems(space=space)))}
        accs_ = {k: v for k, v in sorted(accs_.items(), key=lambda item: item[1])}
        buckets = np.array_split(list(accs_.keys()), sample_count)
        return [random.choice(bucket) for bucket in buckets]
    elif metric == 'latency':
        latency_ = {i: embedding_gen.get_latency(i, space=space, device=device) for i in list(range(embedding_gen.get_numitems(space=space)))}
        latency_ = {k: v for k, v in sorted(latency_.items(), key=lambda item: item[1])}
        buckets = np.array_split(list(latency_.keys()), sample_count)
        return [random.choice(bucket) for bucket in buckets]
    else:
        raise NotImplementedError


representation = args.representation
sample_counts = sample_tests[args.space]
transfer_sample_counts = transfer_sample_tests[args.space]
samp_eff = {}
results_dict = {}
space = args.space
for tr_ in range(args.num_trials):
    for sample_count in sample_counts:
        # Create a function that chooses the best networks with a provided strategy, use only those networks
        train_samps = get_distinct_index(args, embedding_gen, args.space, sample_count, args.sampling_metric, args.metric_device)
        # Create a train data-loader with 'sample_count' samples of each 'source_device'
        train_dataloader, train_indexes = get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='train', indexes=train_samps, devices=args.source_devices, batch_specified=128)
        total_samples = embedding_gen.get_numitems(space) if space not in ['nb101', 'nb201', 'nb301', 'tb101'] else embedding_gen.get_numitems()
        # test_samples = list(set(range(total_samples - 1)) - set(train_indexes))
        test_samples = list(set(range(total_samples)))
        test_dataloader, test_indexes = get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='test', indexes=test_samples, devices=args.source_devices)
        test_dataloaderlowbs, test_indexes = get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='test', indexes=test_samples[:4], devices=args.source_devices)

        input_dim = get_input_dimensions(train_dataloader, representation)
        # import pdb; pdb.set_trace()

        if representation == "adj_gin":
            dual_gcn = args.space not in ["nb101", "nb201", "nb301", "tb101"]
            input_zcp = False
            model = create_gin_model(input_dim, 13, dual_gcn, input_zcp, args)

        elif representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            num_zcps = next(iter(train_dataloader))[0][-3].shape[1]
            dual_gcn = args.space not in ["nb101", "nb201", "nb301", "tb101"]
            input_zcp = True,
            model = create_gin_model(input_dim, num_zcps, dual_gcn, input_zcp, args)

        elif representation in ["adj_mlp", "zcp", "arch2vec", "cate"] or representation.__contains__("adj_mlp"):
            model = FullyConnectedNN(layer_sizes=[input_dim] + [200] * 3 + [1]).to(args.cpu_gpu_device)
        # Train on the train data-loader (pwl_train)
        model.to(args.cpu_gpu_device)
        params_optimize = list(model.parameters())
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(params_optimize, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
        pt_kdt_l5, pt_spr_l5 = [], []
        for epoch in range(args.epochs):
            if epoch > args.epochs - 5:
                start_time = time.time()
                model, num_test_items, mse_loss, spr, kdt = pwl_train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader, epoch)
                pt_kdt_l5.append(kdt)
                pt_spr_l5.append(spr)
                end_time = time.time()
                print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
            else:
                start_time = time.time()
                model, num_test_items, mse_loss, spr, kdt = pwl_train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloaderlowbs, epoch)
                end_time = time.time()
                print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')

        # Save the trained network state_dict
        trained_state_dict = model.state_dict()
        for tfdevice in args.target_devices:
            if tfdevice not in results_dict:
                results_dict[tfdevice] = {}
            for transfer_count in transfer_sample_counts:
                if sample_count not in results_dict[tfdevice]:
                    results_dict[tfdevice][sample_count] = {}
                if transfer_count not in results_dict[tfdevice][sample_count]:
                    results_dict[tfdevice][sample_count][transfer_count] = {'kdt': [], 'spr': []}

                # Find the index of the hardware which has the highest correlation
                # Create a transfer data-loader with 'transfer_count' samples of each 'target_device'
                transfer_samps = get_distinct_index(args, embedding_gen, args.space, transfer_count, args.sampling_metric, tfdevice)
                # import pdb; pdb.set_trace()
                test_samples = list(set(range(total_samples)))
                transfer_dataloader, transfer_indexes = get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='test', indexes=transfer_samps, devices=[tfdevice])
                transfer_test_dataloader, transfer_test_indexes = get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='test', indexes=test_samples, devices=[tfdevice])
                transfer_test_dataloaderlowbs, transfer_test_indexes = get_dataloader(args, embedding_gen, args.space, representation=args.representation, mode='test', indexes=test_samples[:4], devices=[tfdevice])

                
                # Reload the trained network state_dict
                model.load_state_dict(trained_state_dict)
                # Here, if transfer_hwemb is True, measure correlation between each of the source_devices and the tfdevice. 
                if args.transfer_hwemb:
                    s_t_corr = {}
                    for sdev in args.source_devices:
                        # Find source-target device correlation
                        # TODO : We should sum the sample count and transfer count to get the total number of samples for pre-training!!!!!
                        s_t_corr[sdev] = spearmanr([embedding_gen.get_latency(i, device=sdev, space=space) for i in transfer_samps], [embedding_gen.get_latency(i, device=tfdevice, space=space) for i in transfer_samps]).correlation
                    # Choose the source_device with the highest correlation
                    maxcorr_sdev = max(s_t_corr, key=s_t_corr.get)
                    # Find the embedding_gen.get_device_index for maxcorr_sdev
                    maxcorr_sdev_idx = embedding_gen.get_device_index(device=maxcorr_sdev)
                    # Now, replace the hwemb of the tfdevice with the hwemb of maxcorr_sdev using the get_device_index
                    model.hw_emb.weight.data[embedding_gen.get_device_index(device=tfdevice)] = model.hw_emb.weight.data[maxcorr_sdev_idx]

                # Train on the transfer data-loader (pwl_train)
                model.to(args.cpu_gpu_device)
                params_optimize = list(model.parameters())
                optimizer = torch.optim.AdamW(params_optimize, lr=args.transfer_lr, weight_decay=args.weight_decay)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.transfer_epochs, eta_min=args.eta_min)
                tt_kdt_l5, tt_spr_l5 = [], []
                for epoch in range(args.transfer_epochs):
                    if epoch > args.transfer_epochs - 5:
                        start_time = time.time()
                        model, num_test_items, mse_loss, spr, kdt = pwl_train(args, model, transfer_dataloader, criterion, optimizer, scheduler, transfer_test_dataloader, epoch)
                        tt_kdt_l5.append(kdt)
                        tt_spr_l5.append(spr)
                        end_time = time.time()
                        print(f'Epoch {epoch + 1}/{args.transfer_epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
                    else:
                        start_time = time.time()
                        model, num_test_items, mse_loss, spr, kdt = pwl_train(args, model, transfer_dataloader, criterion, optimizer, scheduler, transfer_test_dataloaderlowbs, epoch)
                        end_time = time.time()
                        print(f'Epoch {epoch + 1}/{args.transfer_epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
                results_dict[tfdevice][sample_count][transfer_count]['kdt'].append(sum(tt_kdt_l5)/len(tt_kdt_l5))
                results_dict[tfdevice][sample_count][transfer_count]['spr'].append(sum(tt_spr_l5)/len(tt_spr_l5))
                print("Device: {}, Sample Count: {}, Transfer Count: {}, Spearman: {}, Kendall: {}".format(tfdevice, sample_count, transfer_count, sum(tt_spr_l5)/len(tt_spr_l5), sum(tt_kdt_l5)/len(tt_kdt_l5)))


import random
import pickle
uid = random.randint(0, 1000000000)
# check if truecorrs exists
if not os.path.exists('truecorrs'):
    os.makedirs('truecorrs')
# Check that uid does not exist as a .pkl file. If it does, change it
while os.path.exists('truecorrs/{}/{}.pkl'.format(args.name_desc, uid)):
    uid = random.randint(0, 1000000000)

# Save results_dict as a pickle file in a folder called "truecorrs"
if not os.path.exists('truecorrs/{}'.format(args.name_desc)):
    os.makedirs('truecorrs/{}'.format(args.name_desc))
with open('truecorrs/{}/{}.pkl'.format(args.name_desc, uid), 'wb') as f:
    pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
# Now load it
# with open('truecorrs/{}/{}.pkl'.format(args.name_desc, uid), 'rb') as f:
#     results_dict = pickle.load(f)



if not os.path.exists('correlation_results/{}'.format(args.name_desc)):
    os.makedirs('correlation_results/{}'.format(args.name_desc))

filename = f'correlation_results/{args.name_desc}/{args.space}_samp_eff.csv'

header = "uid,name_desc,seed,space,source_devices,sampling_metric,metric_device,target_device,sample_sizes,transfer_sample_size,representation,gnn_type,num_trials,\
opfpgcn,fgcn,rbgcn,efm,fcd,lr,weight_decay,epochs,transfer_epochs,cpu_gpu_device,hwemb_to_mlp,transfer_hwemb,spr,kdt,spr_std,kdt_std"
# opfpgcn,fgcn,rbgcn,efm,fcd,lr,weight_decay,epochs,transfer_epochs,cpu_gpu_device," + "spr_%s," * len(args.target_devices) % (tuple(args.target_devices)) + "," + "kdt_%s," * len(args.target_devices) % (tuple(args.target_devices)) + "spr_std_%s," * len(args.target_devices) % (tuple(args.target_devices)) + "," + "kdt_std_%s," * len(args.target_devices) % (tuple(args.target_devices))
if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")
# results_dict[tfdevice][sample_count][transfer_count]['spr'].append(sum(tt_spr_l5)/len(tt_spr_l5))
source_device_str = "|".join(args.source_devices)
# sample_sizes_l = "|".join(args.sample_sizes)
with open(filename, 'a') as f:
    for sample_size in args.sample_sizes:
        for transfer_sample_size in args.transfer_sample_sizes:
            for target_device in args.target_devices:
                vals = [
                    str(uid),
                    str(args.name_desc),
                    str(args.seed),
                    str(args.space),
                    str(source_device_str),
                    str(args.sampling_metric),
                    str(args.metric_device),
                    str(target_device),
                    str(sample_size),
                    str(transfer_sample_size),
                    str(args.representation),
                    str(args.gnn_type),
                    str(args.num_trials),
                    str("_".join([str(zlx) for zlx in args.op_fp_gcn_out_dims])),
                    str("_".join([str(zlx) for zlx in args.forward_gcn_out_dims])),
                    str("_".join([str(zlx) for zlx in args.replace_bgcn_mlp_dims])),
                    str(args.ensemble_fuse_method),
                    str("_".join([str(zlx) for zlx in args.fb_conversion_dims])),
                    str(args.lr),
                    str(args.weight_decay),
                    str(args.epochs),
                    str(args.transfer_epochs),
                    str(args.cpu_gpu_device), # use results_dict[tfdevice][sample_count][transfer_count]['spr'].append(sum(tt_spr_l5)/len(tt_spr_l5))
                    str(args.hwemb_to_mlp),
                    str(args.transfer_hwemb),
                    str(np.mean(results_dict[target_device][sample_size][transfer_sample_size]['spr'])),
                    str(np.mean(results_dict[target_device][sample_size][transfer_sample_size]['kdt'])),
                    str(np.std(results_dict[target_device][sample_size][transfer_sample_size]['spr'])),
                    str(np.std(results_dict[target_device][sample_size][transfer_sample_size]['kdt']))
                    ]
                f.write("%s\n" % ','.join(vals))
