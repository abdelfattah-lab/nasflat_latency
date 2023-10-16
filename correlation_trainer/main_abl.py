import os
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
from models_abl_p2 import GIN_Model, FullyConnectedNN
import argparse, sys, time, random, os
import numpy as np
from pprint import pprint
from tqdm import tqdm
from utils import CustomDataset, get_tagates_sample_indices
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

# python -i new_main.py --space nb101 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 72 --batch_size 8
# python -i new_main.py --space nb201 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 40 --batch_size 8

parser = argparse.ArgumentParser()
####################################################### Search Space Choices #######################################################
parser.add_argument('--space', type=str, default='Amoeba')         # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--task', type=str, default='class_scene')     # all tb101 tasks supported
parser.add_argument('--representation', type=str, default='cate')  # adj_mlp, adj_gin, zcp (except nb301), cate, arch2vec, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate supported.
parser.add_argument('--test_tagates', action='store_true')         # Currently only supports testing on NB101 networks. Easy to extend.
parser.add_argument('--loss_type', type=str, default='pwl')        # mse, pwl supported
parser.add_argument('--gnn_type', type=str, default='dense')       # dense, gat, gat_mh supported
parser.add_argument('--back_dense', action="store_true")           # If True, backward flow will be DenseFlow
parser.add_argument('--num_trials', type=int, default=3)
parser.add_argument('--forward_gcn_out_dims', nargs='+', type=int, default=[128, 128, 128, 128, 128])
parser.add_argument('--backward_gcn_out_dims', nargs='+', type=int, default=[128, 128, 128, 128, 128])
parser.add_argument('--replace_bgcn_mlp_dims', nargs='+', type=int, default=[128, 128, 128, 128, 128])
parser.add_argument('--separate_op_fp', action="store_true")        # create separate fp for opemb           # <TEST NOW>
parser.add_argument('--no_residual', action="store_true")                                                    # <DEPRECATED, HARDCODED>
parser.add_argument('--back_mlp', action="store_true")              # True for best result                   # <DEPRECATED, HARDCODED>
parser.add_argument('--back_opemb', action="store_true")            # True for best result                   # <DEPRECATED, HARDCODED>
parser.add_argument('--randopupdate', action="store_true")          # False for best result                  # <DEPRECATED, HARDCODED>
parser.add_argument('--back_opemb_only', action="store_true")       # False for best result                  # <DEPRECATED, HARDCODED>
parser.add_argument('--opemb_direct', action="store_true")          # True for best result (5/8 improvement) # <DEPRECATED, HARDCODED>
parser.add_argument('--bmlp_ally', action="store_true")             # 27 False, 13 True (False best result)  # <DEPRECATED, HARDCODED>
parser.add_argument('--unroll_fgcn', action="store_true")           # False for best result                  # <DEPRECATED, HARDCODED>
parser.add_argument('--back_y_info', action="store_true")           # False for best result                  # <DEPRECATED, HARDCODED>
parser.add_argument('--ensemble_fuse_method', type=str, default='add')   # add, mlp (Need to test)           # <DEPRECATED, HARDCODED>
parser.add_argument('--detach_mode', type=str, default='default')   # default for best result, using none    # <DEPRECATED, HARDCODED>
parser.add_argument('--fb_conversion_dims', nargs='+', type=int, default=[128, 128])
parser.add_argument('--no_leakyrelu', action="store_true")
parser.add_argument('--no_unique_attention_projection', action="store_true")
parser.add_argument('--no_opattention', action="store_true")
parser.add_argument('--no_attention_rescale', action="store_true")
parser.add_argument('--timesteps', type=int, default=2)
###################################################### Other Hyper-Parameters ######################################################
parser.add_argument('--name_desc', type=str, default=None)
parser.add_argument('--sample_sizes', nargs='+', type=int, default=[72, 364, 728, 3645, 7280]) # Default NB101
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--test_size', type=int, default=None)
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
device = args.device
sample_tests = {}
sample_tests[args.space] = args.sample_sizes
args.residual = not args.no_residual
args.unique_attention_projection = not args.no_unique_attention_projection
args.opattention = not args.no_opattention
args.leakyrelu = not args.no_leakyrelu
args.attention_rescale = not args.no_attention_rescale

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
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
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
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
            archs_1 = [torch.stack(list((inputs[indx] for indx in ex_thresh_inds[1])))]
            archs_2 = [torch.stack(list((inputs[indx] for indx in ex_thresh_inds[0])))]
            X_input_1 = archs_1[0].to(device)
            s_1 = model(X_input_1).squeeze()
            X_input_2 = archs_2[0].to(device)
            s_2 = model(X_input_2).squeeze()
        elif args.representation in ["adj_gin"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0])))]
                X_adj_1, X_ops_1, norm_w_d_1 = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device)
                s_1 = model(x_ops_1=X_ops_1, x_adj_1=X_adj_1.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=norm_w_d_1).squeeze()
                X_adj_2, X_ops_2, norm_w_d_2 = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device)
                s_2 = model(x_ops_1=X_ops_2, x_adj_1=X_adj_2.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=norm_w_d_2).squeeze()
            else:
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
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, norm_w_d_1 = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device), archs_1[3].to(device), archs_1[4].to(device)
                s_1 = model(x_ops_1=X_ops_a_1, x_adj_1=X_adj_a_1.to(torch.long), x_ops_2=X_ops_b_1, x_adj_2=X_adj_b_1.to(torch.long), zcp=None, norm_w_d=norm_w_d_1).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, norm_w_d_2 = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device), archs_2[3].to(device), archs_2[4].to(device)
                s_2 = model(x_ops_1=X_ops_a_2, x_adj_1=X_adj_a_2.to(torch.long), x_ops_2=X_ops_b_2, x_adj_2=X_adj_b_2.to(torch.long), zcp=None, norm_w_d=norm_w_d_2).squeeze()
        elif args.representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0])))]
                X_adj_1, X_ops_1, zcp, norm_w_d_1 = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device), archs_1[3].to(device)
                s_1 = model(x_ops_1=X_ops_1, x_adj_1=X_adj_1.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=zcp, norm_w_d=norm_w_d_1).squeeze()
                X_adj_2, X_ops_2, zcp, norm_w_d_2 = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device), archs_2[3].to(device)
                s_2 = model(x_ops_1=X_ops_2, x_adj_1=X_adj_2.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=zcp, norm_w_d=norm_w_d_2).squeeze()
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
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, zcp, norm_w_d_1 = archs_1[0].to(device), archs_1[1].to(device), archs_1[2].to(device), archs_1[3].to(device), archs_1[4].to(device), archs_1[5].to(device)
                s_1 = model(x_ops_1 = X_ops_a_1, x_adj_1 = X_adj_a_1.to(torch.long), x_ops_2 = X_ops_b_1, x_adj_2 = X_adj_b_1.to(torch.long), zcp = zcp, norm_w_d=norm_w_d_1).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, zcp, norm_w_d_2 = archs_2[0].to(device), archs_2[1].to(device), archs_2[2].to(device), archs_2[3].to(device), archs_2[4].to(device), archs_2[5].to(device)
                s_2 = model(x_ops_1 = X_ops_a_2, x_adj_1 = X_adj_a_2.to(torch.long), x_ops_2 = X_ops_b_2, x_adj_2 = X_adj_b_2.to(torch.long), zcp = zcp, norm_w_d=norm_w_d_2).squeeze()
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
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
            pred_scores.append(model(reprs.to(device)).squeeze().detach().cpu().tolist())
        elif args.representation in ["adj_gin"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(device), x_adj_2=reprs[2].to(torch.long), zcp=None, norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu().tolist())
        elif args.representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if args.space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=reprs[2].to(device), norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(device), x_adj_2=reprs[2].to(torch.long), zcp=reprs[4].to(device), norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu().tolist())
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

def get_dataloader(args, embedding_gen, space, sample_count, representation, mode, train_indexes=None, test_size=None):
    representations = []
    accs = []
    if space == "nb101" and args.test_tagates:
        print("Sampling ONLY TAGATES NB101 networks for replication")
        if mode == "train":
            sample_indexes = nb101_train_tagates_sample_indices[:sample_count]
        else:
            sample_indexes = nb101_tagates_sample_indices
    else:
        if mode == "train":
            if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                sample_indexes = random.sample(range(embedding_gen.get_numitems(space)-1), sample_count)
            else:
                sample_indexes = random.sample(range(embedding_gen.get_numitems()-1), sample_count)
        else:
            if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                remaining_indexes = list(set(range(embedding_gen.get_numitems(space)-1)) - set(train_indexes))
            else:
                remaining_indexes = list(set(range(embedding_gen.get_numitems()-1)) - set(train_indexes))
            if test_size is not None:
                sample_indexes = random.sample(remaining_indexes, test_size)
            else:
                sample_indexes = remaining_indexes
    if representation.__contains__("gin") == False: # adj_mlp, zcp, arch2vec, cate --> FullyConnectedNN
        if representation == "adj_mlp": # adj_mlp --> FullyConnectedNN
            for i in tqdm(sample_indexes):
                if space not in ["nb101", "nb201", "nb301", "tb101"]:
                    adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=space).values()
                    norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                    norm_w_d = np.asarray(norm_w_d).flatten()
                    accs.append(embedding_gen.get_valacc(i, space=space))
                    adj_mat_norm = np.asarray(adj_mat_norm).flatten()
                    adj_mat_red = np.asarray(adj_mat_red).flatten()
                    op_mat_norm = torch.Tensor(np.asarray(op_mat_norm)).argmax(dim=1).numpy().flatten() # Careful here.
                    op_mat_red = torch.Tensor(np.asarray(op_mat_red)).argmax(dim=1).numpy().flatten() # Careful here.
                    representations.append(np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red, norm_w_d)).tolist())
                else:
                    adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i))
                    norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                    norm_w_d = np.asarray(norm_w_d).flatten()
                    adj_mat = np.asarray(adj_mat).flatten()
                    op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten() # Careful here.
                    representations.append(np.concatenate((adj_mat, op_mat, norm_w_d)).tolist())
        else:                           # zcp, arch2vec, cate --> FullyConnectedNN
            for i in tqdm(sample_indexes):
                if space in ['nb101', 'nb201', 'nb301']:
                    exec('representations.append(np.concatenate((embedding_gen.get_{}(i), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten())))'.format(representation, space))
                elif space=='tb101':
                    exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}"), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten())))'.format(representation, args.task, args.task))
                else:
                    exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}"), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten())))'.format(representation, space, space))
                if space=='tb101':
                    accs.append(embedding_gen.get_valacc(i, task=args.task))
                elif space not in ['nb101', 'nb201', 'nb301']:
                    accs.append(embedding_gen.get_valacc(i, space=space))
                else:
                    accs.append(embedding_gen.get_valacc(i))
        representations = torch.stack([torch.FloatTensor(nxx) for nxx in representations])
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
                    accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red), torch.Tensor(norm_w_d)))
                else:
                    adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                    op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                    norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                    norm_w_d = np.asarray(norm_w_d).flatten()
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i))
                    representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat), torch.Tensor(norm_w_d)))
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
                    accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red), torch.Tensor(zcp_), torch.Tensor(norm_w_d)))
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
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i))
                    representations.append((torch.Tensor(adj_mat), torch.LongTensor(op_mat), torch.Tensor(zcp_), torch.Tensor(norm_w_d)))

    dataset = CustomDataset(representations, accs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size if mode=='train' else args.test_batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes
    

representation = args.representation
sample_counts = sample_tests[args.space]
samp_eff = {}
across_trials = {sample_count: [] for sample_count in sample_counts}

for tr_ in range(args.num_trials):
    for sample_count in sample_counts:
        # if sample_count > 32:
        #     args.batch_size = int(sample_count//4)
        train_dataloader, train_indexes = get_dataloader(args, embedding_gen, args.space, sample_count, representation, mode='train')
        test_dataloader, test_indexes = get_dataloader(args, embedding_gen, args.space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=args.test_size)
        test_dataloader_lowbs, test_indexes = get_dataloader(args, embedding_gen, args.space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=80)

        if representation == "adj_gin":
            input_dim = next(iter(train_dataloader))[0][1].shape[1]
            none_op_ind = 130 # placeholder
            if args.space in ["nb101", "nb201", "nb301", "tb101"]:
                model = GIN_Model(device=args.device,
                                gtype = args.gnn_type,
                                back_dense=args.back_dense,
                                dual_gcn = False,
                                num_time_steps = args.timesteps,
                                vertices = input_dim,
                                none_op_ind = none_op_ind,
                                input_zcp = False,
                                gcn_out_dims = args.forward_gcn_out_dims,
                                backward_gcn_out_dims = args.backward_gcn_out_dims,
                                fb_conversion_dims = args.fb_conversion_dims,
                                bmlp_ally = args.bmlp_ally,
                                replace_bgcn_mlp_dims = args.replace_bgcn_mlp_dims,
                                residual=args.residual,
                                separate_op_fp = args.separate_op_fp,
                                unroll_fgcn = args.unroll_fgcn,
                                detach_mode = args.detach_mode,
                                back_mlp = args.back_mlp,
                                back_opemb = args.back_opemb,
                                back_y_info = args.back_y_info,
                                ensemble_fuse_method = args.ensemble_fuse_method,
                                randopupdate = args.randopupdate,
                                opemb_direct = args.opemb_direct,
                                unique_attention_projection=args.unique_attention_projection,
                                opattention=args.opattention,
                                back_opemb_only = args.back_opemb_only,
                                leakyrelu=args.leakyrelu,
                                attention_rescale=args.attention_rescale)
            else:
                model = GIN_Model(device=args.device,
                                gtype = args.gnn_type,
                                back_dense=args.back_dense,
                                dual_gcn = True,
                                num_time_steps = args.timesteps,
                                vertices = input_dim,
                                none_op_ind = none_op_ind,
                                unroll_fgcn = args.unroll_fgcn,
                                input_zcp = False,
                                separate_op_fp = args.separate_op_fp,
                                gcn_out_dims = args.forward_gcn_out_dims,
                                backward_gcn_out_dims = args.backward_gcn_out_dims,
                                bmlp_ally = args.bmlp_ally,
                                fb_conversion_dims = args.fb_conversion_dims,
                                replace_bgcn_mlp_dims = args.replace_bgcn_mlp_dims,
                                detach_mode = args.detach_mode,
                                residual=args.residual,
                                back_mlp = args.back_mlp,
                                opemb_direct = args.opemb_direct,
                                back_opemb = args.back_opemb,
                                back_y_info = args.back_y_info,
                                randopupdate = args.randopupdate,
                                ensemble_fuse_method = args.ensemble_fuse_method,
                                unique_attention_projection=args.unique_attention_projection,
                                opattention=args.opattention,
                                back_opemb_only = args.back_opemb_only,
                                leakyrelu=args.leakyrelu,
                                attention_rescale=args.attention_rescale)
        elif representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            input_dim = next(iter(train_dataloader))[0][1].shape[1]
            num_zcps = next(iter(train_dataloader))[0][-2].shape[1]
            none_op_ind = 130 # placeholder
            if args.space in ["nb101", "nb201", "nb301", "tb101"]:
                model = GIN_Model(device=args.device,
                                gtype = args.gnn_type,
                                back_dense=args.back_dense,
                                dual_gcn = False,
                                num_time_steps = args.timesteps,
                                num_zcps = num_zcps,
                                separate_op_fp = args.separate_op_fp,
                                unroll_fgcn = args.unroll_fgcn,
                                vertices = input_dim,
                                none_op_ind = none_op_ind,
                                detach_mode = args.detach_mode,
                                input_zcp = True,
                                gcn_out_dims = args.forward_gcn_out_dims,
                                bmlp_ally = args.bmlp_ally,
                                backward_gcn_out_dims = args.backward_gcn_out_dims,
                                fb_conversion_dims = args.fb_conversion_dims,
                                replace_bgcn_mlp_dims = args.replace_bgcn_mlp_dims,
                                residual=args.residual,
                                back_mlp = args.back_mlp,
                                opemb_direct = args.opemb_direct,
                                back_opemb = args.back_opemb,
                                randopupdate = args.randopupdate,
                                back_y_info = args.back_y_info,
                                ensemble_fuse_method = args.ensemble_fuse_method,
                                unique_attention_projection=args.unique_attention_projection,
                                opattention=args.opattention,
                                back_opemb_only = args.back_opemb_only,
                                leakyrelu=args.leakyrelu,
                                attention_rescale=args.attention_rescale)
            else:
                model = GIN_Model(device=args.device,
                                gtype = args.gnn_type,
                                back_dense=args.back_dense,
                                dual_gcn = True,
                                num_time_steps = args.timesteps,
                                num_zcps = num_zcps,
                                vertices = input_dim,
                                none_op_ind = none_op_ind,
                                separate_op_fp = args.separate_op_fp,
                                detach_mode = args.detach_mode,
                                input_zcp = True,
                                gcn_out_dims = args.forward_gcn_out_dims,
                                backward_gcn_out_dims = args.backward_gcn_out_dims,
                                bmlp_ally = args.bmlp_ally,
                                fb_conversion_dims = args.fb_conversion_dims,
                                residual=args.residual,
                                unroll_fgcn = args.unroll_fgcn,
                                back_mlp = args.back_mlp,
                                randopupdate = args.randopupdate,
                                opemb_direct = args.opemb_direct,
                                back_opemb = args.back_opemb,
                                back_y_info = args.back_y_info,
                                ensemble_fuse_method = args.ensemble_fuse_method,
                                unique_attention_projection=args.unique_attention_projection,
                                opattention=args.opattention,
                                back_opemb_only = args.back_opemb_only,
                                leakyrelu=args.leakyrelu,
                                attention_rescale=args.attention_rescale)
        elif representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
            representation_size = next(iter(train_dataloader))[0].shape[1]
            model = FullyConnectedNN(layer_sizes = [representation_size] + [200] * 3 + [1]).to(device)
        
        model.to(device)
        criterion = torch.nn.MSELoss()
        params_optimize = list(model.parameters())
        optimizer = torch.optim.AdamW(params_optimize, lr = args.lr, weight_decay = args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = args.eta_min)
        kdt_l5, spr_l5 = [], []
        for epoch in range(args.epochs):
            start_time = time.time()
            if args.loss_type == "mse":
                raise NotImplementedError
                # model, mse_loss, spr, kdt = train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader, epoch)
            elif args.loss_type == "pwl":
                if epoch > args.epochs - 5:
                    model, num_test_items, mse_loss, spr, kdt = pwl_train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader, epoch)
                else:
                    model, num_test_items, mse_loss, spr, kdt = pwl_train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader_lowbs, epoch)
            else:
                raise NotImplementedError
            # test_loss, num_test_items, test_spearmanr, test_kendalltau = test(args, model, test_dataloader, criterion)
            end_time = time.time()
            if epoch > args.epochs - 5:
                kdt_l5.append(kdt)
                spr_l5.append(spr)
                print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
        samp_eff[sample_count] = (sum(spr_l5)/len(spr_l5), sum(kdt_l5)/len(kdt_l5))
        print("Sample Count: {}, Spearman: {}, Kendall: {}".format(sample_count, sum(spr_l5)/len(spr_l5), sum(kdt_l5)/len(kdt_l5)))
        pprint(samp_eff)
        across_trials[sample_count].append(samp_eff[sample_count])

# print average across trials for each sample count
for sample_count in sample_counts:
    print("Average KDT: ", sum([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
    # Print variance of KDT across tests
    print("Variance KDT: ", np.var([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))]))
    # print SPR
    print("Average SPR: ", sum([across_trials[sample_count][i][0] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
    # Print variance of SPR across tests
    print("Variance SPR: ", np.var([across_trials[sample_count][i][0] for i in range(len(across_trials[sample_count]))]))

# sample_count = sample_counts[-1]
record_ = {}
for sample_count in sample_counts:
    avkdt = str(sum([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
    kdt_std = str(np.var([across_trials[sample_count][i][1] for i in range(len(across_trials[sample_count]))]))
    avspr = str(sum([across_trials[sample_count][i][0] for i in range(len(across_trials[sample_count]))])/len(across_trials[sample_count]))
    spr_std = str(np.var([across_trials[sample_count][i][0] for i in range(len(across_trials[sample_count]))]))
    record_[sample_count] = [avkdt, kdt_std, avspr, spr_std]

if not os.path.exists('correlation_results/{}'.format(args.name_desc)):
    os.makedirs('correlation_results/{}'.format(args.name_desc))

filename = f'correlation_results/{args.name_desc}/{args.space}_samp_eff.csv'
# 
# parser.add_argument('--forward_gcn_out_dims', nargs='+', type=int, default=[128, 128, 128, 128, 128])
# parser.add_argument('--backward_gcn_out_dims', nargs='+', type=int, default=[128, 128, 128, 128, 128])
# parser.add_argument('--replace_bgcn_mlp_dims', nargs='+', type=int, default=[128, 128, 128, 128, 128])
# parser.add_argument('--back_mlp', action="store_true")
# parser.add_argument('--fb_conversion_dims', nargs='+', type=int, default=[128, 128])
header = "name_desc,seed,batch_size,epochs,space,task,representation,timesteps,pwl_mse,test_tagates,gnn_type,back_dense,key,residual,leakyrelu,uap,opattn,attnresc,fgcn,bgcn,bmlp,bmlpdims,fbcd,back_y_info,back_opemb,ensemble_fuse_method,back_opemb_only,randopupdate,detach_mode,opemb_direct,unroll_fgcn,bmlp_ally,separate_op_fp,spr,kdt,spr_std,kdt_std"
if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")

with open(filename, 'a') as f:
    for key in samp_eff.keys():
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % 
                (
                    str(args.name_desc),
                    str(args.seed),
                    str(args.batch_size),
                    str(args.epochs),
                    str(args.space),
                    str(args.task),
                    str(args.representation),
                    str(args.timesteps),
                    str(args.loss_type),
                    str(args.test_tagates),
                    str(args.gnn_type),
                    str(args.back_dense),
                    str(key),
                    str(args.residual),
                    str(args.leakyrelu),
                    str(args.unique_attention_projection),
                    str(args.opattention),
                    str(args.attention_rescale),
                    str('_'.join([str(x) for x in args.forward_gcn_out_dims])),
                    str('_'.join([str(x) for x in args.backward_gcn_out_dims])),
                    str(args.back_mlp),
                    str('_'.join([str(x) for x in args.replace_bgcn_mlp_dims])),
                    str('_'.join([str(x) for x in args.fb_conversion_dims])),
                    str(args.back_y_info),
                    str(args.back_opemb),
                    str(args.ensemble_fuse_method),
                    str(args.back_opemb_only),
                    str(args.randopupdate),
                    str(args.detach_mode),
                    str(args.opemb_direct),
                    str(args.unroll_fgcn),
                    str(args.bmlp_ally),
                    str(args.separate_op_fp),
                    str(record_[key][2]),
                    str(record_[key][0]),
                    str(record_[key][3]),
                    str(record_[key][1])
                )
        )