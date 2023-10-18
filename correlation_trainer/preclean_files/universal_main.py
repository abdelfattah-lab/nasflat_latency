import os
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
from new_models import GIN_Model, FullyConnectedNN
import argparse, sys, time, random, os
import numpy as np
from pprint import pprint
import copy
from tqdm import tqdm
from utils import CustomDataset, get_tagates_sample_indices
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

# python -i universal_main.py --space nb101 --transfer_space nb201 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_size 1024 --batch_size 128 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 5e-5 --test_size 500
# python -i universal_main.py --space nb201 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 40 --batch_size 8

parser = argparse.ArgumentParser()
####################################################### Search Space Choices #######################################################
parser.add_argument('--space', type=str, default='nb101')            # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--transfer_space', type=str, default='nb201')   # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--task', type=str, default='class_scene')       # all tb101 tasks supported
parser.add_argument('--representation', type=str, default='cate')    # adj_mlp, adj_gin, zcp (except nb301), cate, arch2vec, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate supported.
parser.add_argument('--joint_repr', action='store_true')             # If True, uses the joint representation of the search space for CATE and Arch2Vec
parser.add_argument('--test_tagates', action='store_true')           # Currently only supports testing on NB101 networks. Easy to extend.
parser.add_argument('--loss_type', type=str, default='pwl')          # mse, pwl supported
parser.add_argument('--back_dense', action="store_true")           # If True, backward flow will be DenseFlow
parser.add_argument('--gnn_type', type=str, default='dense')         # dense, gat, gat_mh, ensemble supported
parser.add_argument('--num_trials', type=int, default=3)
parser.add_argument('--no_modify_emb_pretransfer', action='store_true')
###################################################### Other Hyper-Parameters ######################################################
parser.add_argument('--name_desc', type=str, default=None)
parser.add_argument('--sample_size', type=int, default=512)
parser.add_argument('--transfer_sample_sizes', nargs='+', type=int, default=[4, 8, 16, 32]) # Default NB101
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--timesteps', type=int, default=2)
parser.add_argument('--test_size', type=int, default=None)
parser.add_argument('--sourcetest_size', type=int, default=250)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--transfer_epochs', type=int, default=30)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--lr_gamma', type=float, default=0.6)
parser.add_argument('--transfer_lr', type=float, default=1e-3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eta_min', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--id', type=int, default=0)
####################################################################################################################################
args = parser.parse_args()
device = args.device
transfer_sample_tests = {}
transfer_sample_tests[args.transfer_space] = args.transfer_sample_sizes
args.modify_emb_pretransfer = not args.no_modify_emb_pretransfer

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

def pwl_train(args, space, model, dataloader, criterion, optimizer, scheduler, test_dataloader, epoch):
    model.training = True
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
            if inputs.shape[0] == 1 and space in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
            elif inputs.shape[0] <= 2 and space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
        else:
            if inputs[0].shape[0] == 1 and space in ['nb101', 'nb201', 'nb301', 'tb101']:
                continue
            elif inputs[0].shape[0] <= 2 and space not in ['nb101', 'nb201', 'nb301', 'tb101']:
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
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
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
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
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
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(device), x_adj_2=reprs[2].to(torch.long), zcp=None, norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu().tolist())
        elif args.representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
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
from nas_embedding_suite.all_ss import AllSS as EmbGenClass
embedding_gen = EmbGenClass()

def get_dataloader(args, embedding_gen, space, sample_count, representation, mode, train_indexes=None, test_size=None):
    representations = []
    accs = []
    # here, we dont just need the numitems, we actually need the indexs mapped for each SS
    idx_range = embedding_gen.get_ss_idxrange(space)
    min_idx_range = min(idx_range)
    idx_ranges = [zlm - min_idx_range for zlm in idx_range]
    if mode in ["train", "transfer"]:
        sample_indexes = random.sample(idx_ranges, sample_count)
    else: # if mode is train, and we want to test on the same space as train, we pass train_index. else, pass transfer_index to get_dataloader
        remaining_indexes = list(set(idx_ranges) - set(train_indexes))
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
                    adj_mat, op_mat = embedding_gen.get_adj_op(i, bin_space=True).values()
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
                exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}", joint={}), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten()))'.format(representation, space, args.joint_repr, space))
                accs.append(embedding_gen.get_valacc(i, space=space))
        representations = torch.stack([torch.FloatTensor(nxx) for nxx in representations])
    else: # adj_gin, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate --> GIN_Model
        assert representation in ["adj_gin", "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"], "Representation Not Supported!"
        if representation == "adj_gin":
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
                    adj_mat, op_mat = embedding_gen.get_adj_op(i, space=space, bin_space=True).values()
                    op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                    norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                    norm_w_d = np.asarray(norm_w_d).flatten()
                    accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat), torch.Tensor(norm_w_d)))
        else: # "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate"
            for i in tqdm(sample_indexes):
                if space not in ['nb101', 'nb201', 'nb301', 'tb101']:
                    adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=space).values()
                    method_name = 'get_{}'.format(representation.split("_")[-1])
                    method_to_call = getattr(embedding_gen, method_name)
                    zcp_ = method_to_call(i, space=space, joint=args.joint_repr)
                    norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                    norm_w_d = np.asarray(norm_w_d).flatten()
                    op_mat_norm = torch.Tensor(np.array(op_mat_norm)).argmax(dim=1)
                    op_mat_red = torch.Tensor(np.array(op_mat_red)).argmax(dim=1)
                    accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat_norm), torch.Tensor(op_mat_norm), torch.Tensor(adj_mat_red), torch.Tensor(op_mat_red), torch.Tensor(zcp_), torch.Tensor(norm_w_d)))
                else:
                    adj_mat, op_mat = embedding_gen.get_adj_op(i, space=space, bin_space=True).values()
                    method_name = 'get_{}'.format(representation.split("_")[-1])
                    method_to_call = getattr(embedding_gen, method_name)
                    zcp_ = method_to_call(i, space=space, joint=args.joint_repr)
                    norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                    norm_w_d = np.asarray(norm_w_d).flatten()
                    op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                    accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat), torch.LongTensor(op_mat), torch.Tensor(zcp_), torch.Tensor(norm_w_d)))

    dataset = CustomDataset(representations, accs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size if mode in ['train', 'transfer'] else args.test_batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes
    

representation = args.representation
transfer_sample_counts = transfer_sample_tests[args.transfer_space]
samp_eff = {}
across_trials = {transfer_sample_count: [] for transfer_sample_count in transfer_sample_counts}

train_dataloader, train_indexes = get_dataloader(args, embedding_gen, args.space, args.sample_size, representation, mode='train')
test_dataloader_source_smallset, test_indexes = get_dataloader(args, embedding_gen, args.space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=80)
test_dataloader_source_full, test_indexes = get_dataloader(args, embedding_gen, args.space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=args.sourcetest_size)
if representation == "adj_gin":
    # input_dim = max(next(iter(train_dataloader))[0][1].shape[1], next(iter(transfer_dataloader))[0][1].shape[1])
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
                        input_zcp = False)
    else:
        model = GIN_Model(device=args.device,
                        gtype = args.gnn_type,
                        back_dense=args.back_dense,
                        dual_input = True,
                        dual_gcn = True,
                        num_time_steps = args.timesteps,
                        vertices = input_dim,
                        none_op_ind = none_op_ind,
                        input_zcp = False)
elif representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
    # input_dim = max(next(iter(train_dataloader))[0][1].shape[1], next(iter(transfer_dataloader))[0][1].shape[1])
    input_dim = next(iter(train_dataloader))[0][1].shape[1]
    num_zcps = next(iter(train_dataloader))[0][-2].shape[1]
    none_op_ind = 130
    if args.space in ["nb101", "nb201", "nb301", "tb101"]:
        model = GIN_Model(device=args.device,
                        gtype = args.gnn_type,
                        back_dense=args.back_dense,
                        dual_gcn = False,
                        num_time_steps = args.timesteps,
                        num_zcps = num_zcps,
                        vertices = input_dim,
                        none_op_ind = none_op_ind,
                        input_zcp = True)
    else:
        model = GIN_Model(device=args.device,
                        gtype = args.gnn_type,
                        back_dense=args.back_dense,
                        dual_input = True,
                        dual_gcn = True,
                        num_time_steps = args.timesteps,
                        num_zcps = num_zcps,
                        vertices = input_dim,
                        none_op_ind = none_op_ind,
                        input_zcp = True)
elif representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
    representation_size = next(iter(train_dataloader))[0].shape[1]
    model = FullyConnectedNN(layer_sizes = [representation_size] + [128] * 3 + [1]).to(device)

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
            model, num_test_items, mse_loss, spr, kdt = pwl_train(args, args.space, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader_source_full, epoch)
        else:
            model, num_test_items, mse_loss, spr, kdt = pwl_train(args, args.space, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader_source_smallset, epoch)
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
preserved_state = copy.deepcopy(model.state_dict())

if args.modify_emb_pretransfer:
    num_ops, space_idx = embedding_gen.ss_mapper_oprange[args.space]
    source_start_idx = sum([x[0] for _, x in sorted(embedding_gen.ss_mapper_oprange.items(), key=lambda y: y[1]) if x[1] < space_idx])
    source_end_idx = source_start_idx + num_ops
    num_ops, space_idx = embedding_gen.ss_mapper_oprange[args.transfer_space]
    transfer_start_idx = sum([x[0] for _, x in sorted(embedding_gen.ss_mapper_oprange.items(), key=lambda y: y[1]) if x[1] < space_idx])
    transfer_end_idx = transfer_start_idx + num_ops

for tr_ in range(args.num_trials):

    for transfer_sample_count in transfer_sample_counts:
        model.load_state_dict(preserved_state)
        if args.modify_emb_pretransfer:
            modified_tensor = model.op_emb.weight.clone()
            modified_tensor[transfer_start_idx:transfer_end_idx] = torch.cat((preserved_state['op_emb.weight'][source_start_idx:source_end_idx].detach(),)*40, dim=0)[:(transfer_end_idx - transfer_start_idx)]
            model.op_emb.weight.data = modified_tensor
        # if transfer_sample_count > 32:
        #     args.batch_size = int(transfer_sample_count//4)
        transfer_dataloader, transfer_indexes = get_dataloader(args, embedding_gen, args.transfer_space, sample_count=transfer_sample_count, representation=representation, mode='transfer')
        test_dataloader_target_smallset, test_indexes = get_dataloader(args, embedding_gen, args.transfer_space, sample_count=None, representation=representation, mode='test', train_indexes=transfer_indexes, test_size=80)
        test_dataloader_target_full, test_indexes = get_dataloader(args, embedding_gen, args.transfer_space, sample_count=None, representation=representation, mode='test', train_indexes=transfer_indexes, test_size=args.test_size)
        # import pdb; pdb.set_trace()
        ### Transfer
        model.vertices = next(iter(transfer_dataloader))[0][1].shape[1]
        if args.transfer_space not in ["nb101", "nb201", "nb301", "tb101"]:
            model.dual_gcn = True
        if args.transfer_space in ["nb101", "nb201", "nb301", "tb101"]:
            model.dual_gcn = False
        criterion = torch.nn.MSELoss()
        params_optimize = list(model.parameters())
        optimizer = torch.optim.AdamW(params_optimize, lr = args.transfer_lr, weight_decay = args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max = args.transfer_epochs, eta_min = args.eta_min)
        kdt_l5, spr_l5 = [], []
        for epoch in range(args.transfer_epochs):
            start_time = time.time()
            if args.loss_type == "mse":
                raise NotImplementedError
                # model, mse_loss, spr, kdt = train(args, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader, epoch)
            elif args.loss_type == "pwl":
                if epoch > args.transfer_epochs - 5:
                    model, num_test_items, mse_loss, spr, kdt = pwl_train(args, args.transfer_space, model, transfer_dataloader, criterion, optimizer, scheduler, test_dataloader_target_full, epoch)
                else:
                    model, num_test_items, mse_loss, spr, kdt = pwl_train(args, args.transfer_space, model, transfer_dataloader, criterion, optimizer, scheduler, test_dataloader_target_smallset, epoch)
            else:
                raise NotImplementedError
            # test_loss, num_test_items, test_spearmanr, test_kendalltau = test(args, model, test_dataloader, criterion)
            end_time = time.time()
            if epoch > args.transfer_epochs - 5:
                kdt_l5.append(kdt)
                spr_l5.append(spr)
                print(f'Epoch {epoch + 1}/{args.transfer_epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{args.transfer_epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')

        samp_eff[transfer_sample_count] = (sum(spr_l5)/len(spr_l5), sum(kdt_l5)/len(kdt_l5))
        print("Sample Count: {}, Spearman: {}, Kendall: {}".format(transfer_sample_count, sum(spr_l5)/len(spr_l5), sum(kdt_l5)/len(kdt_l5)))
        pprint(samp_eff)
        across_trials[transfer_sample_count].append(samp_eff[transfer_sample_count])

# print average across trials for each sample count
for transfer_sample_count in transfer_sample_counts:
    print("Average KDT: ", sum([across_trials[transfer_sample_count][i][1] for i in range(len(across_trials[transfer_sample_count]))])/len(across_trials[transfer_sample_count]))
    # Print variance of KDT across tests
    print("Variance KDT: ", np.var([across_trials[transfer_sample_count][i][1] for i in range(len(across_trials[transfer_sample_count]))]))
    # print SPR
    print("Average SPR: ", sum([across_trials[transfer_sample_count][i][0] for i in range(len(across_trials[transfer_sample_count]))])/len(across_trials[transfer_sample_count]))
    # Print variance of SPR across tests
    print("Variance SPR: ", np.var([across_trials[transfer_sample_count][i][0] for i in range(len(across_trials[transfer_sample_count]))]))

# sample_count = sample_counts[-1]
record_ = {}
for transfer_sample_count in transfer_sample_counts:
    avkdt = str(sum([across_trials[transfer_sample_count][i][1] for i in range(len(across_trials[transfer_sample_count]))])/len(across_trials[transfer_sample_count]))
    kdt_std = str(np.var([across_trials[transfer_sample_count][i][1] for i in range(len(across_trials[transfer_sample_count]))]))
    avspr = str(sum([across_trials[transfer_sample_count][i][0] for i in range(len(across_trials[transfer_sample_count]))])/len(across_trials[transfer_sample_count]))
    spr_std = str(np.var([across_trials[transfer_sample_count][i][0] for i in range(len(across_trials[transfer_sample_count]))]))
    record_[transfer_sample_count] = [avkdt, kdt_std, avspr, spr_std]

if not os.path.exists(f'correlation_results/{args.name_desc}'):
    os.makedirs(f'correlation_results/{args.name_desc}')

filename = f'correlation_results/{args.name_desc}/{args.space}_{args.transfer_space}_samp_eff.csv'
header = "name_desc,seed,batch_size,transfer_lr,transfer_epochs,membtf,epochs,space,transfer_space,joint_repr,representation,timesteps,pwl_mse,test_tagates,gnn_type,back_dense,key,spr,kdt,spr_std,kdt_std"
if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")

with open(filename, 'a') as f:
    for key in samp_eff.keys():
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % 
                (
                    str(args.name_desc),
                    str(args.seed),
                    str(args.batch_size),
                    str(args.transfer_lr),
                    str(args.transfer_epochs),
                    str(args.modify_emb_pretransfer),
                    str(args.epochs),
                    str(args.space),
                    str(args.transfer_space),
                    str(args.joint_repr),
                    str(args.representation),
                    str(args.timesteps),
                    str(args.loss_type),
                    str(args.test_tagates),
                    str(args.gnn_type),
                    str(args.back_dense),
                    str(key),
                    str(record_[key][2]),
                    str(record_[key][0]),
                    str(record_[key][3]),
                    str(record_[key][1])
                )
        )