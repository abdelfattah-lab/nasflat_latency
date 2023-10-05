import os
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
import argparse, sys, time, random, os
import numpy as np
from pprint import pprint
from tqdm import tqdm
model_path = os.environ['PROJ_BPATH'] + "/correlation_trainer/"
sys.path.append(model_path)
from utils import CustomDataset, get_tagates_sample_indices
from new_models import GIN_Model, FullyConnectedNN
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy, csv

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')

parser = argparse.ArgumentParser()
####################################################### Search Space Choices #######################################################
parser.add_argument('--source_space', type=str, default=None)      # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--target_space', type=str, default='Amoeba')  # nb101, nb201, nb301, tb101, amoeba, darts, darts_fix-w-d, darts_lr-wd, enas, enas_fix-w-d, nasnet, pnas, pnas_fix-w-d supported
parser.add_argument('--task', type=str, default='class_scene')     # all tb101 tasks supported
parser.add_argument('--representation', type=str, default='cate')  # adj_mlp, adj_gin, zcp (except nb301), cate, arch2vec, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate supported.
parser.add_argument('--joint_repr', action='store_true')             # If True, uses the joint representation of the search space for CATE and Arch2Vec
parser.add_argument('--loss_type', type=str, default='pwl')        # mse, pwl supported
parser.add_argument('--gnn_type', type=str, default='dense')       # dense, gat, gat_mh supported
parser.add_argument('--back_dense', action="store_true")           # If True, backward flow will be DenseFlow
parser.add_argument('--periter_samps', type=int, default=10)       # Number of samples per search iteration
parser.add_argument('--samp_lim', type=int, default=2000)          # Number of samples per search iteration
parser.add_argument('--source_samps', type=int, default=512)
parser.add_argument('--num_trials', type=int, default=3)
parser.add_argument('--no_modify_emb_pretransfer', action='store_true')
parser.add_argument('--analysis_mode', action='store_true')
###################################################### Other Hyper-Parameters ######################################################
parser.add_argument('--name_desc', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--timesteps', type=int, default=2)
parser.add_argument('--test_size', type=int, default=None)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--transf_ep', type=int, default=5)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--lr_gamma', type=float, default=0.6)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--transfer_lr', type=float, default=1e-4)
parser.add_argument('--eta_min', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--id', type=int, default=0)
####################################################################################################################################
args = parser.parse_args()
device = args.device

args.modify_emb_pretransfer = not args.no_modify_emb_pretransfer

if args.source_space is None:
    args.transfer_lr = args.lr
    args.transf_ep = args.epochs

assert args.name_desc is not None, "Please provide a name description for the experiment."

sys.path.append("..")

if args.source_space is None:
    if args.target_space in ['Amoeba', 'DARTS', 'DARTS_fix-w-d', 'DARTS_lr-wd', 'ENAS', 'ENAS_fix-w-d', 'NASNet', 'PNAS', 'PNAS_fix-w-d']:
        from nas_embedding_suite.nds_ss import NDS as EmbGenClass
    elif args.target_space in ['nb101', 'nb201', 'nb301']:
        exec("from nas_embedding_suite.nb{}_ss import NASBench{} as EmbGenClass".format(args.target_space[-3:], args.target_space[-3:]))
    elif args.target_space in ['tb101']:
        from nas_embedding_suite.tb101_micro_ss import TransNASBench101Micro as EmbGenClass
    else:
        raise NotImplementedError
    embedding_gen = EmbGenClass(normalize_zcp=True, log_synflow=True)
else:
    print("Pre-training predictor with {} samples of source space: {}".format(args.source_samps, args.source_space))
    from nas_embedding_suite.all_ss import AllSS as EmbGenClass
    embedding_gen = EmbGenClass()

# exit()

if args.analysis_mode:
    for repr_ in ["adj_gin", "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
        for tf_ in [True, False]:
            dataset = args.target_space
            assert args.target_space == 'nb101', "Only nb101 is supported for analysis mode."
            ftype = repr_
            expname = 'exp7' if tf_ else 'exp6'
            fpath = f'./search_results/{expname}/sample_idxs/{dataset}_{ftype}_samples.csv'
            slist = {}
            with open(fpath, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    index = int(row[0])  # Assumes that index is the first column and is an integer.
                    if index not in slist:
                        slist[index] = []
                    slist[index].append([float(value) for value in row[1:]])  # Assumes that values are floats.
            for index, values in slist.items():
                slist[index] = values[0]  # Keep the last row only
            accuracy_list = {}
            for trial, indices in slist.items():
                accuracies = []
                for idx in indices:
                    acc = embedding_gen.nb101.get_valacc(int(idx), normalized=False)
                    accuracies.append(acc)
                accuracy_list[trial] = accuracies
            first_exceed_idx = {}
            for trial, accuracies in accuracy_list.items():
                for idx, acc in enumerate(accuracies):
                    if acc > 0.9422:
                        first_exceed_idx[trial] = idx
                        break
            print(args.target_space, repr_, "\tTransfer" if tf_ is not None else "\tScratch")
            print(first_exceed_idx)
            print(sum(first_exceed_idx.values())/3)
            odx_ = [2, 4, 5, 6, 7, 8, 10, 34, 50, 140]
            for odx in odx_:
                macc = max(accuracy_list[0][:odx]) + max(accuracy_list[1][:odx]) + max(accuracy_list[2][:odx])
                print(odx, " :", macc/3)
            output_file = 'experiment_results_all.csv'
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                transfer = tf_
                if file.tell() == 0:
                    writer.writerow(['Target Space', 'Representation', 'Transfer', 'First Exceed Idx', 'Average Exceed Idx', 'Odx', 'Macc/3'])
                average_exceed_idx = sum(first_exceed_idx.values()) / 3
                for odx in odx_:
                    macc = (max(accuracy_list[0][:odx]) + max(accuracy_list[1][:odx]) + max(accuracy_list[2][:odx])) / 3
                    writer.writerow([args.target_space, repr_, tf_, first_exceed_idx, average_exceed_idx, odx, macc])
    exit()

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

def pwl_train(args, space, model, dataloader, criterion, optimizer, scheduler, test_dataloader, epoch, mode=None):
    model.training = True
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        if inputs[0].shape[0] == 1 and space in ['nb101', 'nb201', 'nb301', 'tb101']:
            continue
        elif inputs[0].shape[0] == 2 and space not in ['nb101', 'nb201', 'nb301', 'tb101']:
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
    if mode == "source_pretrain":
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
        pred_scores = [t for sublist in pred_scores for t in sublist]
        true_scores = [t for sublist in true_scores for t in sublist]
        num_test_items = len(pred_scores)
        return model, num_test_items, running_loss / len(dataloader), spearmanr(true_scores, pred_scores).correlation, kendalltau(true_scores, pred_scores).correlation
    return model, 0, running_loss / len(dataloader), 0, 0

def get_dataloader(args, embedding_gen, space, sample_count, representation, mode, train_indexes=None, test_size=None, fetch_fixed_index=None, explicit_batch_size=None):
    representations = []
    accs = []
    if args.source_space is not None:
        # here, we dont just need the numitems, we actually need the indexs mapped for each SS
        if args.source_space is None:
            idx_range = list(range(embedding_gen.get_numitems(space=space)))
        else:
            jli = embedding_gen.get_ss_idxrange(space=space)
            jli_min = min(jli)
            idx_range = [zlm - jli_min for zlm in jli]
        min_idx_range = min(idx_range)
        idx_ranges = [zlm - min_idx_range for zlm in idx_range]
        if fetch_fixed_index is None:
            if mode in ["train", "transfer"]:
                sample_indexes = random.sample(idx_ranges, sample_count)
            else: # if mode is train, and we want to test on the same space as train, we pass train_index. else, pass transfer_index to get_dataloader
                remaining_indexes = list(set(idx_ranges) - set(train_indexes))
                if test_size is not None:
                    sample_indexes = random.sample(remaining_indexes, test_size)
                else:
                    sample_indexes = remaining_indexes
        else:
            sample_indexes = fetch_fixed_index
    else:
        if fetch_fixed_index is None:
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
        else:
            sample_indexes = fetch_fixed_index
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
                exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}", joint={}), np.asarray(embedding_gen.get_norm_w_d(i, space={})).flatten()))'.format(representation, space, args.joint_repr, space))
                accs.append(embedding_gen.get_valacc(i, space=space))
        representations = torch.stack([torch.FloatTensor(nxx) for nxx in representations])
    else: # adj_gin, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate --> GIN_Model
        assert representation in ["adj_gin", "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"], "Representation Not Supported"
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
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat), torch.Tensor(norm_w_d)))
        else: # "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cat, "adj_gin_a2vcatezcp"e"
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
                    if space == 'tb101':
                        accs.append(embedding_gen.get_valacc(i, task=args.task))
                    else:
                        accs.append(embedding_gen.get_valacc(i, space=space))
                    representations.append((torch.Tensor(adj_mat), torch.LongTensor(op_mat), torch.Tensor(zcp_), torch.Tensor(norm_w_d)))

    dataset = CustomDataset(representations, accs)
    if explicit_batch_size is None:
        dataloader = DataLoader(dataset, batch_size=args.batch_size if mode in ['train', 'transfer'] else args.test_batch_size, shuffle=True if mode=='train' else False)
    else:
        dataloader = DataLoader(dataset, batch_size=explicit_batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes
    
def get_all_scores(model, dataloader, space):
    print("Getting ALL scores for next iteration sampling!")
    pred_scores = []
    for idx, (reprs, scores) in enumerate(tqdm(dataloader)):
        if args.representation in ["adj_mlp", "zcp", "arch2vec", "cate"]:
            pred_ = model(reprs.to(device)).squeeze().detach().cpu()
        elif args.representation in ["adj_gin"]:
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_ = model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu()
            else:
                pred_ = model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(device), x_adj_2=reprs[2].to(torch.long), zcp=None, norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu()
        elif args.representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if space in ['nb101', 'nb201', 'nb301', 'tb101']:
                pred_ = model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=reprs[2].to(device), norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu()
            else:
                pred_ = model(x_ops_1=reprs[1].to(device), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to(device), x_adj_2=reprs[2].to(torch.long), zcp=reprs[4].to(device), norm_w_d=reprs[-1].to(device)).squeeze().detach().cpu()
        else:
            raise NotImplementedError
        pred_scores.append(pred_.tolist())
    pred_scores = [t for sublist in pred_scores for t in sublist]
    return pred_scores

representation = args.representation

# Pre-train predictor and preserve state_dict
if args.source_space is not None:
    train_dataloader, train_indexes = get_dataloader(args, embedding_gen, args.source_space, args.source_samps, representation, mode='train')
    test_dataloader_source_smallset, test_indexes = get_dataloader(args, embedding_gen, args.source_space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=40)
    test_dataloader_source_full, test_indexes = get_dataloader(args, embedding_gen, args.source_space, sample_count=None, representation=representation, mode='test', train_indexes=train_indexes, test_size=512)
    if representation == "adj_gin":
        input_dim = next(iter(train_dataloader))[0][1].shape[1]
        none_op_ind = 130 # placeholder
        if args.source_space in ["nb101", "nb201", "nb301", "tb101"]:
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
        input_dim = next(iter(train_dataloader))[0][1].shape[1]
        num_zcps = next(iter(train_dataloader))[0][-2].shape[1]
        none_op_ind = 130 # placeholder
        if args.source_space in ["nb101", "nb201", "nb301", "tb101"]:
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
        elif args.loss_type == "pwl":
            if epoch > args.epochs - 5:
                model, num_test_items, mse_loss, spr, kdt = pwl_train(args, args.source_space, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader_source_full, epoch, mode='source_pretrain')
            else:
                model, num_test_items, mse_loss, spr, kdt = pwl_train(args, args.source_space, model, train_dataloader, criterion, optimizer, scheduler, test_dataloader_source_smallset, epoch, mode='source_pretrain')
        else:
            raise NotImplementedError
        end_time = time.time()
        if epoch > args.epochs - 5:
            kdt_l5.append(kdt)
            spr_l5.append(spr)
            print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
        else:
            print(f'Epoch {epoch + 1}/{args.epochs} | Train Loss: {mse_loss:.4f} | Epoch Time: {end_time - start_time:.2f}s | Spearman@{num_test_items}: {spr:.4f} | Kendall@{num_test_items}: {kdt:.4f}')
    preserved_state = copy.deepcopy(model.state_dict())
else:
    train_dataloader, _ = get_dataloader(args, embedding_gen, args.target_space, sample_count=None, fetch_fixed_index=list(range(10)), representation=args.representation, mode='transfer')
    if representation == "adj_gin":
        input_dim = next(iter(train_dataloader))[0][1].shape[1]
        none_op_ind = 130 # placeholder
        if args.target_space in ["nb101", "nb201", "nb301", "tb101"]:
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
        input_dim = next(iter(train_dataloader))[0][1].shape[1]
        num_zcps = next(iter(train_dataloader))[0][-2].shape[1]
        none_op_ind = 130 # placeholder
        if args.target_space in ["nb101", "nb201", "nb301", "tb101"]:
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
    preserved_state = copy.deepcopy(model.state_dict())

if args.source_space is None:
    jli = list(range(embedding_gen.get_numitems(space=args.target_space)))
else:
    jli = embedding_gen.get_ss_idxrange(space=args.target_space)
    jli_min = min(jli)
    jli = [zlm - jli_min for zlm in jli]
full_target_space, _ = get_dataloader(args, embedding_gen, args.target_space, sample_count=None, representation=representation, mode='test', train_indexes=None, fetch_fixed_index=jli)
mini_target_space, _ = get_dataloader(args, embedding_gen, args.target_space, sample_count=None, representation=representation, mode='test', train_indexes=None, fetch_fixed_index=random.sample(jli, 40))

# Initialize dictionaries to store statistics for each num_samps
best_accuracies = {}
median_accuracies = {}
mean_accuracies = {}

if args.modify_emb_pretransfer and args.source_space is not None:
    num_ops, space_idx = embedding_gen.ss_mapper_oprange[args.source_space]
    source_start_idx = sum([x[0] for _, x in sorted(embedding_gen.ss_mapper_oprange.items(), key=lambda y: y[1]) if x[1] < space_idx])
    source_end_idx = source_start_idx + num_ops
    num_ops, space_idx = embedding_gen.ss_mapper_oprange[args.target_space]
    transfer_start_idx = sum([x[0] for _, x in sorted(embedding_gen.ss_mapper_oprange.items(), key=lambda y: y[1]) if x[1] < space_idx])
    transfer_end_idx = transfer_start_idx + num_ops


for tr_ in range(args.num_trials):
    print("Trial number: {}".format(tr_))
    model.load_state_dict(preserved_state)
    if args.modify_emb_pretransfer and args.source_space is not None:
        modified_tensor = model.op_emb.weight.clone()
        modified_tensor[transfer_start_idx:transfer_end_idx] = torch.cat((preserved_state['op_emb.weight'][source_start_idx:source_end_idx].detach(),)*40, dim=0)[:(transfer_end_idx - transfer_start_idx)]
        model.op_emb.weight.data = modified_tensor
    sampled_indexes = []
    accuracy_sampled = []
    accuracy_predicted = []
    model.vertices = next(iter(mini_target_space))[0][1].shape[1]
    if args.target_space not in ["nb101", "nb201", "nb301", "tb101"]:
        model.dual_gcn = True
    if args.target_space in ["nb101", "nb201", "nb301", "tb101"]:
        model.dual_gcn = False
    sample_cts = [(2,2),(2,2),(2,2),(2,2),(4,4),(2,2),(2,2),(8,8),(8,8),(16,16),(16,16),(32,32)]
    # sample_cts = [(2,2),(2,2)]
    # 
    # for halv_rate, num_samps in enumerate(list(range(args.periter_samps, args.samp_lim, args.periter_samps))):
    for halv_rate, samp_tuple in enumerate(sample_cts):
        # predict score on entire search space
        pred_scores = get_all_scores(model, full_target_space, space=args.target_space)
        if args.source_space is None:
            jli = list(range(embedding_gen.get_numitems(space=args.target_space)))
        else:
            jli = embedding_gen.get_ss_idxrange(space=args.target_space)
            jli_min = min(jli)
            jli = [zlm - jli_min for zlm in jli]
        best_candidates = sorted(zip(jli, pred_scores), key=lambda p: p[1], reverse=True)
            # iterate best_candidates, and if index, doesnt match, add it to sampled_indexes.
        # Sample best candidates
        cand_tracker = 0
        start_len = len(sampled_indexes)
        while len(sampled_indexes) < start_len + samp_tuple[0]:
            cand = best_candidates[cand_tracker]
            if cand[0] not in sampled_indexes:
                sampled_indexes.append(cand[0])
                accuracy_predicted.append(cand[1])
                if args.target_space in ["nb101", "nb201", "nb301", "tb101"]:
                    if args.target_space == 'tb101':
                        accuracy_sampled.append(embedding_gen.get_valacc(cand[0], task=args.task))
                    else:
                        accuracy_sampled.append(embedding_gen.get_valacc(cand[0], space=args.target_space))
                else:
                    accuracy_sampled.append(embedding_gen.get_valacc(cand[0], space=args.target_space))
            cand_tracker += 1

        # Sample random candidates
        # rand_indx = random.sample(set(jli) - set(sampled_indexes), args.periter_samps//2)
        halv_candlen = max(512, len(jli)//(2**halv_rate))
        # take random elements from the top :halv_candlen candidates of best_candidates
        rand_indx = random.sample(set([x[0] for x in best_candidates[:halv_candlen]]) - set(sampled_indexes), min(samp_tuple[1], len(set([x[0] for x in best_candidates[:halv_candlen]]) - set(sampled_indexes))))
        # insert all elements in rand_indx to sampled_indexes
        sampled_indexes.extend(rand_indx)
        for idx in rand_indx:
            if args.target_space in ["nb101", "nb201", "nb301", "tb101"]:
                if args.target_space == 'tb101':
                    accuracy_sampled.append(embedding_gen.get_valacc(idx, task=args.task))
                else:
                    accuracy_sampled.append(embedding_gen.get_valacc(idx, space=args.target_space))
            else:
                accuracy_sampled.append(embedding_gen.get_valacc(idx, space=args.target_space))
            # accuracy_sampled.append(embedding_gen.get_valacc(idx, space=args.target_space))
        # Calculate statistics for the current num_samps
        best_accuracies.setdefault(len(sampled_indexes), []).append(max(accuracy_sampled))
        median_accuracies.setdefault(len(sampled_indexes), []).append(np.median(accuracy_sampled))
        mean_accuracies.setdefault(len(sampled_indexes), []).append(np.mean(accuracy_sampled))
        # create new dataloader with sampled_indexes
        train_dataloader, train_indexes = get_dataloader(args, embedding_gen, args.target_space, sample_count=None, fetch_fixed_index=sampled_indexes, representation=representation, mode='transfer')
        # full_target_space
        kdt_l5, spr_l5 = [], []
        start_time = time.time()
        if args.loss_type == "mse":
            raise NotImplementedError
        elif args.loss_type == "pwl":
            # Reset to pre-trained state
            model.load_state_dict(preserved_state)
            if args.modify_emb_pretransfer and args.source_space is not None:
                modified_tensor = model.op_emb.weight.clone()
                modified_tensor[transfer_start_idx:transfer_end_idx] = torch.cat((preserved_state['op_emb.weight'][source_start_idx:source_end_idx].detach(),)*40, dim=0)[:(transfer_end_idx - transfer_start_idx)]
                model.op_emb.weight.data = modified_tensor
            criterion = torch.nn.MSELoss()
            # Initialize optimizers and schedulers
            params_optimize = list(model.parameters())
            optimizer = torch.optim.AdamW(params_optimize, lr = args.transfer_lr, weight_decay = args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max = args.transf_ep, eta_min = args.eta_min)
            # Train on training dataloader
            for epoch in range(args.transf_ep):
                model, num_test_items, mse_loss, _, _ = pwl_train(args=args, space=args.target_space, model=model, dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, test_dataloader=full_target_space, epoch=epoch)
        else:
            raise NotImplementedError
        # Print statistics for the current num_samps (best_accuracy, median_accuracy, mean_accuracy)
        print(f'Num Samples: {len(sampled_indexes)} | Best Accuracy: {max(accuracy_sampled):.4f} | Median Accuracy: {np.median(accuracy_sampled):.4f} | Mean Accuracy: {np.mean(accuracy_sampled):.4f}')
        end_time = time.time()
    filename = f'search_results/{args.name_desc}/sample_idxs/{args.target_space}_{args.representation}_samples.csv'
    # make /sample_idxs folder if it doesnt exist
    if not os.path.exists(f'search_results/{args.name_desc}/sample_idxs/'):
        os.makedirs(f'search_results/{args.name_desc}/sample_idxs/')
    # Open the file in append mode
    with open(filename, 'a') as f:
        # write trial number and the sampled_indexes list
        sindx = ",".join([str(x) for x in sampled_indexes])
        s_acc_pred = ",".join([str(x) for x in accuracy_predicted])
        s_acc_sampled = ",".join([str(x) for x in accuracy_sampled])
        f.write(f"{tr_},{sindx}\n")
        f.write(f"{tr_},{s_acc_sampled}\n")
        f.write(f"{tr_},{s_acc_pred}\n")

# Calculate average and standard deviation for each statistic across all trials for each num_samps
av_best_acc = {k: np.mean(v) for k, v in best_accuracies.items()}
std_best_acc = {k: np.std(v) for k, v in best_accuracies.items()}

av_median_acc = {k: np.mean(v) for k, v in median_accuracies.items()}
std_median_acc = {k: np.std(v) for k, v in median_accuracies.items()}

av_mean_acc = {k: np.mean(v) for k, v in mean_accuracies.items()}
std_mean_acc = {k: np.std(v) for k, v in mean_accuracies.items()}

if not os.path.exists(f'search_results/'):
    os.makedirs(f'search_results/')

if not os.path.exists(f'search_results/{args.name_desc}/'):
    os.makedirs(f'search_results/{args.name_desc}/')

# if args.source_space is None:
#     filename = f'search_results/{args.name_desc}/{args.target_space}_search_eff_transfer.csv'
# else:
filename = f'search_results/{args.name_desc}/{args.target_space}_search_eff.csv'
header = "name_desc,seed,batch_size,epochs,source_space,target_space,task,representation,joint_repr,loss_type,gnn_type,back_dense,periter_sampes,samp_lim,source_samps,timesteps,transf_ep,lr,transfer_lr,num_samps,av_best_acc,av_median_acc,av_mean_acc,best_acc_std,median_acc_std,mean_acc_std"
if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(header + "\n")

with open(filename, 'a') as f:
    # for num_samps in list(range(args.periter_samps, args.samp_lim, args.periter_samps)):
    for num_samps in av_median_acc.keys():
        f.write(f"{args.name_desc},\
                  {args.seed},\
                  {args.batch_size},\
                  {args.epochs},\
                  {args.source_space},\
                  {args.target_space},\
                  {args.task},\
                  {args.representation},\
                  {args.joint_repr},\
                  {args.loss_type},\
                  {args.gnn_type},\
                  {args.back_dense},\
                  {args.periter_samps},\
                  {args.samp_lim},\
                  {args.source_samps},\
                  {args.timesteps},\
                  {args.transf_ep},\
                  {args.lr},\
                  {args.transfer_lr},\
                  {num_samps},\
                  {av_best_acc[num_samps]},\
                  {av_median_acc[num_samps]},\
                  {av_mean_acc[num_samps]},\
                  {std_best_acc[num_samps]},\
                  {std_median_acc[num_samps]},\
                  {std_mean_acc[num_samps]}\n")