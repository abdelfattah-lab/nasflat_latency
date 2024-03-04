import os
import json
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# spearmr and kendalltu
from scipy.stats import spearmanr, kendalltau

def flatten_mixed_list(pred_scores):
    flattened = []
    for sublist in pred_scores:
        if isinstance(sublist, (list, tuple)):  # Check if the item is iterable
            flattened.extend(sublist)  # If it's iterable, extend the flattened list
        else:
            flattened.append(sublist)  # If it's not iterable, append it directly
    return flattened

def pwl_flan_train(model, dataloader, criterion, optimizer, scheduler, test_dataloader, epoch, total_epochs):
    model.training = True
    representation = 'adj_gin'
    space = 'nb201'
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        if representation in ["adj_mlp", "zcp", "arch2vec", "cate"]  or representation.__contains__("adj_mlp"):
            if inputs.shape[0] == 1 and space in ['nb101', 'fbnet', 'fbnet', 'nb201', 'nb301', 'tb101']:
                continue
            elif inputs.shape[0] <= 2 and space not in ['nb101', 'fbnet', 'nb201', 'nb301', 'tb101']:
                continue
        else:
            if inputs[0].shape[0] == 1 and space in ['nb101', 'fbnet', 'fbnet', 'nb201', 'nb301', 'tb101']:
                continue
            elif inputs[0].shape[0] <= 2 and space not in ['nb101', 'fbnet', 'nb201', 'nb301', 'tb101']:
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
        if representation in ["adj_mlp", "zcp", "arch2vec", "cate"]  or representation.__contains__("adj_mlp"):
            archs_1 = [torch.stack(list((inputs[indx] for indx in ex_thresh_inds[1])))]
            archs_2 = [torch.stack(list((inputs[indx] for indx in ex_thresh_inds[0])))]
            X_input_1 = archs_1[0].to(dtype=torch.float32, device="cuda:0")
            s_1 = model(X_input_1).squeeze()
            X_input_2 = archs_2[0].to(dtype=torch.float32, device="cuda:0")
            s_2 = model(X_input_2).squeeze()
        elif representation in ["adj_gin"]:
            if space in ['nb101', 'fbnet', 'fbnet', 'nb201', 'nb301', 'tb101']:
                archs_1 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[1]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[1])))]
                archs_2 = [torch.stack(list((inputs[0][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[1][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[2][indx] for indx in ex_thresh_inds[0]))),
                        torch.stack(list((inputs[3][indx] for indx in ex_thresh_inds[0])))]
                X_adj_1, X_ops_1, norm_w_d_1, hw_idx = archs_1[0].to("cuda:0"), archs_1[1].to("cuda:0"), archs_1[2].to("cuda:0"), archs_1[3].to("cuda:0")
                s_1 = model(x_ops_1=X_ops_1, x_adj_1=X_adj_1.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_2, X_ops_2, norm_w_d_2, hw_idx = archs_2[0].to("cuda:0"), archs_2[1].to("cuda:0"), archs_2[2].to("cuda:0"), archs_2[3].to("cuda:0")
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
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, norm_w_d_1, hw_idx = archs_1[0].to("cuda:0"), archs_1[1].to("cuda:0"), archs_1[2].to("cuda:0"), archs_1[3].to("cuda:0"), archs_1[4].to("cuda:0")
                s_1 = model(x_ops_1=X_ops_a_1, x_adj_1=X_adj_a_1.to(torch.long), x_ops_2=X_ops_b_1, x_adj_2=X_adj_b_1.to(torch.long), zcp=None, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, norm_w_d_2, hw_idx = archs_2[0].to("cuda:0"), archs_2[1].to("cuda:0"), archs_2[2].to("cuda:0"), archs_2[3].to("cuda:0"), archs_2[4].to("cuda:0")
                s_2 = model(x_ops_1=X_ops_a_2, x_adj_1=X_adj_a_2.to(torch.long), x_ops_2=X_ops_b_2, x_adj_2=X_adj_b_2.to(torch.long), zcp=None, norm_w_d=norm_w_d_2, hw_idx=hw_idx).squeeze()
        elif representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if space in ['nb101', 'fbnet', 'fbnet', 'nb201', 'nb301', 'tb101']:
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
                X_adj_1, X_ops_1, zcp, norm_w_d_1, hw_idx = archs_1[0].to("cuda:0"), archs_1[1].to("cuda:0"), archs_1[2].to("cuda:0"), archs_1[3].to("cuda:0"), archs_1[4].to("cuda:0")
                import pdb; pdb.set_trace()
                s_1 = model(x_ops_1=X_ops_1, x_adj_1=X_adj_1.to(torch.long), x_ops_2=None, x_adj_2=None, zcp=zcp, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_2, X_ops_2, zcp, norm_w_d_2, hw_idx = archs_2[0].to("cuda:0"), archs_2[1].to("cuda:0"), archs_2[2].to("cuda:0"), archs_2[3].to("cuda:0"), archs_2[4].to("cuda:0")
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
                X_adj_a_1, X_ops_a_1, X_adj_b_1, X_ops_b_1, zcp, norm_w_d_1 = archs_1[0].to("cuda:0"), archs_1[1].to("cuda:0"), archs_1[2].to("cuda:0"), archs_1[3].to("cuda:0"), archs_1[4].to("cuda:0"), archs_1[5].to("cuda:0"), archs_1[6].to("cuda:0")
                s_1 = model(x_ops_1 = X_ops_a_1, x_adj_1 = X_adj_a_1.to(torch.long), x_ops_2 = X_ops_b_1, x_adj_2 = X_adj_b_1.to(torch.long), zcp = zcp, norm_w_d=norm_w_d_1, hw_idx=hw_idx).squeeze()
                X_adj_a_2, X_ops_a_2, X_adj_b_2, X_ops_b_2, zcp, norm_w_d_2 = archs_2[0].to("cuda:0"), archs_2[1].to("cuda:0"), archs_2[2].to("cuda:0"), archs_2[3].to("cuda:0"), archs_2[4].to("cuda:0"), archs_2[5].to("cuda:0"), archs_2[6].to("cuda:0")
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
    repr_max = int(80/16)
    for repr_idx, (reprs, scores) in enumerate(tqdm(test_dataloader)):
        if epoch < total_epochs - 5 and repr_idx > repr_max:
            break
        if representation in ["adj_mlp", "zcp", "arch2vec", "cate"] or representation.__contains__("adj_mlp"):
            pred_scores.append(model(reprs.to("cuda:0", dtype=torch.float32)).squeeze().detach().cpu().tolist())
        elif representation in ["adj_gin"]:
            if space in ['nb101', 'fbnet', 'fbnet', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to("cuda:0"), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=None, norm_w_d=reprs[-2].to("cuda:0"), hw_idx=reprs[-1].to("cuda:0")).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to("cuda:0"), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to("cuda:0"), x_adj_2=reprs[2].to(torch.long), zcp=None, norm_w_d=reprs[-2].to("cuda:0"), hw_idx=reprs[-1].to("cuda:0")).squeeze().detach().cpu().tolist())
        elif representation in ["adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"]:
            if space in ['nb101', 'fbnet', 'fbnet', 'nb201', 'nb301', 'tb101']:
                pred_scores.append(model(x_ops_1=reprs[1].to("cuda:0"), x_adj_1=reprs[0].to(torch.long), x_ops_2=None, x_adj_2=None, zcp=reprs[2].to("cuda:0"), norm_w_d=reprs[-2].to("cuda:0"), hw_idx=reprs[-1].to("cuda:0")).squeeze().detach().cpu().tolist())
            else:
                pred_scores.append(model(x_ops_1=reprs[1].to("cuda:0"), x_adj_1=reprs[0].to(torch.long), x_ops_2=reprs[3].to("cuda:0"), x_adj_2=reprs[2].to(torch.long), zcp=reprs[4].to("cuda:0"), norm_w_d=reprs[-2].to("cuda:0"), hw_idx=reprs[-1].to("cuda:0")).squeeze().detach().cpu().tolist())
        else:
            raise NotImplementedError
        true_scores.append(scores.cpu().tolist())
    # pred_scores = [t for sublist in pred_scores for t in sublist]
    # true_scores = [t for sublist in true_scores for t in sublist]
    pred_scores = flatten_mixed_list(pred_scores)
    true_scores = flatten_mixed_list(true_scores)
    num_test_items = len(pred_scores)
    return model, num_test_items, running_loss / len(dataloader), spearmanr(true_scores, pred_scores).correlation, kendalltau(true_scores, pred_scores).correlation, pred_scores, true_scores

def get_help_dataloader(embedding_gen, space, mode, indexes, devices, batch_specified=None, representation="adj_gin"):
    representations = []
    batch_size = 16
    test_batch_size = 128
    accs = []
    sample_indexes = indexes
    for device in devices:
        if representation.__contains__("gin") == False: # adj_mlp, zcp, arch2vec, cate --> FullyConnectedNN
            if representation.__contains__("adj_mlp"): # adj_mlp --> FullyConnectedNN
                for i in tqdm(sample_indexes):
                    if space not in ["nb101", "fbnet", "nb201", "nb301", "tb101"]:
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
                        metric_val = np.asarray(eval('embedding_gen.get_{}(i, "{}")'.format(representation.replace("adj_mlp_", ""), "")))
                        representations.append(np.concatenate((adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red, norm_w_d, hw_idx, metric_val)))
                    else:
                        adj_mat, op_mat = embedding_gen.get_adj_op(i).values()
                        if space == 'tb101':
                            if device == None: accs.append(embedding_gen.get_valacc(i, task=""));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        else:
                            if device == None: accs.append(embedding_gen.get_valacc(i));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        adj_mat = np.asarray(adj_mat).flatten()
                        op_mat = torch.Tensor(np.asarray(op_mat)).argmax(dim=1).numpy().flatten() # Careful here.
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat)).flatten()
                        metric_val = np.asarray(eval('embedding_gen.get_{}(i, "{}")'.format(representation.replace("adj_mlp_", ""), "")))
                        representations.append(np.concatenate((adj_mat, op_mat, norm_w_d, hw_idx, metric_val)))
            else:                           # zcp, arch2vec, cate --> FullyConnectedNN
                for i in tqdm(sample_indexes):
                    hw_idx = np.asarray([embedding_gen.get_device_index(device),] * 8).flatten()
                    if space in ['nb101', 'fbnet', 'nb201', 'nb301']:
                        exec('representations.append(np.concatenate((embedding_gen.get_{}(i), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten(), hw_idx)))'.format(representation, space))
                    elif space=='tb101':
                        exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}"), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten(), hw_idx)))'.format(representation, "", ""))
                    else:
                        exec('representations.append(np.concatenate((embedding_gen.get_{}(i, "{}"), np.asarray(embedding_gen.get_norm_w_d(i, space="{}")).flatten(), hw_idx)))'.format(representation, space, space))
                    if space=='tb101':
                        if device == None: accs.append(embedding_gen.get_valacc(i, task=""));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                    elif space not in ['nb101', 'fbnet', 'nb201', 'nb301']:
                        if device == None: accs.append(embedding_gen.get_valacc(i, space=space));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                    else:
                        if device == None: accs.append(embedding_gen.get_valacc(i));
                        else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
            # representations = torch.stack([torch.FloatTensor(nxx) for nxx in representations])
        else: # adj_gin, adj_gin_zcp, adj_gin_arch2vec, adj_gin_cate --> GIN_Model
            assert representation in ["adj_gin", "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate", "adj_gin_a2vcatezcp"], "Representation Not Supported!"
            if representation == "adj_gin":
                for i in tqdm(sample_indexes):
                    if space not in ['nb101', 'fbnet', 'nb201', 'nb301', 'tb101']:
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
                            if device == None: accs.append(embedding_gen.get_valacc(i, task=""));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        else:
                            if device == None: accs.append(embedding_gen.get_valacc(i));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        representations.append((torch.Tensor(adj_mat), torch.Tensor(op_mat), torch.Tensor(norm_w_d), torch.Tensor(hw_idx)))
            else: # "adj_gin_zcp", "adj_gin_arch2vec", "adj_gin_cate"
                for i in tqdm(sample_indexes):
                    if space not in ['nb101', 'fbnet', 'nb201', 'nb301', 'tb101']:
                        adj_mat_norm, op_mat_norm, adj_mat_red, op_mat_red = embedding_gen.get_adj_op(i, space=space).values()
                        method_name = 'get_{}'.format(representation.split("_")[-1])
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
                        method_name = 'get_{}'.format(representation.split("_")[-1])
                        method_to_call = getattr(embedding_gen, method_name)
                        if space == 'tb101':
                            zcp_ = method_to_call(i, task="")
                        else:
                            zcp_ = method_to_call(i)
                        norm_w_d = embedding_gen.get_norm_w_d(i, space=space)
                        norm_w_d = np.asarray(norm_w_d).flatten()
                        op_mat = torch.Tensor(np.array(op_mat)).argmax(dim=1)
                        hw_idx = np.asarray([embedding_gen.get_device_index(device),] * len(op_mat)).flatten()
                        if space == 'tb101':
                            if device == None: accs.append(embedding_gen.get_valacc(i, task=""));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        else:
                            if device == None: accs.append(embedding_gen.get_valacc(i));
                            else: accs.append(embedding_gen.get_latency(i, device=device, space=space))
                        representations.append((torch.Tensor(adj_mat), torch.LongTensor(op_mat), torch.Tensor(zcp_), torch.Tensor(norm_w_d), torch.Tensor(hw_idx)))
    dataset = CustomDataset(representations, accs)
    if batch_specified != None:
        dataloader = DataLoader(dataset, batch_size=batch_specified, shuffle=True if mode=='train' else False)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size if mode=='train' else test_batch_size, shuffle=True if mode=='train' else False)
    return dataloader, sample_indexes
    
    
def get_tagates_sample_indices(args):
    import os
    BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
    BASE_PATH2 = "/home/ya255/projects/iclr_nas_embedding/correlation_trainer/"
    if args.space == 'nb101' and args.test_tagates:
        print("Explicit TAGATES comparision")
        # Check if nb101_train_tagates.npy and nb101_test_tagates.npy exist
        if not os.path.exists(BASE_PATH2 + '/tagates_replication/nb101_train_tagates.npy') or not os.path.exists(BASE_PATH2 + '/tagates_replication/nb101_test_tagates.npy'):
            from nb123.nas_bench_101.cell_101 import Cell101
            from nasbench import api as NB1API
            import pickle
            BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
            nb1_api = NB1API.NASBench(BASE_PATH + 'nasbench_only108_caterec.tfrecord')
            hash_to_idx = {v: idx for idx,v in enumerate(list(nb1_api.hash_iterator()))}
            with open(os.environ['PROJ_BPATH'] + "/" + "/correlation_trainer/tagates_replication/nb101_hash.txt", "rb") as fp:
                nb101_hash = pickle.load(fp)
            nb101_tagates_sample_indices = [hash_to_idx[hash_] for hash_ in nb101_hash]
            with open(os.environ['PROJ_BPATH'] + "/" + "/correlation_trainer/tagates_replication/nb101_hash_train.txt", "rb") as fp:
                nb101_train_hash = pickle.load(fp)
            nb101_train_tagates_sample_indices = [hash_to_idx[hash_] for hash_ in nb101_train_hash]
            np.save(BASE_PATH2 + '/tagates_replication/nb101_test_tagates.npy', nb101_tagates_sample_indices)
            np.save(BASE_PATH2 + '/tagates_replication/nb101_train_tagates.npy', nb101_train_tagates_sample_indices)
        else:
            print("Loading from npy")
            nb101_tagates_sample_indices = np.load(BASE_PATH2 + '/tagates_replication/nb101_test_tagates.npy')
            nb101_train_tagates_sample_indices = np.load(BASE_PATH2 + '/tagates_replication/nb101_train_tagates.npy')
        return nb101_train_tagates_sample_indices, nb101_tagates_sample_indices
    # if args.space == 'nb201' and args.test_tagates:
    #     print("Explicit TAGATES comparision")
    #     with open(os.environ['PROJ_BPATH'] + "/correlation_trainer/tagates_replication/nasbench201_zsall_train.pkl", "rb") as fp:
    #         train_data = pickle.load(fp)
    #     with open(os.environ['PROJ_BPATH'] + "/correlation_trainer/tagates_replication/nasbench201_zsall_valid.pkl", "rb") as fp:
    #         valid_data = pickle.load(fp)
        
    return [], []

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def is_empty(self):
        return self.cnt == 0

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

        
class CustomDataset(Dataset):
    def __init__(self, representations, accuracies):
        self.representations = representations
        self.accuracies = accuracies

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return self.representations[idx], self.accuracies[idx]

def _build_dataset(dataset, list):
    indices = np.random.permutation(list)
    X_adj = []
    X_ops = []
    for ind in indices:
        X_adj.append(torch.Tensor(dataset[str(ind)]['module_adjacency']))
        X_ops.append(torch.Tensor(dataset[str(ind)]['module_operations']))
    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return X_adj, X_ops, torch.Tensor(indices)


def load_json(f_name):
    """load nas-bench-101 dataset."""
    with open(f_name, 'r') as infile:
        dataset = json.loads(infile.read())
    return dataset

def save_checkpoint(model, optimizer, epoch, loss, dim, name, dropout, seed):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    dir_name = 'pretrained/dim-{}'.format(dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_path = os.path.join(dir_name, 'model-ae-{}.pt'.format(name))
    torch.save(checkpoint, f_path)


def save_checkpoint_vae(model, optimizer, text_signature, epoch, loss, dim, name, dropout, seed):
    """Saves a checkpoint."""
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    dir_name = 'pretrained/dim-{}'.format(dim)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    f_path = os.path.join(dir_name, 'model-%s-%s.pt' % (text_signature, name))
    torch.save(checkpoint, f_path)

def normalize_adj(A):
    D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
    D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
    DA = stacked_spmm(D_in, A)  # swap D_in and D_out
    DAD = stacked_spmm(DA, D_out)
    return DAD

def preprocessing(A, H, method, lbd=None):
    # FixMe: Attention multiplying D or lbd are not friendly with the crossentropy loss in GAE
    assert A.dim()==3

    if method == 0:
        return A, H

    if method==1:
        # Adding global node with padding
        A = F.pad(A, (0,1), 'constant', 1.0)
        A = F.pad(A, (0,0,0,1), 'constant', 0.0)
        H = F.pad(H, (0,1,0,1), 'constant', 0.0 )
        H[:, -1, -1] = 1.0

    if method==1:
        # using A^T instead of A
        # and also adding a global node
        A = A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A) # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        return DAD, H

    elif method == 2:
        assert lbd!=None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1-lbd)*A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A)  # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        def prep_reverse(DAD, H):
            AD = stacked_spmm(1.0/D_in, DAD)
            A =  stacked_spmm(AD, 1.0/D_out)
            return A.triu(1), H
        return DAD, H, prep_reverse

    elif method == 3:
        # bidirectional DAG
        assert lbd != None
        # using lambda*A + (1-lambda)*A^T
        A = lbd * A + (1 - lbd) * A.transpose(-1, -2)
        def prep_reverse(A, H):
            return 1.0/lbd*A.triu(1), H
        return A, H, prep_reverse

    elif method == 4:
        A = A + A.triu(1).transpose(-1, -2)
        def prep_reverse(A, H):
            return A.triu(1), H
        return A, H, prep_reverse


def get_accuracy(inputs, targets):
    N, I, _ = inputs[0].shape
    ops_recon, adj_recon = inputs[0], inputs[1]
    ops, adj = targets[0], targets[1]
    # post processing, assume non-symmetric
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    correct_ops = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float().mean().item()
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    threshold = 0.5 # hard threshold
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)

    ops_correct = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float()
    adj_correct = adj_recon_thre.eq(adj.type(torch.bool)).float()
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj

def get_train_acc(inputs, targets):
    acc_train = get_accuracy(inputs, targets)
    return 'training batch: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(*acc_train)

def get_train_NN_accuracy_str(inputs, targets, decoderNN, inds):
    acc_train = get_accuracy(inputs, targets)
    acc_val = get_NN_acc(decoderNN, targets, inds)
    return 'acc_ops:{0:.4f}({4:.4f}), mean_corr_adj:{1:.4f}({5:.4f}), mean_fal_pos_adj:{2:.4f}({6:.4f}), acc_adj:{3:.4f}({7:.4f}), top-{8} index acc {9:.4f}'.format(
        *acc_train, *acc_val)

def get_NN_acc(decoderNN, targets, inds):
    ops, adj = targets[0], targets[1]
    op_recon, adj_recon, op_recon_tk, adj_recon_tk, _, ind_tk_list = decoderNN.find_NN(ops, adj, inds)
    correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, acc = get_accuracy((op_recon, adj_recon), targets)
    pred_k = torch.tensor(ind_tk_list,dtype=torch.int)
    correct = pred_k.eq(torch.tensor(inds, dtype=torch.int).view(-1,1).expand_as(pred_k))
    topk_acc = correct.sum(dtype=torch.float) / len(inds)
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, pred_k.shape[1], topk_acc.item()

def get_val_acc(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon,_ = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

def get_val_acc_vae(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon,mu, logvar = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

def stacked_mm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def to_operations_darts(ops):
    transform_dict = {'c_k-2': 0, 'c_k-1': 1, 'none': 2, 'max_pool_3x3': 3, 'avg_pool_3x3': 4, 'skip_connect': 5,
                      'sep_conv_3x3': 6, 'sep_conv_5x5': 7, 'dil_conv_3x3': 8, 'dil_conv_5x5': 9, 'output': 10}

    ops_array = np.zeros([11, 11], dtype='int8')
    for row, op in enumerate(ops):
        ops_array[row, op] = 1
    return ops_array

def one_hot_darts(ops):
    transform_dict = {'c_k-2': 0, 'c_k-1': 1, 'none': 2, 'max_pool_3x3': 3, 'avg_pool_3x3': 4, 'skip_connect': 5,
                      'sep_conv_3x3': 6, 'sep_conv_5x5': 7, 'dil_conv_3x3': 8, 'dil_conv_5x5': 9, 'output': 10}

    ops_array = np.zeros([11, 11], dtype='int8')
    for row, op in enumerate(ops):
        ops_array[row, op] = 1
    return ops_array

def to_ops_darts(idx):
    transform_dict = {0:'c_k-2',1:'c_k-1',2:'none',3:'max_pool_3x3',4:'avg_pool_3x3',5:'skip_connect',6:'sep_conv_3x3',7:'sep_conv_5x5',8:'dil_conv_3x3',9:'dil_conv_5x5',10:'output'}
    ops = []
    for id in idx:
        ops.append(transform_dict[id.item()])
    return ops

def to_ops_nasbench201(idx):
    transform_dict = {0:'input',1:'nor_conv_1x1',2:'nor_conv_3x3',3:'avg_pool_3x3',4:'skip_connect',5:'none',6:'output'}
    ops = []
    for id in idx:
        ops.append(transform_dict[id.item()])
    return ops

def is_valid_nasbench201(adj, ops):
    if ops[0] != 'input' or ops[-1] != 'output':
        return False
    for i in range(2, len(ops)-1):
        if ops[i] not in ['nor_conv_1x1','nor_conv_3x3','avg_pool_3x3','skip_connect','none']:
            return False
    return True

def is_valid_darts(adj, ops):
    if ops[0] != 'c_k-2' or ops[1] != 'c_k-1' or ops[-1] != 'output':
        return False
    for i in range(2, len(ops)-1):
        if ops[i] not in ['none','max_pool_3x3','avg_pool_3x3','skip_connect','sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5']:
            return False
    adj = np.array(adj)
    #B0
    if sum(adj[:2,2]) == 0 or sum(adj[:2,3]) == 0:
        return False
    if sum(adj[4:,2]) > 0 or sum(adj[4:,3]) >0:
        return False
    #B1:
    if sum(adj[:4,4]) == 0 or sum(adj[:4,5]) == 0:
        return False
    if sum(adj[6:,4]) > 0 or sum(adj[6:,5]) > 0:
        return False
    #B2:
    if sum(adj[:6,6]) == 0 or sum(adj[:6,7]) == 0:
        return False
    if sum(adj[8:,6]) > 0 or sum(adj[8:,7]) > 0:
        return False
    #B3:
    if sum(adj[:8,8]) == 0 or sum(adj[:8,9]) == 0:
        return False
    return True






