####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################

import os
import torch, random
from parser import get_parser
from help import HELP
import pickle

def main(args):
    set_seed(args)
    args = set_gpu(args)
    args = set_path(args)
    corr_results = {trial : [] for trial in range(args.num_trials)}
    for trial in range(args.num_trials):
        print(f'==> mode is [{args.mode}] ...')
        model = HELP(args)

        if args.mode == 'meta-train':
            model.meta_train()
            corr_results[trial].append(model.test_predictor())
    results_dir = '/home/ya255/projects/flan_hardware/correlation_trainer/correlation_results'
    new_dir = os.path.join(results_dir, args.name_desc)
    # make it if it doesnt exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # UID logic
    uid = random.randint(0, 1000000000)
    # check if truecorrs exists
    if not os.path.exists('truecorrs'):
        os.makedirs('truecorrs')
    # Check that uid does not exist as a .pkl file. If it does, change it
    while os.path.exists('truecorrs/{}/{}.pkl'.format(args.name_desc, uid)):
        uid = random.randint(0, 1000000000)

    # Save corr_results as a pickle file in a folder called "truecorrs"
    if not os.path.exists('truecorrs/{}'.format(args.name_desc)):
        os.makedirs('truecorrs/{}'.format(args.name_desc))
    with open('truecorrs/{}/{}.pkl'.format(args.name_desc, uid), 'wb') as f:
        pickle.dump(corr_results, f, pickle.HIGHEST_PROTOCOL)

    # Extracting required args and saving them along with calculated metrics
    header_args = ['seed', 'name_desc', 'num_trials', 'search_space', 'meta_train_devices', 
                   'meta_valid_devices', 'num_inner_tasks', 'num_meta_train_sample', 
                   'num_samples', 'num_query', 'meta_lr', 'num_episodes', 'num_train_updates', 
                   'num_eval_updates', 'alpha_on', 'inner_lr', 'second_order', 'hw_embed_on', 
                   'hw_embed_dim', 'layer_size', 'transfer_device', 'spr', 'spr_std', 'kdt', 'kdt_std']
    
    # Extract values based on header_args
    values = [getattr(args, attr) for attr in header_args if hasattr(args, attr)]
    # if attribute is a list, convert it to a string
    values = ['|'.join(map(str, value)) if isinstance(value, list) else value for value in values]

    # import pdb; pdb.set_trace()
    new_corr_results = {dev: {"spr": [], "kdt": []} for dev in corr_results[0][0].keys()}
    for trial in corr_results.keys():
        for dev in new_corr_results.keys():
            new_corr_results[dev]["spr"].append(corr_results[trial][0][dev][0]["spr"])
            new_corr_results[dev]["kdt"].append(corr_results[trial][0][dev][0]["kdt"])
    corr_results = new_corr_results
    old_values = values.copy()
    for device, results in corr_results.items():
        print(device)
        spr_values = results["spr"]
        kdt_values = results["kdt"]
        # Use numpy for calculating mean and std
        import numpy as np
        mean_spr = np.mean(spr_values)
        mean_spr_std = np.std(spr_values)
        mean_kdt = np.mean(kdt_values)
        mean_kdt_std = np.std(kdt_values)
        
        # Add the calculated values to the values list
        values.extend([str(device), mean_spr, mean_spr_std, mean_kdt, mean_kdt_std])
        # Write header if file doesnt exist
        if not os.path.exists(os.path.join(new_dir, f'help_samp_eff.csv')):
            with open(os.path.join(new_dir, f'help_samp_eff.csv'), 'a') as f:
                f.write(','.join(header_args) + '\n')
        # Write values to file
        with open(os.path.join(new_dir, f'help_samp_eff.csv'), 'a') as f:
            f.write(','.join(map(str, values)) + '\n')

        values = old_values.copy()

    # header_args = 'seed,name_desc,num_trials,search_space,meta_train_devices,meta_valid_devices,transfer_device,num_inner_tasks,num_meta_train_sample,num_samples,num_query,meta_lr,num_episodes,num_train_updates,num_eval_updates,alpha_on,inner_lr,second_order,hw_embed_on,hw_embed_dim,layer_size,spr,spr_std,kdt,kdt_std'
    # For each 'trnasfer_device' in the corr_results, calculate the spr, spr_std, kdt, kdt_std and save them along with the args described in the header above.


        # elif args.mode == 'meta-test':
        #     model.test_predictor()
            
        # elif args.mode == 'nas':
        #     model.nas()     

        
def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES']= '-1' if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args 

def set_path(args):
    args.data_path = os.path.join(
        args.main_path, 'data', args.search_space)
    args.save_path = os.path.join(
            args.save_path, args.search_space)        
    args.save_path = os.path.join(args.save_path, args.exp_name)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        if args.mode != 'nas':
            os.makedirs(os.path.join(args.save_path, 'checkpoint'))
    print(f'==> save path is [{args.save_path}] ...')   
    return args 

if __name__ == '__main__':
    main(get_parser())
