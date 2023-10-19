import optuna
import os
import subprocess




def objective(trial):
    # num_train_updates = trial.suggest_int("num_train_updates", low=1, high=20)
    num_train_updates = 2
    meta_lr           = trial.suggest_float("meta_lr",          low=1e-5, high=0.01, step=0.0001) 
    inner_lr          = trial.suggest_float("inner_lr",         low=1e-5, high=0.01, step=0.0001) 
    num_eval_updates  = trial.suggest_int  ("num_eval_updates", low=1,    high=20) 
    z_scaling         = trial.suggest_float("z_scaling",        low=1e-3, high=0.1, step=0.001) 
    kl_scaling        = trial.suggest_float("kl_scaling",       low=1e-2, high=1, step=0.01) 
    mc_sampling       = trial.suggest_int  ("mc_sampling",      low=5,    high=16)

    try:
        command_to_execute = '''
        
        #!/bin/sh

        python main.py --gpu 0 \\
                --search_space nasbench201 \\
                --mode 'meta-train' \\
                --num_meta_train_sample 900 \\
                --num_inner_tasks 3 \\
                --exp_name 'smallset_adv_num_samples_10_seed_5' \\
                --seed 5 \\
                --num_samples 10 \\
                --meta_lr %s \\
                --inner_lr %s \\
                --num_train_updates %s \\
                --num_eval_updates %s \\
                --z_scaling %s \\
                --kl_scaling %s \\
                --mc_sampling %s \\
                --load_path '/home/ya255/projects/unified_nas_representation/HELP/results/nasbench201/smallset_adv_num_samples_10_seed_5/checkpoint/help_max_corr.pt' \\
                --meta_train_devices '2080ti_32,titanxp_32,1080ti_32' \\
                --meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \\
                --meta_test_devices 'eyeriss,pixel3,raspi4'

        python main.py --gpu 0 \\
                --search_space nasbench201 \\
                --mode 'meta-test' \\
                --num_samples 10 \\
                --seed 5 \\
                --num_meta_train_sample 900 \\
                --meta_lr %s \\
                --inner_lr %s \\
                --num_train_updates %s \\
                --num_eval_updates %s \\
                --z_scaling %s \\
                --kl_scaling %s \\
                --mc_sampling %s \\
                --load_path '/home/ya255/projects/unified_nas_representation/HELP/results/nasbench201/smallset_adv_num_samples_10_seed_5/checkpoint/help_max_corr.pt' \\
                --meta_train_devices '2080ti_32,titanxp_32,1080ti_32' \\
                --meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \\
                --meta_test_devices 'eyeriss,pixel3,raspi4' > /home/ya255/projects/unified_nas_representation/HELP/meta_test_results.txt
        ''' % (str(meta_lr), str(inner_lr), str(num_train_updates), str(num_eval_updates), str(z_scaling), str(kl_scaling), str(mc_sampling), str(meta_lr), str(inner_lr), str(num_train_updates), str(num_eval_updates), str(z_scaling), str(kl_scaling), str(mc_sampling))
        print(command_to_execute)
        text_file = open("script/temporary_file.sh", "w")
        text_file.write(command_to_execute)
        text_file.close()
        os.system('chmod u+rx script/temporary_file.sh')
        rc = subprocess.call("script/temporary_file.sh", shell=True)
        # exit(0)
        result_file = open('/home/ya255/projects/unified_nas_representation/HELP/meta_test_results.txt', 'r')
        results = result_file.read()
        result_file.close()
        accuracy = float(results.split("average] spearman ")[1].split(" MSE")[0])
        write_to_csv = str(meta_lr)+ ", " + str(inner_lr)+ ", " + str(num_train_updates)+ ", " + str(num_eval_updates)+ ", " + str(z_scaling)+ ", " + str(kl_scaling)+ ", " + str(mc_sampling)+ ", " + str(accuracy)
        record_file = open("optuna_record.csv", "a")
        record_file.write(write_to_csv + "\n")
        record_file.close()
        return accuracy
    except Exception as e:
        write_to_csv = str(meta_lr)+ ", " + str(inner_lr)+ ", " + str(num_train_updates)+ ", " + str(num_eval_updates)+ ", " + str(z_scaling)+ ", " + str(kl_scaling)+ ", " + str(mc_sampling)+ ", " + str(0)
        record_file = open("failed_nums.csv", "a")
        record_file.write(write_to_csv + "\n")
        record_file.close()
        print(e)
        return 0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)