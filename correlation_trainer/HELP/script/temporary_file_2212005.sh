
                #!/bin/sh

                python main.py --gpu 0                         --search_space 'fbnet'                         --mode 'meta-train'                         --num_meta_train_sample 4000                         --num_inner_tasks 4                         --exp_name 'super_script_models_2212005'                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_samples 20                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/fbnet/super_script_models_2212005/checkpoint/help_max_corr.pt'                         --meta_train_devices 2080ti_1,titanxp_1,silver_4210r,titan_rtx_32                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices eyeriss,pixel3,raspi4,samsung_a50

                python main.py --gpu 0                         --search_space 'fbnet'                         --mode 'meta-test'                         --num_samples 20                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_meta_train_sample 4000                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/fbnet/super_script_models_2212005/checkpoint/help_max_corr.pt'                         --meta_train_devices 2080ti_1,titanxp_1,silver_4210r,titan_rtx_32                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices eyeriss,pixel3,raspi4,samsung_a50 >  /home/ya255/projects/few_shot_hardware/HELP/meta_results/meta_test_results_2212005.txt