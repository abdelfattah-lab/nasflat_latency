
                #!/bin/sh

                python main.py --gpu 0                         --search_space 'nasbench201'                         --mode 'meta-train'                         --num_meta_train_sample 900                         --num_inner_tasks 9                         --exp_name 'super_script_models_11111'                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_samples 10                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/nasbench201/super_script_models_11111/checkpoint/help_max_corr.pt'                         --meta_train_devices 1080ti_1,1080ti_32,1080ti_256,silver_4114,silver_4210r,samsung_a50,pixel3,essential_ph_1,samsung_s7                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices titan_rtx_256,gold_6226,fpga,pixel2,raspi4,eyeriss

                python main.py --gpu 0                         --search_space 'nasbench201'                         --mode 'meta-test'                         --num_samples 10                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_meta_train_sample 900                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/nasbench201/super_script_models_11111/checkpoint/help_max_corr.pt'                         --meta_train_devices 1080ti_1,1080ti_32,1080ti_256,silver_4114,silver_4210r,samsung_a50,pixel3,essential_ph_1,samsung_s7                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices titan_rtx_256,gold_6226,fpga,pixel2,raspi4,eyeriss >  /home/ya255/projects/few_shot_hardware/HELP/meta_results/meta_test_results_11111.txt