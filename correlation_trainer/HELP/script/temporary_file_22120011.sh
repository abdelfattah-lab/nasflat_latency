
                #!/bin/sh

                python main.py --gpu 0                         --search_space 'nasbench201'                         --mode 'meta-train'                         --num_meta_train_sample 900                         --num_inner_tasks 8                         --exp_name 'super_script_models_22120011'                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_samples 6                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/nasbench201/super_script_models_22120011/checkpoint/help_max_corr.pt'                         --meta_train_devices 1080ti_1,2080ti_1,samsung_s7,gold_6226,silver_4114,pixel3,mobile_gpu_snapdragon_450_adreno_506_int8,mobile_cpu_snapdragon_855_kryo_485_int8                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16,mobile_gpu_snapdragon_675_adreno_612_int8

                python main.py --gpu 0                         --search_space 'nasbench201'                         --mode 'meta-test'                         --num_samples 6                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_meta_train_sample 900                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/nasbench201/super_script_models_22120011/checkpoint/help_max_corr.pt'                         --meta_train_devices 1080ti_1,2080ti_1,samsung_s7,gold_6226,silver_4114,pixel3,mobile_gpu_snapdragon_450_adreno_506_int8,mobile_cpu_snapdragon_855_kryo_485_int8                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16,mobile_gpu_snapdragon_675_adreno_612_int8 >  /home/ya255/projects/few_shot_hardware/HELP/meta_results/meta_test_results_22120011.txt