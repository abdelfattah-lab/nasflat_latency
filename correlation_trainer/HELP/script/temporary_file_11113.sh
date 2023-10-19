
                #!/bin/sh

                python main.py --gpu 0                         --search_space 'nasbench201'                         --mode 'meta-train'                         --num_meta_train_sample 900                         --num_inner_tasks 5                         --exp_name 'super_script_models_11113'                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_samples 10                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/nasbench201/super_script_models_11113/checkpoint/help_max_corr.pt'                         --meta_train_devices embedded_gpu_jetson_nano_fp32,mobile_gpu_snapdragon_450_adreno_506_int8,mobile_cpu_snapdragon_450_cortex_a53_int8,mobile_gpu_snapdragon_675_adreno_612_int8,mobile_cpu_snapdragon_675_kryo_460_int8                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices embedded_tpu_edge_tpu_int8,desktop_cpu_core_i7_7820x_fp32,desktop_gpu_gtx_1080ti_fp32

                python main.py --gpu 0                         --search_space 'nasbench201'                         --mode 'meta-test'                         --num_samples 10                         --seed 4                         --z_on True                         --alpha_on True                         --second_order True                         --num_meta_train_sample 900                         --meta_lr 0.00101                         --inner_lr 0.00031                         --num_train_updates 2                         --num_eval_updates 16                         --z_scaling 0.015                         --z_on False                         --kl_scaling 0.96                         --mc_sampling 15                         --load_path '/home/ya255/projects/few_shot_hardware/HELP/results/nasbench201/super_script_models_11113/checkpoint/help_max_corr.pt'                         --meta_train_devices embedded_gpu_jetson_nano_fp32,mobile_gpu_snapdragon_450_adreno_506_int8,mobile_cpu_snapdragon_450_cortex_a53_int8,mobile_gpu_snapdragon_675_adreno_612_int8,mobile_cpu_snapdragon_675_kryo_460_int8                         --meta_valid_devices titanx_1,titanx_32,gold_6240                         --meta_test_devices embedded_tpu_edge_tpu_int8,desktop_cpu_core_i7_7820x_fp32,desktop_gpu_gtx_1080ti_fp32 >  /home/ya255/projects/few_shot_hardware/HELP/meta_results/meta_test_results_11113.txt