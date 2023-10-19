

# python main.py --gpu 0 \
# 		--search_space nasbench201 \
# 		--mode 'meta-train' \
# 		--num_meta_train_sample 900 \
# 		--num_inner_tasks 3 \
# 		--exp_name "smallset_adv_num_samples_10_seed_5" \
# 		--seed 5 \
# 		--num_samples 10 \
# 		--meta_lr 1e-3 \
# 		--inner_lr 5e-3 \
# 		--num_eval_updates 10 \
# 		--z_scaling 0.05 \
# 		--kl_scaling 0.3 \
# 		--mc_sampling 10 \
# 		--load_path "/home/ya255/projects/unified_nas_representation/HELP/results/nasbench201/smallset_adv_num_samples_$3_seed_$2/checkpoint/help_max_corr.pt" \
# 		--meta_train_devices '2080ti_32,titanxp_32,1080ti_32' \
# 		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
# 		--meta_test_devices 'eyeriss,pixel3,raspi4'

python main.py --gpu $1 \
		--search_space nasbench201 \
		--mode 'meta-test' \
		--num_samples $3 \
		--seed $2 \
		--meta_lr 1e-3 \
		--inner_lr 5e-3 \
		--num_eval_updates 10 \
		--z_scaling 0.05 \
		--kl_scaling 0.3 \
		--mc_sampling 10 \
		--num_meta_train_sample 900 \
		--load_path "/home/ya255/projects/unified_nas_representation/HELP/results/nasbench201/smallset_adv_num_samples_$3_seed_$2/checkpoint/help_max_corr.pt" \
		--meta_train_devices '2080ti_32,titanxp_32,1080ti_32' \
		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
		--meta_test_devices 'eyeriss,pixel3,raspi4'

# python main.py --gpu $1 \
# 		--search_space nasbench201 \
# 		--mode 'meta-train' \
# 		--num_meta_train_sample 900 \
# 		--mc_sampling 100 \
# 		--num_samples 128 \
# 		--alpha_on True \
# 		--second_order True \
# 		--hw_embed_on True \
# 		--hw_embed_dim 100 \
# 		--layer_size 100 \
# 		--z_on True \
# 		--determ False \
# 		--seed 3 \
# 		--num_inner_tasks 10 \
# 		--exp_name 'ACLT0.7_MCS_100_HDW_100_NS128' \
# 		--meta_train_devices 'titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32' \
# 		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
# 		--meta_test_devices 'eyeriss,pixel3,raspi4'


		
# python main.py --gpu $1 \
# 		--search_space nasbench201 \
# 		--mode 'meta-train' \
# 		--num_samples 10 \
# 		--seed 3 \
# 		--num_meta_train_sample 900 \
# 		--exp_name 'reproduce' \
# 		--num_inner_tasks 9 \
# 		--meta_train_devices '1080ti_1,1080ti_256,1080ti_32,2080ti_1,2080ti_32,titan_rtx_1,titan_rtx_32,titanx_1,titanx_32,titanxp_1,titanxp_32' \
# 		--meta_valid_devices 'titanx_1,titanx_32,gold_6240' \
# 		--meta_test_devices 'fpga,raspi4,eyeriss'


# python main.py --gpu $1 \
# 		--search_space nasbench201 \
# 		--mode 'meta-train' \
# 		--num_samples 10 \
# 		--num_meta_train_sample 900 \
# 		--hw_embed_on True \
# 		--hw_embed_dim 10 \
# 		--alpha_on True \
# 		--seed 3 \
# 		--save_path '/home/ya255/projects/unified_nas_representation/HELP/results/' \
# 		--exp_name 'low_corr_eagle_to_help' \
# 		--num_inner_tasks 2 \
# 		--meta_train_devices 'desktop_cpu_core_i7_7820x_fp32,embedded_tpu_edge_tpu_int8' \
# 		--meta_valid_devices 'desktop_cpu_core_i7_7820x_fp32,embedded_tpu_edge_tpu_int8' \
# 		--meta_test_devices  'raspi4,silver_4114,titan_rtx_256'


# 		# --num_inner_tasks 9 \
# 		# --exp_name 'type_A' \
# 		# --meta_train_devices '1080ti_1,2080ti_1,titan_rtx_32,titanx_32,titanxp_32,titanx_1,titanxp_1,1080ti_32,2080ti_32' \
# 		# --meta_valid_devices '1080ti_1,2080ti_1' \
# 		# --meta_test_devices  'desktop_cpu_core_i7_7820x_fp32,eyeriss,desktop_gpu_gtx_1080ti_fp32,embedded_tpu_edge_tpu_int8'

# 		# --exp_name 'type_A_invt' \
# 		# --num_inner_tasks 4 \
# 		# --meta_train_devices 'desktop_cpu_core_i7_7820x_fp32,eyeriss,desktop_gpu_gtx_1080ti_fp32,embedded_tpu_edge_tpu_int8'\
# 		# --meta_valid_devices 'desktop_cpu_core_i7_7820x_fp32,eyeriss'  \
# 		# --meta_test_devices  '1080ti_1,2080ti_1,titan_rtx_32,titanx_32,titanxp_32,titanx_1,titanxp_1,1080ti_32,2080ti_32' 


# 		# --exp_name 'type_B' \
# 		# --num_inner_tasks 3 \
# 		# --meta_train_devices 'titan_rtx_32,titan_rtx_1,silver_4114' \
# 		# --meta_valid_devices 'titan_rtx_32,titan_rtx_1,silver_4114' \
# 		# --meta_test_devices  'desktop_cpu_core_i7_7820x_fp32,eyeriss,desktop_gpu_gtx_1080ti_fp32,embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16'

# 		# --exp_name 'type_B_invt' \
# 		# --num_inner_tasks 5 \
# 		# --meta_train_devices 'desktop_cpu_core_i7_7820x_fp32,eyeriss,desktop_gpu_gtx_1080ti_fp32,embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16' \
# 		# --meta_valid_devices 'desktop_cpu_core_i7_7820x_fp32,eyeriss' \
# 		# --meta_test_devices  'titan_rtx_32,titan_rtx_1,silver_4114'




# 		# --exp_name 'type_C' \
# 		# --num_inner_tasks 10 \
# 		# --meta_train_devices '1080ti_1,2080ti_1,pixel2,titan_rtx_32,titanxp_32,titan_rtx_1,titanx_1,titanxp_1,1080ti_32,silver_4114' \
# 		# --meta_valid_devices '1080ti_1,2080ti_1'  \
# 		# --meta_test_devices  'embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16,eyeriss,desktop_cpu_core_i7_7820x_fp32,desktop_gpu_gtx_1080ti_fp32,mobile_dsp_snapdragon_855_hexagon_690_int8'

# 		# --exp_name 'type_C_invt' \
# 		# --num_inner_tasks 6 \
# 		# --meta_train_devices 'embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16,eyeriss,desktop_cpu_core_i7_7820x_fp32,desktop_gpu_gtx_1080ti_fp32,mobile_dsp_snapdragon_855_hexagon_690_int8' \
# 		# --meta_valid_devices 'embedded_tpu_edge_tpu_int8,embedded_gpu_jetson_nano_fp16'  \
# 		# --meta_test_devices  '1080ti_1,2080ti_1,pixel2,titan_rtx_32,titanxp_32,titan_rtx_1,titanx_1,titanxp_1,1080ti_32,silver_4114'