# python main.py --gpu $1 \
# 		--search_space nasbench201 \
# 		--mode 'meta-test' \
# 		--num_samples 128 \
# 		--mc_sampling 100 \
# 		--alpha_on True \
# 		--second_order True \
# 		--hw_embed_on True \
# 		--hw_embed_dim 100 \
# 		--layer_size 100 \
# 		--z_on True \
# 		--determ False \
# 		--seed 3 \
# 		--num_meta_train_sample 900 \
# 		--load_path '/home/ya255/projects/unified_nas_representation/HELP/results/nasbench201/ACLT0.7_MCS_100_HDW_100_NS128/checkpoint/help_max_corr.pt' \
# 		--meta_train_devices 'titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32' \
# 		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
# 		--meta_test_devices 'eyeriss,pixel3,raspi4,titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32'

python main.py --gpu $1 \
		--search_space nasbench201 \
		--mode 'meta-test' \
		--num_samples 10 \
		--seed 3 \
		--num_meta_train_sample 900 \
		--load_path '/home/ya255/projects/unified_nas_representation/HELP/results/nasbench201/alt_lt_07_woutliers_fullset_large_noidx/checkpoint/help_max_corr.pt' \
		--meta_train_devices 'titan_rtx_1,2080ti_1,titanxp_1,titanx_1,1080ti_1,titan_rtx_32,titanx_32,2080ti_32,titanxp_32,1080ti_32' \
		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
		--meta_test_devices 'eyeriss,pixel3,raspi4'