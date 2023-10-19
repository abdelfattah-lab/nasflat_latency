
python main.py --gpu $1 \
		--search_space fbnet \
		--mode 'meta-test' \
		--num_samples 10 \
		--num_episodes 4000 \
		--seed 3 \
		--num_meta_train_sample 4000 \
		--load_path '/home/ya255/projects/unified_nas_representation/HELP/results/fbnet/Corr_Lt_0.3/checkpoint/help_max_corr.pt' \
		--meta_train_devices '2080ti_1,titan_rtx_1,1080ti_1,titanxp_1,titanx_1' \
		--meta_valid_devices 'titanx_1,titanx_32,titanx_64,gold_6240' \
		--meta_test_devices 'fpga,raspi4,eyeriss'