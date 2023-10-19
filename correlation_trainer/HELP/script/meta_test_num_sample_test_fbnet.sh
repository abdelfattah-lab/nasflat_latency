

python main.py --gpu $1 \
		--search_space nasbench201 \
		--mode $2 \
		--num_samples $3 \
		--num_meta_train_sample $4 \
		--hw_embed_on True  \
		--hw_embed_dim $5 \
		--alpha_on True \
		--seed 3 \
		--load_path './data/fbnet/checkpoint/help_max_corr.pt' \
		--meta_train_devices $7 \
		--meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' \
		--meta_test_devices $8