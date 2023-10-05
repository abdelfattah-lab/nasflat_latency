#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python -i run.py --do_train --parallel --train_data data/nasbench201/train_data.pt --train_pair data/nasbench201/train_pair_k2_d200000_metric_params.pt  --valid_data data/nasbench201/test_data.pt --valid_pair data/nasbench201/test_pair_k2_d200000_metric_params.pt --dataset nasbench201 --search_space nasbench201 --n_vocab 7 --graph_d_model 32 --pair_d_model 32
