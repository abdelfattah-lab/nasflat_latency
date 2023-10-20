#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run.py --do_train --parallel --train_data data/fbnet/train_data.pt --train_pair data/fbnet/train_pair_k2_d2000000_metric_params.pt  --valid_data data/fbnet/test_data.pt --valid_pair data/fbnet/test_pair_k2_d2000000_metric_params.pt --dataset fbnet --search_space fbnet --n_vocab 9 --graph_d_model 32 --pair_d_model 32
