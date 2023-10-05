#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0


python run.py --do_train --parallel --train_data data/nds/nds_Amoeba_normal_train_data.pt \
    --train_pair data/nds/nds_Amoeba_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_Amoeba_normal_test_data.pt \
    --valid_pair data/nds/nds_Amoeba_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space Amoeba --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_PNAS_fix-w-d_normal_train_data.pt \
    --train_pair data/nds/nds_PNAS_fix-w-d_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_PNAS_fix-w-d_normal_test_data.pt \
    --valid_pair data/nds/nds_PNAS_fix-w-d_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space PNAS_fix-w-d --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_ENAS_fix-w-d_normal_train_data.pt \
    --train_pair data/nds/nds_ENAS_fix-w-d_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_ENAS_fix-w-d_normal_test_data.pt \
    --valid_pair data/nds/nds_ENAS_fix-w-d_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 8 --search_space ENAS_fix-w-d --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_NASNet_normal_train_data.pt \
    --train_pair data/nds/nds_NASNet_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_NASNet_normal_test_data.pt \
    --valid_pair data/nds/nds_NASNet_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 16 --search_space NASNet --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_DARTS_normal_train_data.pt \
    --train_pair data/nds/nds_DARTS_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_DARTS_normal_test_data.pt \
    --valid_pair data/nds/nds_DARTS_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space DARTS --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_ENAS_normal_train_data.pt \
    --train_pair data/nds/nds_ENAS_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_ENAS_normal_test_data.pt \
    --valid_pair data/nds/nds_ENAS_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 8 --search_space ENAS --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_PNAS_normal_train_data.pt \
    --train_pair data/nds/nds_PNAS_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_PNAS_normal_test_data.pt \
    --valid_pair data/nds/nds_PNAS_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space PNAS --type normal --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_DARTS_lr-wd_normal_train_data.pt \
    --train_pair data/nds/nds_DARTS_lr-wd_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_DARTS_lr-wd_normal_test_data.pt \
    --valid_pair data/nds/nds_DARTS_lr-wd_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space DARTS_lr-wd --type normal --graph_d_model 16 --pair_d_model 16

    
python run.py --do_train --parallel --train_data data/nds/nds_DARTS_fix-w-d_normal_train_data.pt \
    --train_pair data/nds/nds_DARTS_fix-w-d_normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_DARTS_fix-w-d_normal_test_data.pt \
    --valid_pair data/nds/nds_DARTS_fix-w-d_normal_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space DARTS_fix-w-d --type normal --graph_d_model 16 --pair_d_model 16




python run.py --do_train --parallel --train_data data/nds/nds_Amoeba_reduce_train_data.pt \
    --train_pair data/nds/nds_Amoeba_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_Amoeba_reduce_test_data.pt \
    --valid_pair data/nds/nds_Amoeba_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space Amoeba --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_PNAS_fix-w-d_reduce_train_data.pt \
    --train_pair data/nds/nds_PNAS_fix-w-d_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_PNAS_fix-w-d_reduce_test_data.pt \
    --valid_pair data/nds/nds_PNAS_fix-w-d_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space PNAS_fix-w-d --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_ENAS_fix-w-d_reduce_train_data.pt \
    --train_pair data/nds/nds_ENAS_fix-w-d_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_ENAS_fix-w-d_reduce_test_data.pt \
    --valid_pair data/nds/nds_ENAS_fix-w-d_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 8 --search_space ENAS_fix-w-d --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_NASNet_reduce_train_data.pt \
    --train_pair data/nds/nds_NASNet_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_NASNet_reduce_test_data.pt \
    --valid_pair data/nds/nds_NASNet_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 16 --search_space NASNet --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_DARTS_reduce_train_data.pt \
    --train_pair data/nds/nds_DARTS_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_DARTS_reduce_test_data.pt \
    --valid_pair data/nds/nds_DARTS_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space DARTS --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_ENAS_reduce_train_data.pt \
    --train_pair data/nds/nds_ENAS_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_ENAS_reduce_test_data.pt \
    --valid_pair data/nds/nds_ENAS_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 8 --search_space ENAS --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_PNAS_reduce_train_data.pt \
    --train_pair data/nds/nds_PNAS_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_PNAS_reduce_test_data.pt \
    --valid_pair data/nds/nds_PNAS_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space PNAS --type reduce --graph_d_model 16 --pair_d_model 16

python run.py --do_train --parallel --train_data data/nds/nds_DARTS_lr-wd_reduce_train_data.pt \
    --train_pair data/nds/nds_DARTS_lr-wd_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_DARTS_lr-wd_reduce_test_data.pt \
    --valid_pair data/nds/nds_DARTS_lr-wd_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space DARTS_lr-wd --type reduce --graph_d_model 16 --pair_d_model 16

    
python run.py --do_train --parallel --train_data data/nds/nds_DARTS_fix-w-d_reduce_train_data.pt \
    --train_pair data/nds/nds_DARTS_fix-w-d_reduce_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/nds/nds_DARTS_fix-w-d_reduce_test_data.pt \
    --valid_pair data/nds/nds_DARTS_fix-w-d_reduce_test_pair_k1_d200000_metric_params.pt --dataset nds --n_vocab 11 --search_space DARTS_fix-w-d --type reduce --graph_d_model 16 --pair_d_model 16


