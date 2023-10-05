#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run.py --do_train --parallel --train_data data/transnasbench101/class_scene_train_data.pt \
    --train_pair data/transnasbench101/class_scene_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/class_scene_test_data.pt --task class_scene \
    --valid_pair data/transnasbench101/class_scene_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task class_scene --graph_d_model 32 --pair_d_model 32


python run.py --do_train --parallel --train_data data/transnasbench101/class_object_train_data.pt \
    --train_pair data/transnasbench101/class_object_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/class_object_test_data.pt --task class_object \
    --valid_pair data/transnasbench101/class_object_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task class_object --graph_d_model 32 --pair_d_model 32


python run.py --do_train --parallel --train_data data/transnasbench101/autoencoder_train_data.pt \
    --train_pair data/transnasbench101/autoencoder_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/autoencoder_test_data.pt --task autoencoder \
    --valid_pair data/transnasbench101/autoencoder_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task autoencoder --graph_d_model 32 --pair_d_model 32


python run.py --do_train --parallel --train_data data/transnasbench101/normal_train_data.pt \
    --train_pair data/transnasbench101/normal_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/normal_test_data.pt --task normal \
    --valid_pair data/transnasbench101/normal_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task normal --graph_d_model 32 --pair_d_model 32


python run.py --do_train --parallel --train_data data/transnasbench101/jigsaw_train_data.pt \
    --train_pair data/transnasbench101/jigsaw_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/jigsaw_test_data.pt --task jigsaw \
    --valid_pair data/transnasbench101/jigsaw_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task jigsaw --graph_d_model 32 --pair_d_model 32


python run.py --do_train --parallel --train_data data/transnasbench101/room_layout_train_data.pt \
    --train_pair data/transnasbench101/room_layout_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/room_layout_test_data.pt --task room_layout \
    --valid_pair data/transnasbench101/room_layout_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task room_layout --graph_d_model 32 --pair_d_model 32


python run.py --do_train --parallel --train_data data/transnasbench101/segmentsemantic_train_data.pt \
    --train_pair data/transnasbench101/segmentsemantic_train_pair_k1_d200000_metric_params.pt  \
    --valid_data data/transnasbench101/segmentsemantic_test_data.pt --task segmentsemantic \
    --valid_pair data/transnasbench101/segmentsemantic_test_pair_k1_d200000_metric_params.pt --dataset transnasbench101 --n_vocab 6 --search_space transnasbench101 --task segmentsemantic --graph_d_model 32 --pair_d_model 32
