#!/usr/bin/env bash

python search_methods/reinforce_search_NB201_8x8.py --dataset_name cifar10_valid_converged --latent_dim 32 --model_path model-dim_32_search_space_nasbench201-nasbench201.pt

# python search_methods/reinforce_search_NB201_8x8.py --dataset_name cifar100 --latent_dim 32 --model_path model-nasbench201.pt

# python search_methods/reinforce_search_NB201_8x8.py --dataset_name ImageNet16_120 --latent_dim 32 --model_path model-nasbench201.pt

