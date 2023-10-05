#!/usr/bin/env bash

python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_Amoeba_type_normal-nds.pt --data_path data/nds_Amoeba/nds_Amoeba_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_Amoeba_type_reduce-nds.pt --data_path data/nds_Amoeba/nds_Amoeba_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_DARTS_fix-w-d_type_normal-nds.pt --data_path data/nds_DARTS_fix-w-d/nds_DARTS_fix-w-d_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_DARTS_fix-w-d_type_reduce-nds.pt --data_path data/nds_DARTS_fix-w-d/nds_DARTS_fix-w-d_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_DARTS_lr-wd_type_normal-nds.pt --data_path data/nds_DARTS_lr-wd/nds_DARTS_lr-wd_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_DARTS_lr-wd_type_reduce-nds.pt --data_path data/nds_DARTS_lr-wd/nds_DARTS_lr-wd_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_DARTS_type_normal-nds.pt --data_path data/nds_DARTS/nds_DARTS_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_DARTS_type_reduce-nds.pt --data_path data/nds_DARTS/nds_DARTS_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_ENAS_fix-w-d_type_normal-nds.pt --data_path data/nds_ENAS_fix-w-d/nds_ENAS_fix-w-d_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_ENAS_fix-w-d_type_reduce-nds.pt --data_path data/nds_ENAS_fix-w-d/nds_ENAS_fix-w-d_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_ENAS_type_normal-nds.pt --data_path data/nds_ENAS/nds_ENAS_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_ENAS_type_reduce-nds.pt --data_path data/nds_ENAS/nds_ENAS_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_NASNet_type_normal-nds.pt --data_path data/nds_NASNet/nds_NASNet_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_NASNet_type_reduce-nds.pt --data_path data/nds_NASNet/nds_NASNet_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_PNAS_fix-w-d_type_normal-nds.pt --data_path data/nds_PNAS_fix-w-d/nds_PNAS_fix-w-d_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_PNAS_fix-w-d_type_reduce-nds.pt --data_path data/nds_PNAS_fix-w-d/nds_PNAS_fix-w-d_reduce.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_PNAS_type_normal-nds.pt --data_path data/nds_PNAS/nds_PNAS_normal.json
python search_methods/reinforce.py --dim 16 --model_path  model-dim_16_search_space_PNAS_type_reduce-nds.pt --data_path data/nds_PNAS/nds_PNAS_reduce.json



















