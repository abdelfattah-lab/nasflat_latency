#!/usr/bin/env bash

python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_autoencoder-tb101.pt --data_path  data/tb101micro_autoencoder.json
python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_class_object-tb101.pt --data_path  data/tb101micro_class_object.json
python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_class_scene-tb101.pt --data_path  data/tb101micro_class_scene.json
python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_jigsaw-tb101.pt --data_path  data/tb101micro_jigsaw.json
python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_normal-tb101.pt --data_path  data/tb101micro_normal.json
python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_room_layout-tb101.pt --data_path  data/tb101micro_room_layout.json
python search_methods/reinforce.py --dim 32 --model_path model-dim_32_search_space_transnasbench101_task_segmentsemantic-tb101.pt --data_path  data/tb101micro_segmentsemantic.json







