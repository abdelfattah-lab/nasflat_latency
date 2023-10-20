#!/usr/bin/env bash
python -i  models/pretraining_fbnet.py  --input_dim 5 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 50 --seed 1 --name nasbench101
