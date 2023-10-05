#!/usr/bin/env bash
python -i models/pretraining_nasbench201.py  --input_dim 7 --hops 5 --epochs 10 --bs 32 --cfg 4 --seed 4 --name nasbench201
