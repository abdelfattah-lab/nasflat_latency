#!/usr/bin/env bash
python -i models/pretraining_nasbench301.py  --input_dim 11 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 8 --seed 1 --name nasbench301
