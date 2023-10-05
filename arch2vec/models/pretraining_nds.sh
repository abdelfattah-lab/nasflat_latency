#!/usr/bin/env bash

python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space Amoeba --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS_fix-w-d --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS_fix-w-d --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space NASNet --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_lr-wd --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_fix-w-d --type normal


python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space Amoeba --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS_fix-w-d --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS_fix-w-d --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space NASNet --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_lr-wd --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_fix-w-d --type reduce