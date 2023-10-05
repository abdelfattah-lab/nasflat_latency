#!/usr/bin/env bash

python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task class_scene
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task class_object
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task autoencoder
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task normal
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task jigsaw
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task room_layout
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task segmentsemantic

