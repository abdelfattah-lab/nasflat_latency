####### PREPROCESSING #######

# !!!! ALL SS !!!!
python preprocessing/gen_alljson.py     

# NASBench101
python preprocessing/gen_json.py             
# NASBench201
python preprocessing/nasbench201_json.py     
# NASBench301
python preprocessing/nasbench301_json.py     

# NDS Normal Cell
python preprocessing/nds_json.py --search_space Amoeba --type normal 
python preprocessing/nds_json.py --search_space PNAS_fix-w-d --type normal 
python preprocessing/nds_json.py --search_space ENAS_fix-w-d --type normal 
python preprocessing/nds_json.py --search_space NASNet --type normal 
python preprocessing/nds_json.py --search_space DARTS --type normal 
python preprocessing/nds_json.py --search_space ENAS --type normal 
python preprocessing/nds_json.py --search_space PNAS --type normal 
python preprocessing/nds_json.py --search_space DARTS_lr-wd --type normal 
python preprocessing/nds_json.py --search_space DARTS_fix-w-d --type normal 

# NDS Reduce Cell
python preprocessing/nds_json.py --search_space Amoeba --type reduce
python preprocessing/nds_json.py --search_space PNAS_fix-w-d --type reduce
python preprocessing/nds_json.py --search_space ENAS_fix-w-d --type reduce
python preprocessing/nds_json.py --search_space NASNet --type reduce
python preprocessing/nds_json.py --search_space DARTS --type reduce
python preprocessing/nds_json.py --search_space ENAS --type reduce
python preprocessing/nds_json.py --search_space PNAS --type reduce
python preprocessing/nds_json.py --search_space DARTS_lr-wd --type reduce
python preprocessing/nds_json.py --search_space DARTS_fix-w-d --type reduce

# TransNASBench-101 Micro
python preprocessing/transnasbench101_json.py --task class_scene
python preprocessing/transnasbench101_json.py --task class_object
python preprocessing/transnasbench101_json.py --task autoencoder
python preprocessing/transnasbench101_json.py --task normal
python preprocessing/transnasbench101_json.py --task jigsaw
python preprocessing/transnasbench101_json.py --task room_layout
python preprocessing/transnasbench101_json.py --task segmentsemantic

####### PRETRAINING #######

# !!!! ALL SS !!!!
python -i models/pretraining_allss.py --input_dim 20 --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name all_ss 

# NASBench101
python -i  models/pretraining_nasbench101.py  --input_dim 5 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 50 --seed 1 --name nasbench101
# NASBench201
python -i models/pretraining_nasbench201.py  --input_dim 7 --hops 5 --epochs 10 --bs 32 --cfg 4 --seed 4 --name nasbench201
# NASBench301
python -i models/pretraining_nasbench301.py  --input_dim 11 --hops 5 --dim 32 --cfg 4 --bs 32 --epochs 8 --seed 1 --name nasbench301
# NDS Normal Cell
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space Amoeba --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS_fix-w-d --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS_fix-w-d --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space NASNet --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_lr-wd --type normal
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_fix-w-d --type normal
# NDS Reduce Cell
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space Amoeba --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS_fix-w-d --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS_fix-w-d --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space NASNet --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space ENAS --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space PNAS --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_lr-wd --type reduce
python models/pretraining_nds.py --hops 5 --dim 16 --cfg 4 --bs 32 --epochs 100 --seed 1 --name nds --search_space DARTS_fix-w-d --type reduce
# TransNASBench-101 Micro
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task class_scene
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task class_object
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task autoencoder
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task normal
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task jigsaw
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task room_layout
python models/pretraining_transnasbench101.py   --hops 5 --latent_dim 32 --cfg 4 --bs 32 --epochs 100 --seed 1 --name tb101 --task segmentsemantic

####### Arch2Vec Extraction #######

# !!!! ALL SS !!!!
bash run_scripts/extract_allss.sh

# NASBench101
bash run_scripts/extract_arch2vec.sh
# NASBench201
bash run_scripts/extract_arch2vec_nasbench201.sh
# NASBench301
bash run_scripts/extract_arch2vec_nasbench301.sh
# NDS
bash run_scripts/extract_arch2vec_nds.sh
# TransNASBench-101 Micro
bash run_scripts/extract_arch2vec_transnasbench101.sh