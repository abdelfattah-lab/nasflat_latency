####### PREPROCESSING #######

# !!!! ALL SS !!!!
python preprocessing/gen_alljson.py

# NASBench101
python preprocessing/gen_json.py
# NASBench201
python preprocessing/gen_json_nb201.py
# NASBench301
python preprocessing/gen_json_darts.py
# NDS Normal Cell
python preprocessing/gen_json_nds.py --type normal  --search_space Amoeba 
python preprocessing/gen_json_nds.py --type normal  --search_space PNAS_fix-w-d 
python preprocessing/gen_json_nds.py --type normal  --search_space ENAS_fix-w-d 
python preprocessing/gen_json_nds.py --type normal  --search_space NASNet 
python preprocessing/gen_json_nds.py --type normal  --search_space DARTS 
python preprocessing/gen_json_nds.py --type normal  --search_space ENAS 
python preprocessing/gen_json_nds.py --type normal  --search_space PNAS 
python preprocessing/gen_json_nds.py --type normal  --search_space DARTS_lr-wd 
python preprocessing/gen_json_nds.py --type normal  --search_space DARTS_fix-w-d 
# NDS Reduce Cell
python preprocessing/gen_json_nds.py --search_space Amoeba --type reduce
python preprocessing/gen_json_nds.py --search_space PNAS_fix-w-d --type reduce
python preprocessing/gen_json_nds.py --search_space ENAS_fix-w-d --type reduce
python preprocessing/gen_json_nds.py --search_space NASNet --type reduce
python preprocessing/gen_json_nds.py --search_space DARTS --type reduce
python preprocessing/gen_json_nds.py --search_space ENAS --type reduce
python preprocessing/gen_json_nds.py --search_space PNAS --type reduce
python preprocessing/gen_json_nds.py --search_space DARTS_lr-wd --type reduce
python preprocessing/gen_json_nds.py --search_space DARTS_fix-w-d --type reduce
# TransNASBench-101 Micro
python preprocessing/gen_json_transnasbench101.py  --task class_scene
python preprocessing/gen_json_transnasbench101.py  --task class_object
python preprocessing/gen_json_transnasbench101.py  --task autoencoder
python preprocessing/gen_json_transnasbench101.py  --task normal
python preprocessing/gen_json_transnasbench101.py  --task jigsaw
python preprocessing/gen_json_transnasbench101.py  --task room_layout
python preprocessing/gen_json_transnasbench101.py  --task segmentsemantic

####### DATA GENERATION #######

# !!!! ALL SS !!!!
python preprocessing/data_generate.py --dataset all_ss --flag extract_seq
python preprocessing/data_generate.py --dataset all_ss --flag build_pair --k 2 --d 2000000 --metric params

# NASBench101
python preprocessing/data_generate.py --dataset nasbench101 --flag extract_seq
python preprocessing/data_generate.py --dataset nasbench101 --flag build_pair --k 2 --d 2000000 --metric params

# NASBench201
python preprocessing/data_generate.py --dataset nasbench201 --flag extract_seq
python preprocessing/data_generate.py --dataset nasbench201 --flag build_pair --k 2 --d 200000 --metric params

# NASBench301
python preprocessing/data_generate.py --dataset nasbench301 --flag extract_seq
python preprocessing/data_generate.py --dataset nasbench301 --flag build_pair --k 2 --d 2000000 --metric flops

# NDS
python preprocessing/data_generate.py --dataset nds --search_space Amoeba --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space Amoeba  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space Amoeba  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space Amoeba --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space PNAS_fix-w-d --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space ENAS_fix-w-d --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space NASNet --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space NASNet  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space NASNet  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space NASNet --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space DARTS --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space DARTS  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space DARTS  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space DARTS --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space ENAS --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space ENAS  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space ENAS  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space ENAS --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space PNAS --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space PNAS  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space PNAS  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space PNAS --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space DARTS_lr-wd --type reduce --flag build_pair --k 2 --d 200000 --metric params

python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d --type normal --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d  --type normal --flag build_pair --k 2 --d 200000 --metric params
python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d  --type reduce --flag extract_seq
python preprocessing/data_generate.py --dataset nds --search_space DARTS_fix-w-d --type reduce --flag build_pair --k 2 --d 200000 --metric params

# TransNASBench-101 Micro
python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task class_scene
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task class_scene

python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task class_object
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task class_object

python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task autoencoder
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task autoencoder

python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task normal
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task normal

python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task jigsaw
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task jigsaw

python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task room_layout
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task room_layout

python preprocessing/data_generate.py --dataset transnasbench101 --flag extract_seq --task segmentsemantic
python preprocessing/data_generate.py --dataset transnasbench101 --flag build_pair --k 2 --d 200000 --metric params --task segmentsemantic

####### PRETRAINING #######

# !!!! ALL SS !!!!
bash run_scripts/pretrain_allss.sh

bash run_scripts/pretrain_nasbench101.sh
bash run_scripts/pretrain_nasbench201.sh
bash run_scripts/pretrain_nasbench301.sh
bash run_scripts/pretrain_nds.sh
bash run_scripts/pretrain_transnasbench101.sh

####### CATE Extraction #######

# !!!! ALL SS !!!!
python inference/inference.py --pretrained_path model/all_ss_model_best.pth.tar --train_data data/all_ss/train_data.pt --valid_data data/all_ss/test_data.pt --dataset all_ss --search_space all_ss --n_vocab 20 --graph_d_model 32 --pair_d_model 32

# NASBench101
python inference/inference.py --pretrained_path model/nasbench101_model_best.pth.tar --train_data data/nasbench101/train_data.pt --valid_data data/nasbench101/test_data.pt --dataset nasbench101 --search_space nasbench101 --n_vocab 5 --graph_d_model 32 --pair_d_model 32
# NASBench201
python inference/inference.py --pretrained_path model/nasbench201_model_best.pth.tar --train_data data/nasbench201/train_data.pt --valid_data data/nasbench201/test_data.pt --dataset nasbench201 --search_space nasbench201 --n_vocab 7 --graph_d_model 32 --pair_d_model 32
# NASBench301
python inference/inference.py --pretrained_path model/nasbench301_model_best.pth.tar --train_data data/nasbench301/train_data.pt --valid_data data/nasbench301/test_data.pt --dataset nasbench301 --search_space nasbench301 --n_vocab 11 --graph_d_model 32 --pair_d_model 32
# NDS
python inference/inference.py --pretrained_path model/nds_Amoeba_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_Amoeba_normal_train_data.pt --valid_data data/nds/nds_Amoeba_normal_test_data.pt --dataset nds --search_space Amoeba --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_Amoeba_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_Amoeba_reduce_train_data.pt --valid_data data/nds/nds_Amoeba_reduce_test_data.pt --dataset nds --search_space Amoeba --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_PNAS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_PNAS_fix-w-d_normal_test_data.pt --dataset nds --search_space PNAS_fix-w-d --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_PNAS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_PNAS_fix-w-d_reduce_test_data.pt --dataset nds --search_space PNAS_fix-w-d --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_ENAS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_ENAS_fix-w-d_normal_test_data.pt --dataset nds --search_space ENAS_fix-w-d --type normal --n_vocab 8 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_ENAS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_ENAS_fix-w-d_reduce_test_data.pt --dataset nds --search_space ENAS_fix-w-d --type reduce --n_vocab 8 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_NASNet_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_NASNet_normal_train_data.pt --valid_data data/nds/nds_NASNet_normal_test_data.pt --dataset nds --search_space NASNet --type normal --n_vocab 16 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_NASNet_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_NASNet_reduce_train_data.pt --valid_data data/nds/nds_NASNet_reduce_test_data.pt --dataset nds --search_space NASNet --type reduce --n_vocab 16 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_DARTS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_normal_train_data.pt --valid_data data/nds/nds_DARTS_normal_test_data.pt --dataset nds --search_space DARTS --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_DARTS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_reduce_train_data.pt --valid_data data/nds/nds_DARTS_reduce_test_data.pt --dataset nds --search_space DARTS --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_ENAS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_normal_train_data.pt --valid_data data/nds/nds_ENAS_normal_test_data.pt --dataset nds --search_space ENAS --type normal --n_vocab 8 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_ENAS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_ENAS_reduce_train_data.pt --valid_data data/nds/nds_ENAS_reduce_test_data.pt --dataset nds --search_space ENAS --type reduce --n_vocab 8 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_PNAS_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_normal_train_data.pt --valid_data data/nds/nds_PNAS_normal_test_data.pt --dataset nds --search_space PNAS --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_PNAS_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_PNAS_reduce_train_data.pt --valid_data data/nds/nds_PNAS_reduce_test_data.pt --dataset nds --search_space PNAS --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_DARTS_lr-wd_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_lr-wd_normal_train_data.pt --valid_data data/nds/nds_DARTS_lr-wd_normal_test_data.pt --dataset nds --search_space DARTS_lr-wd --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_DARTS_lr-wd_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_lr-wd_reduce_train_data.pt --valid_data data/nds/nds_DARTS_lr-wd_reduce_test_data.pt --dataset nds --search_space DARTS_lr-wd --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16

python inference/inference.py --pretrained_path model/nds_DARTS_fix-w-d_normal_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_fix-w-d_normal_train_data.pt --valid_data data/nds/nds_DARTS_fix-w-d_normal_test_data.pt --dataset nds --search_space DARTS_fix-w-d --type normal --n_vocab 11 --graph_d_model 16 --pair_d_model 16
python inference/inference.py --pretrained_path model/nds_DARTS_fix-w-d_reduce_checkpoint_Epoch_10.pth.tar --train_data data/nds/nds_DARTS_fix-w-d_reduce_train_data.pt --valid_data data/nds/nds_DARTS_fix-w-d_reduce_test_data.pt --dataset nds --search_space DARTS_fix-w-d --type reduce --n_vocab 11 --graph_d_model 16 --pair_d_model 16
# TransNASBench101 Micro
python inference/inference.py --pretrained_path model/transnasbench101_autoencoder_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/autoencoder_train_data.pt --valid_data data/transnasbench101/autoencoder_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task autoencoder --n_vocab 6 --graph_d_model 32 --pair_d_model 32

python inference/inference.py --pretrained_path model/transnasbench101_class_object_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/class_object_train_data.pt --valid_data data/transnasbench101/class_object_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task class_object --n_vocab 6 --graph_d_model 32 --pair_d_model 32

python inference/inference.py --pretrained_path model/transnasbench101_class_scene_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/class_scene_train_data.pt --valid_data data/transnasbench101/class_scene_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task class_scene --n_vocab 6 --graph_d_model 32 --pair_d_model 32

python inference/inference.py --pretrained_path model/transnasbench101_jigsaw_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/jigsaw_train_data.pt --valid_data data/transnasbench101/jigsaw_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task jigsaw --n_vocab 6 --graph_d_model 32 --pair_d_model 32

python inference/inference.py --pretrained_path model/transnasbench101_normal_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/normal_train_data.pt --valid_data data/transnasbench101/normal_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task normal --n_vocab 6 --graph_d_model 32 --pair_d_model 32

python inference/inference.py --pretrained_path model/transnasbench101_room_layout_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/room_layout_train_data.pt --valid_data data/transnasbench101/room_layout_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task room_layout --n_vocab 6 --graph_d_model 32 --pair_d_model 32

python inference/inference.py --pretrained_path model/transnasbench101_segmentsemantic_checkpoint_Epoch_10.pth.tar --train_data data/transnasbench101/segmentsemantic_train_data.pt --valid_data data/transnasbench101/segmentsemantic_test_data.pt --dataset transnasbench101 --search_space transnasbench101 --task segmentsemantic --n_vocab 6 --graph_d_model 32 --pair_d_model 32
