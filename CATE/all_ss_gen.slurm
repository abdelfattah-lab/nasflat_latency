#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/CATE/large_scale_run_logs/help_tests_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/CATE/large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem=200000
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-01 


export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate cate

cd /home/ya255/projects/iclr_nas_embedding/CATE/

# python preprocessing/gen_alljson.py
# python preprocessing/data_generate.py --dataset all_ss --flag extract_seq
# python preprocessing/data_generate.py --dataset all_ss --flag build_pair --k 2 --d 5000000 --metric params 
bash run_scripts/pretrain_allss.sh
python inference/inference.py --pretrained_path model/all_ss_model_best.pth.tar --train_data data/all_ss/train_data.pt --valid_data data/all_ss/test_data.pt --dataset all_ss --search_space all_ss --n_vocab 20 --graph_d_model 32 --pair_d_model 32
