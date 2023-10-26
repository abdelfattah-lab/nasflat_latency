#!/bin/bash
#SBATCH -J s641ts
#SBATCH -o /home/ya255/projects/flan_hardware/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/flan_hardware/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=30000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4
export PROJ_BPATH="/home/ya255/projects/flan_hardware"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/flan_hardware/correlation_trainer

python main_trf.py --seed 42 --name_desc study_6_6_1 --sample_sizes 256 --task_index 4 --space fbnet --gnn_type ensemble  --representation adj_gin_zcp --sampling_metric zcp --sampling_method cosine --num_trials 3 --transfer_sample_sizes 5 10 15 20 30 --transfer_lr 0.003 --transfer_epochs 40 --transfer_hwemb # with transferemb and hw-op-emb, cosine zcp sampler, adjginzcp
