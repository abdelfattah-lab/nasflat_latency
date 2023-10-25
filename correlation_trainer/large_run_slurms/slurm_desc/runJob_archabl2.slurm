#!/bin/bash
#SBATCH -J archabl2
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

python main_trf.py --seed 42 --name_desc study_6_2_2_f0 --sample_sizes 512 --representation adj_gin --num_trials 3 --transfer_sample_sizes 5 10 20 --transfer_lr 0.0001 --transfer_epochs 20 --transfer_hwemb --task_index 0 --space fbnet --gnn_type ensemble
