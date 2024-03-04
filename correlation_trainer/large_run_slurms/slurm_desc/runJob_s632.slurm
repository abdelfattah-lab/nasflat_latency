#!/bin/bash
#SBATCH -J s632
#SBATCH -o /home/ya255/projects/flan_hardware/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/flan_hardware/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=30000
#SBATCH -t 24:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4
export PROJ_BPATH="/home/ya255/projects/flan_hardware"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/flan_hardware/correlation_trainer

python main_trf.py --seed 42 --name_desc study_6_3_2_suppl --sample_sizes 256 --task_index 5 --representation adj_gin_a2vcatezcp --num_trials 5 --transfer_sample_sizes 5 6 7 8 9 10 20 --transfer_lr 0.001 --transfer_epochs 30 --transfer_hwemb --space fbnet --gnn_type ensemble --sampling_metric a2vcatezcp --sampling_method kmeancosine
