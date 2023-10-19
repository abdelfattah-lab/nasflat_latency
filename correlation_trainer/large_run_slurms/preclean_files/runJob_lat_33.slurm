#!/bin/bash
#SBATCH -J lat_33
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

python main_abl.py --seed 42 --name_desc lattest --sample_sizes 4 8 12 16 24 32 64 128 --batch_size 8 --space nb201 --representation zcp --gnn_type dense --separate_op_fp --device eyeriss --num_trials 5
