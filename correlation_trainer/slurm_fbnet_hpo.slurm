#!/bin/bash
#SBATCH -J archabl
#SBATCH -o /home/ya255/projects/flan_hardware/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/flan_hardware/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=30000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-01
#SBATCH --partition=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4

export PROJ_BPATH="/home/ya255/projects/flan_hardware"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/flan_hardware/correlation_trainer

python optuna_hpo_fbnet.py