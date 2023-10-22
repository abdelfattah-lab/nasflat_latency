#!/bin/bash
#SBATCH -J helptest
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

cd /home/ya255/projects/flan_hardware/correlation_trainer/HELP

python main.py --gpu 0 --mode 'meta-train' --seed 42 --num_trials 3 --name_desc 'help_baselines' --num_meta_train_sample 4000 --mc_sampling 10 --num_episodes 2000 --task_index 4 --search_space fbnet --num_samples 10
