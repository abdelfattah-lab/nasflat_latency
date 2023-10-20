#!/bin/bash
#SBATCH -J helptest
#SBATCH -o /home/ya255/projects/flan_hardware/fbnet_zcp/logs/%j.out
#SBATCH -e /home/ya255/projects/flan_hardware/fbnet_zcp/logs/%j.err
#SBATCH -N 1
#SBATCH --mem=20000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 1
export PROJ_BPATH="/home/ya255/projects/flan_hardware"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/flan_hardware/fbnet_zcp

python fbnet_model_initializer.py --start 4950 --end 5000
