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

python main.py --gpu 0 --search_space nasbench201 --mode 'meta-train' --num_samples 20 --seed 42 --num_trials 5 --num_meta_train_sample 512 --meta_train_devices '1080ti_1,1080ti_32,1080ti_256,silver_4114,silver_4210r,samsung_a50,pixel3,essential_ph_1,samsung_s7' --meta_valid_devices 'titanx_1,titanx_32,titanx_256,gold_6240' --meta_test_devices 'titan_rtx_256,gold_6226,fpga,pixel2,raspi4,eyeriss'
