#!/bin/bash
#SBATCH -J archabl34
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

python main_abl.py --seed 42 --name_desc archabl --sample_sizes 64 128 --batch_size 8 --space nb201 --timesteps 2 --representation adj_gin --gnn_type dense --forward_gcn_out_dims 128 128 128 --backward_gcn_out_dims 128 128 128 --replace_bgcn_mlp_dims 128 128 128 --back_mlp --back_y_info
