# On Latency Predictors for Neural Architecture Search

This repository contains the necessary resources and instructions to reproduce the results presented in our paper "On Latency Predictors for Neural Architecture Search." Our work introduces a comprehensive suite of latency prediction tasks and a novel latency predictor, NASFLAT, that significantly outperforms existing methods in hardware-aware NAS.

## Abstract

We address the challenge of co-optimizing accuracy and latency in neural network deployment, crucial for hardware-aware neural architecture search (NAS). Our work introduces a suite of latency prediction tasks and a latency predictor that incorporates advanced techniques in predictor architecture, NN sample selection, hardware device representation, and operation encoding schemes. The predictor, NASFLAT, demonstrates superior performance in latency prediction, facilitating a more efficient NAS process.

## Environment Setup

- Ensure the `env_setup.py` script is executed correctly for environment setup. Modify the script paths as necessary for your system.

## Dataset Preparation

- Download the NDS dataset from [here](https://dl.fbaipublicfiles.com/nds/data.zip) and place it in the `nas_embedding_suite` folder with the structure `NDS/nds_data/*.json`.
- Download and unzip `nasflat_embeddings_04_03_24.zip` into `./nas_embedding_suite/` from [Google Drive](https://drive.google.com/file/d/1oJyH0zox_cbRUX-hgzkliOLAUaz3gIxw/view?usp=sharing).

## Execution Instructions

- Training and testing commands for the predictors are in `./correlation_trainer/large_run_slurms/unified_joblist.log`.
- To reproduce results for MultiPredict and HELP, refer to `multipredict_unified_joblist.log` and `help_unified_joblist.log`.
- For SLURM setups, use `parallelized_executor.sh`, adapting it as necessary for your environment. These commands can also be adjusted for non-SLURM execution.

## Contributions

1. **NN Sampling Methods:** We improve few-shot latency predictor efficiency by investigating and testing different NN sampling methods.
2. **Operation Embeddings:** Introducing hardware-specific NN operation embeddings and supplementary encodings, we enhance prediction accuracy significantly.
3. **NASFLAT Predictor:** NASFLAT combines effective techniques to deliver substantial improvements in latency prediction, aiding faster and more efficient NAS.

Our methodology provides deep insights into latency predictor design, yielding substantial improvements across a variety of experimental settings and demonstrating considerable speed-up in NAS processes.

## Example Executions

Below are specific example commands that demonstrate how to execute various processes within the framework. These examples cover training from scratch, utilizing supplementary encodings, transferring predictors between spaces, and running NAS on a given search space.

The files referenced below are located at correlation_trainer and nas_search.

### Train on FBNet using CAZ Sampler

```bash
python main_trf.py --seed 42 --name_desc study_6_5_f_zcp --sample_sizes 800 --task_index 5 --representation adj_gin_zcp --num_trials 3 --transfer_sample_sizes 20 --transfer_lr 0.001 --transfer_epochs 30 --transfer_hwemb --space fbnet --gnn_type ensemble --sampling_metric a2vcatezcp --ensemble_fuse_method add
```

### Compare HWEmb Transfer and Hardware Embedding Effectiveness

```bash
python main_trf.py --seed 42 --name_desc arch_abl --sample_sizes 512 --representation adj_gin --num_trials 7 --transfer_sample_sizes 5 10 20 --transfer_lr 0.0001 --transfer_epochs 20 --transfer_hwemb --hwemb_to_mlp --task_index 4 --space nb201
python main_trf.py --seed 42 --name_desc arch_abl --sample_sizes 512 --representation adj_gin --num_trials 7 --transfer_sample_sizes 5 10 20 --transfer_lr 0.0001 --transfer_epochs 20 --transfer_hwemb --task_index 4 --space nb201
python main_trf.py --seed 42 --name_desc arch_abl --sample_sizes 512 --representation adj_gin --num_trials 7 --transfer_sample_sizes 5 10 20 --transfer_lr 0.0001 --transfer_epochs 20  --task_index 4 --space nb201
python main_trf.py --seed 42 --name_desc arch_abl --sample_sizes 512 --representation adj_gin --num_trials 7 --transfer_sample_sizes 5 10 20 --transfer_lr 0.0001 --transfer_epochs 20 --hwemb_to_mlp --task_index 4 --space nb201
```

### Investigate Samplers

```bash
python main_trf.py --seed 42 --name_desc study_6_3_1_t2 --sample_sizes 512 --representation adj_gin --num_trials 7 --transfer_sample_sizes 5 10 15 20 30 --transfer_lr 0.0001 --transfer_epochs 20 --transfer_hwemb --task_index 1 --space nb201 --gnn_type ensemble --sampling_metric [random/params/arch2vec/cate/zcp/a2vcatezcp/latency]
```

### Investigate Supplementary Encodings

```bash
python main_trf.py --seed 42 --name_desc study_6_3_2 --sample_sizes 512 --representation [adj_gin/adj_gin_arch2vec/adj_gin_zcp/adj_gin_a2vcatezcp/adj_gin_cate] --num_trials 5 --transfer_sample_sizes 5 10 20 --transfer_lr 0.0001 --transfer_epochs 20 --transfer_hwemb --task_index 3 --space nb201 --gnn_type ensemble --sampling_metric a2vcatezcp
```


### Run MultiPredict Baselines

All scripts can be found at correlation_trainer/large_run_slurms/multipredict_unified_joblist.log

```bash
python fsh_advanced_training.py --name_desc multipredict_baseline_r --task_index 0 --space fbnet --emb_transfer_samples 16
```

### Run HELP Baselines

All scripts can be found at correlation_trainer/large_run_slurms/help_unified_joblist.log

```bash
python main.py --gpu 0 --mode 'meta-train' --seed 42 --num_trials 3 --name_desc 'help_baselines_r' --num_meta_train_sample 4000 --mc_sampling 10 --num_episodes 2000 --task_index 5 --search_space fbnet --num_samples 10
```