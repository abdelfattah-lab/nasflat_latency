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

## Reproducing Results

For a detailed guide on reproducing our results, please refer to the provided execution instructions and ensure that the dataset and embedding preparations are correctly followed. The structured approach allows for a comprehensive evaluation and replication of our study's findings.
