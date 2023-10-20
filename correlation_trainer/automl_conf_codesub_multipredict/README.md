### MultiPredict Transfer Learning Implementation

## TL Experiments (Latency Cross-Hardware) (Table 1, 2, 3)

1. The .csv for tests can be found in config_mapper.log
2. Please identify the base_config_%s.json % (str(test_idx)) from the config_mapper to find the corresponding Transfer Learning tests
3. Configuration changes can be made in the base_config_*.json files to run the main code.
4. You can run the TL code simply by executing ```python -i fsh_advanced_training.py --config_idx test_idx```

## Transfer Search Space Experiments (Accuracy Cross-Search-Space) (Figure 9)

1. To run: ```transfer_experiments.py --crosstask_m 'all'```. The task transfer sets are provided in the ```transfer_experiments.py``` codebase. 
2. Custom transfer tasks can be designed and executed by adding an if condition to crosstask_m and setting the appropriate ```training_spaces``` and ```testing_sets```. 
3. Confusion matrices will be saved in ```fs_zcp_crosstask```. Note that this test studies whether pre-training with ZCP of a source search space helps in few-shot prediction on a target search space.

## HWL FBNet <-> NASBench201 transfer (Latency Cross-Search-Space) (Figure 7)

1. To run: ```python hwl_fb_nb201_transfer.py --mode 'fb_to_nb' / 'nb_to_fb'```
2. The graphs will be saved in ```cross_arch_nb_fb_hwl_vec```

## Link to unified dataset
1. You can download the unified_dataset.zip and unzip it in ```./unified_dataset/```. [[Dataset Link](https://osf.io/kxty7/?view_only=a1324f5105ef4d5c8cbc173d6a71a741)]
2. ```./unified_dataset/NAS-Bench-201-v1_1-096897.pth``` needs to be downloaded separately. Use the following command: ```gdown 16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_``` 