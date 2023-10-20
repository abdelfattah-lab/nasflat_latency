
import os
import pandas as pd
# spearmanr
from scipy.stats import spearmanr
from tqdm import tqdm

valid_zcp_spacelist = ['Amoeba','DARTS','DARTS_fix-w-d','DARTS_lr-wd','ENAS','ENAS_fix-w-d','PNAS','PNAS_fix-w-d','NASNet']

master_space_corrs = pd.DataFrame()
for space in valid_zcp_spacelist:
    master_set = pd.concat([pd.read_json('./' + space + "_zcps/" + fx).T for fx in tqdm(os.listdir('./' + space + "_zcps/"))])
    print(master_set.shape)
    # Remove duplicates by index
    master_set = master_set[~master_set.index.duplicated(keep='first')]
    master_set = master_set.sort_index()
    print(master_set.shape)
    # Save master_set to file
    master_set.to_csv("./../embedding_datasets/nds_zcps/" + space + "_zcps.csv")
