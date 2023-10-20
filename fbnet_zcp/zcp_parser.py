import os
import pandas as pd
# spearmanr
from scipy.stats import spearmanr
from tqdm import tqdm

from matplotlib import pyplot as plt

from pylab import *
params = {
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False
}
rcParams.update(params)

valid_zcp_spacelist = ['Amoeba','DARTS','DARTS_fix-w-d','DARTS_lr-wd','ENAS','ENAS_fix-w-d','PNAS','PNAS_fix-w-d','NASNet']

master_space_corrs = pd.DataFrame()
for space in valid_zcp_spacelist:
    master_set = pd.concat([pd.read_json('./' + space + "_zcps/" + fx).T for fx in tqdm(os.listdir('./' + space + "_zcps/"))])
    print(master_set.shape)
    # Remove duplicates by index
    master_set = master_set[~master_set.index.duplicated(keep='first')]
    master_set_backup = master_set
    # master_set = master_set_backup.sample(n=500)
    print(master_set.shape)
    our_dict = {}
    for col_name in master_set.columns:
        corr, p = spearmanr(master_set[col_name], master_set["val_accuracy"])
        our_dict[col_name] = corr
    main_dict = pd.DataFrame.from_dict({key: [our_dict[key]] for key in our_dict.keys()}).T
    main_dict.columns = [space]
    master_space_corrs = pd.concat([master_space_corrs, main_dict], axis=1)

print(master_space_corrs)
master_space_corrs.drop("val_accuracy", inplace=True)
# plt.ylabel = "Spearman Correlation"
ax = master_space_corrs.plot.bar(figsize=(20,10), rot=0)
ax.set_ylabel("Spearman Correlation")
plt.savefig("corrs_zcp_x.png")
plt.cla()
plt.clf()
# plt.ylabel = "Spearman Correlation"
ax = master_space_corrs.T.plot.bar(figsize=(20,10), rot=0)
ax.set_ylabel("Spearman Correlation")
plt.savefig("corrs_SS_x.png")