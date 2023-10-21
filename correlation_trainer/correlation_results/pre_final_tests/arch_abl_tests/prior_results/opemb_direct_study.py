import pandas as pd

key_ = 64

    # df_opemb_direct = df_opemb_direct[df_opemb_direct['key'] == key_]
    # df_detach_abl = df_detach_abl[df_detach_abl['key'] == key_]

def get_matching_rows_for_space(space):
    # Load data from both CSVs
    df_opemb_direct = pd.read_csv(f'./opemb_direct/{space}_samp_eff.csv')
    df_detach_abl = pd.read_csv(f'./detach_abl/{space}_samp_eff.csv')
    
    # Add a column for opemb_direct in the detach_abl dataframe
    df_detach_abl['opemb_direct'] = False
    
    df_opemb_direct = df_opemb_direct[df_opemb_direct['key'] == key_]
    df_detach_abl = df_detach_abl[df_detach_abl['key'] == key_]

    df_opemb_direct['kdt'] = df_opemb_direct['kdt'].round(4)
    df_detach_abl['kdt'] = df_detach_abl['kdt'].round(4)
    # Prepare column lists for matching
    drop_cols = ['name_desc', 'spr', 'spr_std', 'kdt_std', 'detach_mode', 'kdt', 'opemb_direct']
    match_cols = [col for col in df_opemb_direct.columns if col not in drop_cols]
    # import pdb; pdb.set_trace()
    # Find matching rows
    common = df_opemb_direct.merge(df_detach_abl, on=match_cols, suffixes=('_opemb', '_detach'))
    
    return space, common['kdt_opemb'].values, common['kdt_detach'].values

# Gather data for each space
spaces = ['PNAS', 'ENAS', 'nb201', 'nb101']
results = [get_matching_rows_for_space(space) for space in spaces]

# Generate LaTeX table
print(r"\begin{table}")
print(r"\centering")
print(r"\begin{tabular}{|c|c|c|}")
print(r"\hline")
print("Space & KDT (opemb\_direct) & KDT (detach\_abl) \\")
print(r"\hline")
for space, kdt_opemb, kdt_detach in results:
    # Assuming single matched rows, so using [0] to get the kdt value
    print(f"{space} & {kdt_opemb[0]} & {kdt_detach[0]} \\\\")
    # print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")
