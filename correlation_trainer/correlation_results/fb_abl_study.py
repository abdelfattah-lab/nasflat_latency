import pandas as pd

def csv_to_latex(fpath, key=64):
    # Load CSV
    df = pd.read_csv(fpath)
    
    # Filter rows where 'key' == key
    df = df[df['key'] == key]
    # Sort by increasing 'kdt'
    df = df.sort_values(by='kdt')
    # Filter columns
    cols = ["timesteps", "bmlp", "back_y_info", "back_opemb", "back_opemb_only", "kdt", "kdt_std"]
    df = df[cols]
    
    # Convert True and False values
    df['bmlp'] = df['bmlp'].apply(lambda x: r'\ding{51}' if x else r'\ding{55}')
    df['back_y_info'] = df['back_y_info'].apply(lambda x: r'\ding{51}' if x else r'\ding{55}')
    df['back_opemb'] = df['back_opemb'].apply(lambda x: r'\ding{51}' if x else r'\ding{55}')
    df['back_opemb_only'] = df['back_opemb_only'].apply(lambda x: r'\ding{51}' if x else r'\ding{55}')
    
    # Convert to 4 decimal precision
    df['kdt'] = df['kdt'].round(4)
    df['kdt_std'] = df['kdt_std'].round(4)

    # Print LaTeX table
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\begin{tabular}{|c|c|c|c|c|c|c|}")
    print(r"\toprule")
    # print(" & ".join(cols) + r" \\\midrule")
    print(" & ".join([col.replace('_', r'\_') for col in cols]) + r" \\\midrule")

    # print(r"\hline")
    
    for index, row in df.iterrows():
        print(" & ".join(map(str, row)) + r" \\")
        # print(r"\hline")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Desired data from CSV}")
    print(r"\end{table}")

# Call function
fpath = '/home/ya255/projects/flan_hardware/correlation_trainer/correlation_results/fb_abl_study/ENAS_samp_eff.csv'  # Update this path
csv_to_latex(fpath, key=128)
