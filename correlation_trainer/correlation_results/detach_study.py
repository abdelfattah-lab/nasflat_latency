import pandas as pd

def generate_latex_table(space, key_):
    # Read CSV
    df = pd.read_csv(f'./detach_abl/{space}_samp_eff.csv')
    
    # Filter rows based on the specified key
    df = df[df['key'] == key_]
    df = df.sort_values(by='kdt')
    # Replace True and False values
    df['back_y_info'] = df['back_y_info'].replace({True: r'\ding{51}', False: r'\ding{55}'})
    df['back_opemb'] = df['back_opemb'].replace({True: r'\ding{51}', False: r'\ding{55}'})
    
    # Map the values in the detach_mode column
    detach_map = {"detach_all": "all", "detach_none": "none", "default": "def"}
    df['detach_mode'] = df['detach_mode'].map(detach_map)
    
    # Round the values
    df['kdt'] = df['kdt'].round(4)
    df['kdt_std'] = df['kdt_std'].round(4)
    
    # Select relevant columns
    df = df[['back_y_info', 'back_opemb', 'detach_mode', 'kdt', 'kdt_std']]
    
    # Generate LaTeX table
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\toprule")
    print(" & ".join(df.columns) + r" \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        print(" & ".join(row.astype(str)) + r" \\")
    print(r"\bottomrule")
    print(r"\end{table}")

# Sample use:
generate_latex_table('nb101', 64)
generate_latex_table('nb101', 128)
# generate_latex_table('ENAS', 64)
# generate_latex_table('ENAS', 128)
# generate_latex_table('PNAS', 64)
# generate_latex_table('PNAS', 128)
# generate_latex_table('nb201', 64)
# generate_latex_table('nb201', 128)
