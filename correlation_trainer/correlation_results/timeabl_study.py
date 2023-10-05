import pandas as pd

spaces = ['PNAS', 'nb101', 'nb201', 'ENAS']

# Take key_ as a parsearg in python using argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--key', type=int, default=64)
args = parser.parse_args()
key_ = args.key


def read_and_process_csv(space):
    # Load CSV
    df = pd.read_csv(f'./timeabl/{space}_samp_eff.csv')
    
    # Filter rows where 'key' == 64
    df = df[df['key'] == key_]
    
    # Filter relevant columns
    df = df[['space', 'timesteps', 'kdt', 'kdt_std']]
    
    # Round values to 4 decimal precision
    df['kdt'] = df['kdt'].round(4)
    df['kdt_std'] = df['kdt_std'].round(4)
    
    # Create consolidated column
    df['value'] = df['kdt'].astype(str) + "_{" + df['kdt_std'].astype(str) + "}"
    
    return df.pivot(index='space', columns='timesteps', values='value')

dfs = [read_and_process_csv(space) for space in spaces]
final_df = pd.concat(dfs)

def generate_latex_table(df):
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\begin{tabular}{" + "|c" + "|c" * len(df.columns) + "|}")
    print(r"\hline")
    print("Space & " + " & ".join(df.columns.astype(str)) + r" \\")
    print(r"\hline")
    for idx, row in df.iterrows():
        print(idx + " & " + " & ".join([str(xa) for xa in row]) + r" \\")
        print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Consolidated data from CSVs}")
    print(r"\end{table}")

generate_latex_table(final_df)
import pandas as pd
import matplotlib.pyplot as plt

spaces = ['PNAS', 'nb101', 'nb201', 'ENAS']

def plot_data_from_csv(space):
    # Load CSV
    df = pd.read_csv(f'./timeabl/{space}_samp_eff.csv')
    
    # Filter rows where 'key' == 64
    df = df[df['key'] == key_]
    
    # Filter relevant columns
    df = df[['timesteps', 'kdt', 'kdt_std']]
    
    # Sort dataframe by timesteps
    df = df.sort_values(by='timesteps')

    kdt_min, kdt_max = df['kdt'].min(), df['kdt'].max()
    df['normalized_kdt'] = (df['kdt'] - kdt_min) / (kdt_max - kdt_min)
    df['normalized_kdt_std'] = df['kdt_std'] / (kdt_max - kdt_min)

    # Plot data
    plt.plot(df['timesteps'], df['normalized_kdt'], label=space, marker='o')
    plt.fill_between(df['timesteps'], 
                     df['normalized_kdt'] - df['normalized_kdt_std'], 
                     df['normalized_kdt'] + df['normalized_kdt_std'], alpha=0.2)
    plt.xticks(df['timesteps'])

for space in spaces:
    plot_data_from_csv(space)

plt.xlabel('Timesteps')
plt.ylabel('0-1 Normalzied KDT')
plt.legend()
# plt.title('0-1 Normalized KDT vs Timesteps')
# include key_ in title
plt.title(f'0-1 Normalized KDT vs Timesteps (samples={key_})')
# Show every tick

plt.grid(True)
# plt.show()
plt.savefig('./time_abl_plot%s.png' % (str(key_)))
plt.savefig('./time_abl_plot%s.pdf' % (str(key_)))
