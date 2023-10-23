import pandas as pd

def format_number(spr, spr_std):
    """Format numbers with subscript and 3 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"{spr:.3f}_{{{spr_std:.3f}}}"

def generate_latex_table(file_path):
    df = pd.read_csv(file_path)
    # only keep rows where 'transfer_sample_size' == 20
    df = df[df['transfer_sample_size'] == 20]
    if 'nb201' in file_path:
        prefix = 'N'
    else:
        prefix = 'F'

    rows = [
        'GIN',
        '(+TransferEmb)',
        '(+Op-HWEmb)',
        '(+Sampler)',
        '(+Supplementary Encoding)'
    ]

    # Create an empty dict to store table values
    table_data = {row: [] for row in rows}

    for index, row in df.iterrows():
        task = f"{prefix}{row['task_index']}"
        value = format_number(row['spr'], row['spr_std'])
        
        if row['representation'] == 'adj_gin':
            if row['hwemb_to_mlp'] and not row['transfer_hwemb'] and row['sampling_metric'] == 'random':
                table_data['GIN'].append((task, value))
            elif row['hwemb_to_mlp'] and row['transfer_hwemb'] and row['sampling_metric'] == 'random':
                table_data['(+TransferEmb)'].append((task, value))
            elif not row['hwemb_to_mlp'] and row['transfer_hwemb'] and row['sampling_metric'] == 'random':
                table_data['(+Op-HWEmb)'].append((task, value))
            elif not row['hwemb_to_mlp'] and row['transfer_hwemb'] and row['sampling_metric'] != 'random':
                table_data['(+Sampler)'].append((task, value))
        else:
            if not row['hwemb_to_mlp'] and row['transfer_hwemb'] and row['sampling_metric'] != 'random':
                table_data['(+Supplementary Encoding)'].append((task, value))

    # Prepare LaTeX table
    tasks = sorted(list(set([item[0] for sublist in table_data.values() for item in sublist])))
    latex_table = " & " + " & ".join(tasks) + r" \\" + "\n" + r"\hline" + "\n"
    
    for row in rows:
        latex_table += row
        for task in tasks:
            value = [item[1] for item in table_data[row] if item[0] == task]
            if value:
                latex_table += " & $" + value[0] + "$"
            else:
                latex_table += " & -"
        latex_table += r" \\" + "\n"

    return latex_table

# Generate LaTeX table for each file
nb201_latex = generate_latex_table('study_6_5_f_900/nb201_samp_eff.csv')
fbnet_latex = generate_latex_table('study_6_5_f_4k/fbnet_samp_eff.csv')

print("NB201 Table:\n", nb201_latex)
print("\nFBNet Table:\n", fbnet_latex)
