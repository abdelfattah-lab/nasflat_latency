import pandas as pd

# Define function to replace True and False with \ding for tick and cross
def replace_with_ding(value):
    if value:
        return "\\ding{51}"  # Tick
    else:
        return "\\ding{55}"  # Cross

def format_number(spr, spr_std):
    """Format numbers with subscript and 4 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"${spr:.2f}_{{{spr_std:.4f}}}$"

def generate_latex_tables_from_file(filename):
    # Load CSV data from the file
    data = pd.read_csv(filename)
    
    # Replace values in 'hwemb_to_mlp' and 'transfer_hwemb'
    data['hwemb_to_mlp'] = data['hwemb_to_mlp'].apply(replace_with_ding)
    data['transfer_hwemb'] = data['transfer_hwemb'].apply(replace_with_ding)
    
    # Format 'spr' and 'spr_std'
    data['spr_std_formatted'] = data.apply(lambda x: format_number(x['spr'], x['spr_std']), axis=1)

    # List to store pivot tables for each group
    pivot_tables = []

    # Group by 'transfer_sample_size'
    grouped = data.groupby('transfer_sample_size')

    for transfer_sample_size, group in grouped:
        # Pivot table to get desired format
        pivot_table = group.pivot(index=['hwemb_to_mlp', 'transfer_hwemb'], 
                                  columns='target_device', 
                                  values='spr_std_formatted')
        
        # Calculate mean for each row
        pivot_table['Mean'] = group.groupby(['hwemb_to_mlp', 'transfer_hwemb'])['spr'].mean()
        pivot_table['Mean'] = pivot_table['Mean'].apply(lambda x: f"${x:.3f}$")
        pivot_table['Transfer Sample'] = transfer_sample_size
        # Sort pivot table by 'Mean'
        # pivot_table.sort_values(by=['Mean'], inplace=True)
        pivot_tables.append(pivot_table)

    # Concatenate pivot tables
    combined_table = pd.concat(pivot_tables)
    combined_table.set_index('Transfer Sample', append=True, inplace=True)
    combined_table = combined_table.swaplevel(0,2)
    
    # Print the combined LaTeX table
    print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
    print(combined_table.to_latex(escape=False))
    print("}\\end{table}")

# Call the function for your file
generate_latex_tables_from_file('tfhwemb/nb201_samp_eff.csv')
