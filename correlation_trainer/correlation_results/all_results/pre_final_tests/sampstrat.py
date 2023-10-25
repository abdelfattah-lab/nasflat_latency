
# import pandas as pd

# # Define function to format numbers with subscript and 4 decimal precision
# def format_number(spr, spr_std):
#     """Format numbers with subscript and 4 decimal precision."""
#     spr = 0.0 if pd.isnull(spr) else float(spr)
#     spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
#     return f"${spr:.4f}_{{{spr_std:.4f}}}$"

# def generate_latex_tables_from_file(filename, help_filename):
#     # Load CSV data from the file with only the desired columns
#     data = pd.read_csv(filename, usecols=['sampling_metric', 'representation', 'spr', 'spr_std', 'target_device', 'transfer_sample_size'])

#     # Load HELP data
#     help_data = pd.read_csv(help_filename, usecols=['num_samples', 'transfer_device', 'spr', 'spr_std'])
#     help_data.rename(columns={'num_samples': 'transfer_sample_size', 'transfer_device': 'target_device'}, inplace=True)
#     help_data['sampling_metric'] = 'HELP'
    
#     # Merge both datasets
#     combined_data = pd.concat([data, help_data], ignore_index=True)

#     # Sort combined_data for interleaving 
#     combined_data.sort_values(by=['transfer_sample_size', 'sampling_metric'], inplace=True)

#     # Format 'spr' and 'spr_std'
#     combined_data['spr_std_formatted'] = combined_data.apply(lambda x: format_number(x['spr'], x['spr_std']), axis=1)

#     # List to store pivot tables for each group
#     pivot_tables = []

#     # Group by 'transfer_sample_size'
#     grouped = combined_data.groupby('transfer_sample_size')

#     for transfer_sample_size, group in grouped:
#         # Calculate mean for each sampling_metric
#         mean_spr = group.groupby('sampling_metric')['spr'].mean()

#         # Sort group by mean 'spr' for interleaving
#         group = group.set_index('sampling_metric')
#         group['Mean'] = mean_spr
#         group = group.reset_index()
#         group = group.sort_values(by=['Mean', 'sampling_metric'])

#         import pdb; pdb.set_trace()
#         # Pivot table to get desired format
#         pivot_table_A = group.pivot(index='sampling_metric',
#                                   columns='target_device',
#                                   values='spr_std_formatted')

#         pivot_table['Mean'] = mean_spr.apply(lambda x: f"${x:.4f}$").reindex(pivot_table.index)
#         pivot_table['Transfer Sample'] = transfer_sample_size
#         pivot_tables.append(pivot_table)

#     # Concatenate pivot tables
#     combined_table = pd.concat(pivot_tables)
#     combined_table.set_index('Transfer Sample', append=True, inplace=True)
#     combined_table = combined_table.swaplevel(0,1)

#     # Print the combined LaTeX table
#     print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
#     print(combined_table.to_latex(escape=False))
#     print("}\\end{table}")

# # Call the function for your file
# generate_latex_tables_from_file('sampstrat/nb201_samp_eff.csv', 'help_test/help_samp_eff.csv')


import pandas as pd

# Define function to format numbers with subscript and 4 decimal precision
def format_number(spr, spr_std):
    """Format numbers with subscript and 4 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"${spr:.2f}_{{{spr_std:.4f}}}$"

def generate_latex_tables_from_file(filename, help_filename):
    # Load CSV data from the file with only the desired columns
    data = pd.read_csv(filename, usecols=['sampling_metric', 'representation', 'spr', 'spr_std', 'target_device', 'transfer_sample_size'])
    # data = data[data['representation'] != 'adj_gin_zcp']
    data = data[data['representation'] != 'adj_mlp_zcp']
    # Load HELP data
    help_data = pd.read_csv(help_filename, usecols=['num_samples', 'transfer_device', 'spr', 'spr_std'])
    help_data.rename(columns={'num_samples': 'transfer_sample_size', 'transfer_device': 'target_device'}, inplace=True)
    help_data['sampling_metric'] = 'HELP'
    
    # Merge both datasets
    combined_data = pd.concat([data, help_data], ignore_index=True)
    
    # Format 'spr' and 'spr_std'
    combined_data['spr_std_formatted'] = combined_data.apply(lambda x: format_number(x['spr'], x['spr_std']), axis=1)

    # List to store pivot tables for each group
    pivot_tables = []

    # Group by 'transfer_sample_size'
    grouped = combined_data.groupby('transfer_sample_size')

    for transfer_sample_size, group in grouped:
        # Calculate mean for each sampling_metric
        mean_spr = group.groupby('sampling_metric')['spr'].mean()
        
        # Sort group by mean 'spr'
        group = group.set_index('sampling_metric')
        group['Mean'] = mean_spr
        group = group.reset_index()
        group = group.sort_values(by='Mean')

        # Pivot table to get desired format
        pivot_table = group.pivot(index='sampling_metric', 
                                  columns='target_device', 
                                  values='spr_std_formatted')
        
        pivot_table['Mean'] = mean_spr.apply(lambda x: f"${x:.4f}$").reindex(pivot_table.index)
        pivot_table.sort_values(by=['Mean'], inplace=True)
        pivot_table['Transfer Sample'] = transfer_sample_size
        pivot_tables.append(pivot_table)

    # Concatenate pivot tables
    combined_table = pd.concat(pivot_tables)
    combined_table.set_index('Transfer Sample', append=True, inplace=True)
    combined_table = combined_table.swaplevel(0,1)
    
    # Print the combined LaTeX table
    print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
    print(combined_table.to_latex(escape=False))
    print("}\\end{table}")

# Call the function for your file
generate_latex_tables_from_file('sampstrat/nb201_samp_eff.csv', 'help_test/help_samp_eff.csv')



# def generate_latex_tables_from_file(filename):
#     # Load CSV data from the file with only the desired columns
#     data = pd.read_csv(filename, usecols=['sampling_metric', 'spr', 'spr_std', 'target_device', 'transfer_sample_size'])
    
#     # Format 'spr' and 'spr_std'
#     data['spr_std_formatted'] = data.apply(lambda x: format_number(x['spr'], x['spr_std']), axis=1)

#     # List to store pivot tables for each group
#     pivot_tables = []

#     # Group by 'transfer_sample_size'
#     grouped = data.groupby('transfer_sample_size')


#     for transfer_sample_size, group in grouped:
#         # Pivot table to get desired format
#         pivot_table = group.pivot(index='sampling_metric', 
#                                   columns='target_device', 
#                                   values='spr_std_formatted')
        
#         # Calculate mean for each row
#         pivot_table['Mean'] = group.groupby('sampling_metric')['spr'].mean()
#         pivot_table['Mean'] = pivot_table['Mean'].apply(lambda x: f"${x:.3f}$")

#         pivot_table.sort_values(by=['Mean'], inplace=True)
#         pivot_table['Transfer Sample'] = transfer_sample_size
#         pivot_tables.append(pivot_table)

#     # Concatenate pivot tables
#     combined_table = pd.concat(pivot_tables)
#     combined_table.set_index('Transfer Sample', append=True, inplace=True)
#     combined_table = combined_table.swaplevel(0,1)
    
#     # Print the combined LaTeX table
#     print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
#     print(combined_table.to_latex(escape=False))
#     print("}\\end{table}")

# # Call the function for your file
# generate_latex_tables_from_file('sampstrat/nb201_samp_eff.csv')
