
# import os
# import pandas as pd

# # List all the CSV files in the directory
# csv_files = [f for f in os.listdir('./bmlpally/') if f.endswith('_samp_eff.csv')]

# latex_tables = []

# for csv_file in csv_files:
#     # Load CSV using pandas
#     df = pd.read_csv(os.path.join('./bmlpally/', csv_file))
    
#     # Pivot the data to get the desired table format
#     table_df = df.pivot_table(index='bmlp_ally', columns='key', values='kdt', aggfunc='mean')
    
#     # Convert the DataFrame to LaTeX table
#     latex_table = table_df.to_latex()
    
#     # Append to the list of tables
#     latex_tables.append(latex_table)

# # Print each table
# for table in latex_tables:
#     print(table)
#     print("\n\\hline\n")  # Separator between tables


import os
import pandas as pd

# List all the CSV files in the directory
csv_files = [f for f in os.listdir('./bmlpally/') if f.endswith('_samp_eff.csv')]

# Create an empty DataFrame to hold all the data
all_data = pd.DataFrame()

for csv_file in csv_files:
    # Load CSV using pandas
    df = pd.read_csv(os.path.join('./bmlpally/', csv_file))
    
    df['kdt'] = df['kdt'].round(3)
    df['kdt_std'] = df['kdt_std'].round(3)
    
    # Append the DataFrame to the main DataFrame
    all_data = all_data.append(df)

# Pivot the data to get the desired table format
# Use both 'space' and 'bmlp_ally' as columns to create a multi-index
table_df = all_data.pivot_table(index='key', columns=['space', 'bmlp_ally'], values='kdt', aggfunc='mean')

# Convert the DataFrame to LaTeX table
latex_table = table_df.to_latex()

print(latex_table)
