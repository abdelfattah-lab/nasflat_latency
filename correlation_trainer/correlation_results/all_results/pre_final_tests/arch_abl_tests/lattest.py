# # import pandas as pd

# # def format_number(spr, spr_std):
# #     """Format numbers with subscript and 4 decimal precision."""
# #     spr = 0.0 if pd.isnull(spr) else float(spr)
# #     spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    
# #     return f"${spr:.4f}_{{{spr_std:.4f}}}$"

# # def generate_latex_tables_from_file(filename):
# #     """Generate LaTeX tables for the specified file."""
# #     # Load CSV data from the file
# #     data = pd.read_csv(filename)
    
# #     # Replace device None with 'accuracy'
# #     data['device'] = data['device'].fillna('accuracy')
    
# #     # Get unique keys
# #     unique_keys = data['key'].unique()
    
# #     for key in unique_keys:
# #         print(f"Table for Key: {key}\n")
        
# #         # Filter data for the current key
# #         filtered_data = data[data['key'] == key]
        
# #         # Create a pivot table with 'representation' as index, 'device' as columns, and 'spr' and 'spr_std' as values
# #         table = filtered_data.pivot(index='representation', columns='device', values=['spr', 'spr_std'])
        
# #         # Format the values using the 'format_number' function
# #         formatted_table = table.apply(lambda row: format_number(row[('spr', row.name)], row[('spr_std', row.name)]), axis=1)
        
# #         # Print the LaTeX table
# #         print(formatted_table.to_latex(escape=False))
# #         print("\n\n")

# # # Call the function for your file
# # generate_latex_tables_from_file('lattest/nb201_samp_eff.csv')
import pandas as pd

def replace_representation_names(data, old_data=False):
    """Replace representation names based on given rules."""
    if old_data:
        replacements = {
            'adj_gin_cate': 'oFLAN$_{CATE}$',
            'adj_gin_zcp': 'oFLAN$_{ZCP}$',
            'adj_gin': 'oFLAN'
        }
    else:
        replacements = {
            'adj_gin': 'FLAN',
            '_cate': '$_{CATE}$',
            '_arch2vec': '$_{arch2vec}$',
            '_zcp': '$_{ZCP}$',
            "adj_mlp": 'FLANADJ',
            "zcp": "MultiPredict"
        }

    for key, val in replacements.items():
        data['representation'] = data['representation'].str.replace(key, val)
    return data

def process_data(filename, old_data=False):
    """Process data from the given file."""
    data = pd.read_csv(filename)
    
    # devices = ["raspi4", "fpga", "eyeriss", "None"]
    # data = data[data['device'].isin(devices)]
    keys = [4, 8, 24, 64]
    
    data = data[data['key'].isin(keys)]
    if not old_data:
        # Filter based on devices, representations, and keys for the new dataset
        
        representations = ["zcp", "cate", "adj_gin_cate", "adj_gin_zcp"]
        data = data[data['representation'].isin(representations)]
    
    data = replace_representation_names(data, old_data)
    return data


def format_number(spr, spr_std):
    """Format numbers with subscript and 4 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    
    return f"${spr:.3f}_{{{spr_std:.2f}}}$"


def generate_latex_tables():
    """Generate LaTeX tables combining both files."""
    new_data = process_data('lattest/nb201_samp_eff.csv')
    old_data = process_data('oldlattest/nb201_samp_eff.csv', old_data=True)
    
    combined_data_df = pd.concat([new_data, old_data])
    
    formatted_data_list = []
    unique_keys = combined_data_df['key'].unique()
    
    for key in unique_keys:
        filtered_data = combined_data_df[combined_data_df['key'] == key]
        formatted_data = pd.DataFrame(index=filtered_data['representation'].unique())
        
        for _, row in filtered_data.iterrows():
            device = row['device']
            representation = row['representation']
            spr = row['spr']
            spr_std = row['spr_std']
            formatted_data.at[representation, device] = format_number(spr, spr_std)
        
        formatted_data['Key'] = key
        formatted_data_list.append(formatted_data)
    
    result_df = pd.concat(formatted_data_list)
    result_df.set_index('Key', append=True, inplace=True)
    result_df = result_df.swaplevel(0,1)
    
    print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
    print(result_df.to_latex(escape=False))
    print("}\\end{table}")

# Call the function
generate_latex_tables()
# def generate_latex_tables_from_file(filename):
#     """Generate LaTeX tables for the specified file."""
#     # Load CSV data from the file
#     data = pd.read_csv(filename)
    
#     # Replace device None with 'accuracy'
#     # data['device'] = data['device'].fillna('accuracy')
#     # Only get the following devices from 'device' column:
#     devices = ["raspi4", "fpga", "eyeriss", "None"]
#     data = data[data['device'].isin(devices)]
#     # Only get the following representations:
#     representations = ["zcp", "cate", "adj_gin_cate", "adj_gin_zcp"]
#     # Only get the following keys
#     keys = [4, 8, 24, 64]
#     data = data[data['key'].isin(keys)]
#     data = data[data['representation'].isin(representations)]
#     data = replace_representation_names(data)
    
#     # Create an empty DataFrame for the formatted values
#     combined_data = []
    
#     # Get unique keys and representations
#     unique_keys = data['key'].unique()
    
#     for key in unique_keys:
#         # Filter data for the current key
#         filtered_data = data[data['key'] == key]
        
#         # DataFrame for the current key's formatted values
#         formatted_data = pd.DataFrame(index=filtered_data['representation'].unique())
        
#         for _, row in filtered_data.iterrows():
#             device = row['device']
#             representation = row['representation']
#             spr = row['spr']
#             spr_std = row['spr_std']
#             formatted_data.at[representation, device] = format_number(spr, spr_std)
        
#         formatted_data['Key'] = key
#         combined_data.append(formatted_data)

#     combined_data_df = pd.concat(combined_data)
#     combined_data_df.set_index('Key', append=True, inplace=True)
#     combined_data_df = combined_data_df.swaplevel(0,1)
    
#     # Print the LaTeX table
#     print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
#     print(combined_data_df.to_latex(escape=False))
#     print("}\\end{table}")

# # Call the function for your file
# generate_latex_tables_from_file('lattest/nb201_samp_eff.csv')

            # import pandas as pd

            # def replace_representation_names(data):
            #     """Replace representation names based on given rules."""
            #     replacements = {
            #         'adj_gin': 'FLAN',
            #         '_cate': '$_{CATE}$',
            #         '_arch2vec': '$_{arch2vec}$',
            #         '_zcp': '$_{zcp}$',
            #         "adj_mlp": 'FLANADJ',
            #         "zcp": "MultiPredict"
            #     }
            #     for key, val in replacements.items():
            #         data['representation'] = data['representation'].str.replace(key, val)
            #     return data
            # def format_number(spr, spr_std):
            #     """Format numbers with subscript and 4 decimal precision."""
            #     spr = 0.0 if pd.isnull(spr) else float(spr)
            #     spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
                
            #     return f"${spr:.4f}_{{{spr_std:.4f}}}$"

            # def generate_latex_tables_from_file(filename):
            #     """Generate LaTeX tables for the specified file."""
            #     # Load CSV data from the file
            #     data = pd.read_csv(filename)
                
            #     # Replace device None with 'accuracy'
            #     data['device'] = data['device'].fillna('accuracy')
            #     data = replace_representation_names(data)
                
            #     # Get unique keys and representations
            #     unique_keys = data['key'].unique()
                
            #     for key in unique_keys:
            #         # print(f"Table for Key: {key}\n")
                    
            #         # Filter data for the current key
            #         filtered_data = data[data['key'] == key]
                    
            #         # Create an empty DataFrame for the formatted values
            #         formatted_data = pd.DataFrame(index=filtered_data['representation'].unique(), 
            #                                     columns=filtered_data['device'].unique())
                    
            #         for _, row in filtered_data.iterrows():
            #             device = row['device']
            #             representation = row['representation']
            #             spr = row['spr']
            #             spr_std = row['spr_std']
            #             formatted_data.at[representation, device] = format_number(spr, spr_std)
                    
            #         # Print the LaTeX table
            #         print("\\begin{table}[h!]\n\centering\n\\resizebox{\linewidth}{!}{%")
            #         print(formatted_data.to_latex(escape=False))
            #         print("}\\end{table}")
            #         # print("\n\n")

            # # Call the function for your file
            # generate_latex_tables_from_file('lattest/nb201_samp_eff.csv')
# import pandas as pd

# def format_number(spr, spr_std):
#     """Format numbers with subscript and 4 decimal precision."""
#     spr = 0.0 if pd.isnull(spr) else float(spr)
#     spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    
#     return f"${spr:.4f}_{{{spr_std:.4f}}}$"

# def replace_representation_names(data):
#     """Replace representation names based on given rules."""
#     replacements = {
#         'adj_gin': 'FLAN',
#         '_cate': '$_{CATE}$',
#         '_arch2vec': '$_{arch2vec}$',
#         '_zcp': '$_{zcp}$',
#         "adj_mlp": 'FLANADJ',
#         "zcp": "MultiPredict"
#     }
#     for key, val in replacements.items():
#         data['representation'] = data['representation'].str.replace(key, val)
#     return data

# def generate_latex_tables_from_file(filename):
#     """Generate LaTeX tables for the specified file."""
#     # Load CSV data from the file
#     data = pd.read_csv(filename)
    
#     # Replace device None with 'accuracy'
#     data['device'] = data['device'].fillna('accuracy')
    
#     # Replace representation names
#     data = replace_representation_names(data)
    
#     # Get unique keys
#     unique_keys = data['key'].unique()
    
#     all_tables = []
    
#     for key in unique_keys:
#         # Filter data for the current key
#         filtered_data = data[data['key'] == key]
        
#         # Create an empty DataFrame for the formatted values
#         formatted_data = pd.DataFrame(index=filtered_data['representation'].unique())
        
#         for _, row in filtered_data.iterrows():
#             device = row['device']
#             representation = row['representation']
#             spr = row['spr']
#             spr_std = row['spr_std']
#             formatted_data[(key, device)] = format_number(spr, spr_std)  # Multi-level column headers
        
#         all_tables.append(formatted_data)
    
#     # Concatenate tables vertically
#     final_table = pd.concat(all_tables)
    
#     print(final_table.to_latex(escape=False, multirow=True))

# # Call the function for your file
# generate_latex_tables_from_file('lattest/nb201_samp_eff.csv')
