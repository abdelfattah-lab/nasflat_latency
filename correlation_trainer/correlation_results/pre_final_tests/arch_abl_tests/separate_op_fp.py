import pandas as pd

def load_data_for_space(space, folder="seperate_op_fp"):
    """Load CSV data for a given space."""
    filename = f"{folder}/{space}_samp_eff.csv"
    return pd.read_csv(filename)

def format_number(kdt, kdt_std):
    """Format numbers with subscript and 4 decimal precision."""
    return f"${kdt:.4f}_{{{kdt_std:.2f}}}$"

def generate_latex_table(spaces):
    """Generate LaTeX table for given spaces."""
    # DataFrame to hold the final table 
    result = pd.DataFrame()
    for space in spaces:
        try:
            # Load data for the current space
            data = load_data_for_space(space)
            
            # Filter and format data for separate_op_fp=True
            true_data = data[data['separate_op_fp'] == True]
            true_column = true_data.apply(lambda row: format_number(row['kdt'], row['kdt_std']), axis=1)
            
            # Filter and format data for separate_op_fp=False
            false_data = data[data['separate_op_fp'] == False]
            false_column = false_data.apply(lambda row: format_number(row['kdt'], row['kdt_std']), axis=1)
            
            # Load data from the "separate_op_fp_2" folder and format
            pm_data = load_data_for_space(space, folder="separate_op_fp_2")
            pm_column = pm_data.apply(lambda row: format_number(row['kdt'], row['kdt_std']), axis=1)
            
            # Construct multi-index columns
            true_column.index = true_data['key']
            false_column.index = false_data['key']
            pm_column.index = pm_data['key']
            columns = pd.MultiIndex.from_product([[space], ['True', 'True$_{pm}$', 'False']])
            space_df = pd.DataFrame({(space, 'True'): true_column, (space, 'True$_{pm}$'): pm_column, (space, 'False'): false_column})
            
            result = pd.concat([result, space_df], axis=1)
        except Exception as e:
            print(e)
            pass

    # Print the LaTeX table
    print(result.to_latex(escape=False, multirow=True))

# Provide a list of spaces and call the function
spaces = ["NASNet","Amoeba","DARTS","nb201","ENAS_fix-w-d","nb101"]
generate_latex_table(spaces)
