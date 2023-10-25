import pandas as pd

def format_number(spr, spr_std):
    """Format numbers with subscript and 3 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"${spr:.3f}_{{{spr_std:.3f}}}$"

def generate_latex_table_from_csv(file_path):
    # Read the CSV file with the specified columns
    df = pd.read_csv(file_path, usecols=["task_index", "num_samples", "spr", "spr_std"])

    # Create a new column to store the formatted values of spr and spr_std
    df['formatted_spr'] = df.apply(lambda row: format_number(row['spr'], row['spr_std']), axis=1)

    # Pivot the DataFrame
    pivoted_df = df.pivot(index="num_samples", columns="task_index", values="formatted_spr")

    # Convert the pivoted DataFrame to LaTeX format
    latex_table = pivoted_df.to_latex(escape=False)
    
    return latex_table

latex_table = generate_latex_table_from_csv('aggr_help_baselines/nb201_samp_eff.csv')
print("HELP")
print(latex_table)
print("MULTIPREDICT")
latex_table = generate_latex_table_from_csv('aggr_multipredict_baseline/nb201_samp_eff.csv')
print(latex_table)