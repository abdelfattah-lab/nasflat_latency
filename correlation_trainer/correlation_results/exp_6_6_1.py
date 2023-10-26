import pandas as pd

def format_number(spr, spr_std):
    """Format numbers with subscript and 3 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"${spr:.3f}_{{{spr_std:.3f}}}$"

def generate_latex_tables(filepath):
    # Read the csv file
    df = pd.read_csv(filepath)
    
    # Filter based on transfer_sample_size == 10
    df = df[df['transfer_sample_size'] == 10]
    
    # Columns of interest
    columns = ['task_index', 'transfer_hwemb', 'hwemb_to_mlp', 'sampling_method', 'sampling_metric', 'representation', 'spr', 'spr_std']
    df = df[columns]
    
    # Remove underscores
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('_', ' ')
        
    # Replace representations
    repr_dict = {
        "adj gin": "GIN",
        " cate": "$_{CATE}$",
        " arch2vec": "$_{Arch2Vec}$",
        " zcp": "$_{ZCP}$",
        " a2vcatezcp": "$_{CAZ}$",
    }
    
    df['representation'] = df['representation'].replace(repr_dict)
    
    # Format spr and spr_std
    df['spr'] = df.apply(lambda row: format_number(row['spr'], row['spr_std']), axis=1)
    df.drop(columns='spr_std', inplace=True) # drop spr_std after formatting
    
    # Print LaTeX tables for each task_index
    for task, sub_df in df.groupby('task_index'):
        print(f"Table for task_index: {task}\n")
        print(sub_df.drop(columns='task_index').to_latex(index=False, escape=False))
        print("\n\n")

# Example usage
generate_latex_tables('/home/ya255/projects/flan_hardware/correlation_trainer/correlation_results/aggr_study_6_6_1/fbnet_samp_eff.csv')

generate_latex_tables('/home/ya255/projects/flan_hardware/correlation_trainer/correlation_results/aggr_study_6_6_1/nb201_samp_eff.csv')
