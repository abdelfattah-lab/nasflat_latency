import pandas as pd

# Define function to format the numbers with subscript and 3 decimal precision
def format_number(spr, spr_std):
    """Format numbers with subscript and 3 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"${spr:.3f}_{{{spr_std:.3f}}}$"

def generate_latex_table_from_csv(file_path):
    # Read the CSV file with the specified columns
    # df = pd.read_csv(file_path, usecols=["task_index", "gnn_type", "spr", "spr_std"])

    df = pd.read_csv(file_path)
    # Filter rows with the specified transfer_sample_size
    df = df[df["transfer_sample_size"] == 5]
    df = df[["task_index", "sampling_metric", "spr", "spr_std"]]
    # Format the spr and spr_std columns
    df["formatted_spr"] = df.apply(lambda row: format_number(row["spr"], row["spr_std"]), axis=1)
    df = df[["task_index", "sampling_metric", "formatted_spr"]]

    # Set the task_index column as the index and rename it to "Task"
    df = df.set_index("task_index")
    df.index.name = "Task"

    # Sort by task_index
    df = df.sort_index()

    # Rename the columns as per requirements
    name_dict = {
        "sampling_metric": "Sampler",
        "formatted_spr": "SPR",
    }
    df = df.rename(columns=name_dict)
    # Rename the following strings in the whole df
    # df = df.replace("ensemble", "GCN+GAT")
    # df = df.replace("dense", "GCN")
    # df = df.replace("gat", "GAT")

    df = df.pivot_table(index="Task", 
                             columns=["Sampler"], 
                             values="SPR", 
                             aggfunc="first")
    latex_table = df.T.to_latex(escape=False, multirow=True)
    return latex_table

# # For nb201_samp_eff.csv
# latex_nb201 = generate_latex_table_from_csv('aggr_study_6_3_1_t2/nb201_samp_eff.csv')
# print(latex_nb201)

# # For fbnet_samp_eff.csv
# latex_fbnet = generate_latex_table_from_csv('aggr_study_6_3_1_t2/fbnet_samp_eff.csv')
# print(latex_fbnet)

# # For fbnet_samp_eff.csv
# latex_fbnet = generate_latex_table_from_csv('aggr_study_6_3_1_t2_f0/fbnet_samp_eff.csv')
# print(latex_fbnet)


# For nb201_samp_eff.csv
latex_nb201 = generate_latex_table_from_csv('aggr_study_6_3_0/nb201_samp_eff.csv')
print(latex_nb201)

# For fbnet_samp_eff.csv
latex_fbnet = generate_latex_table_from_csv('aggr_study_6_3_0/fbnet_samp_eff.csv')
print(latex_fbnet)

# For fbnet_samp_eff.csv
latex_fbnet = generate_latex_table_from_csv('aggr_study_6_3_1_t2_f0/fbnet_samp_eff.csv')
print(latex_fbnet)
