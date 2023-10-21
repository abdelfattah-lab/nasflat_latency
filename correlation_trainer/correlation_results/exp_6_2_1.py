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
    return f"${spr:.3f}_{{{spr_std:.3f}}}$"

def generate_latex_table_from_csv(file_path):
    # Read the CSV file with the specified columns
    df = pd.read_csv(file_path)
    df = df[df["transfer_sample_size"] == 20]
    # import pdb; pdb.set_trace()
    # df = pd.read_csv(file_path, usecols=["task_index", "hwemb_to_mlp", "transfer_hwemb", "spr", "spr_std"])
    df = df[["task_index", "hwemb_to_mlp", "transfer_hwemb", "spr", "spr_std"]]
    # only use transfer_sample_size as 20
    # Apply the replace_with_ding function to the hwemb_to_mlp and transfer_hwemb columns
    # import pdb; pdb.set_trace()
    df["hwemb_to_mlp"] = df["hwemb_to_mlp"].apply(replace_with_ding)
    df["transfer_hwemb"] = df["transfer_hwemb"].apply(replace_with_ding)
    
    # Format the spr and spr_std columns
    df["formatted_spr"] = df.apply(lambda row: format_number(row["spr"], row["spr_std"]), axis=1)
    df = df[["task_index", "hwemb_to_mlp", "transfer_hwemb", "formatted_spr"]]
    # Drop index
    df = df.set_index("task_index")
    # replace task_index with Task
    df.index.name = "Task"
    # Sort by task_index
    df = df.sort_index()
    # Replace hwemb_to_mlp and transfer_hwemb with corresponding dict values from name_dict
    name_dict = {
        "hwemb_to_mlp": "HWEmb$\longrightarrow$MLP",
        "transfer_hwemb": "Transfer HWEmb",
        "formatted_spr": "SPR",
    }
    df = df.rename(columns=name_dict)
    # Generate the LaTeX table
    # latex_table = df.pivot_table(index="task_index", 
    #                          columns=["hwemb_to_mlp", "transfer_hwemb"], 
    #                          values="formatted_spr", 
    #                          aggfunc="first").T.to_latex(escape=False)
    latex_table = df.to_latex(escape=False)
    # # Generate the LaTeX table
    # latex_table = df.pivot_table(index="task_index", 
    #                          columns=["hwemb_to_mlp", "transfer_hwemb"], 
    #                          values="formatted_spr", 
    #                          aggfunc="first").T.to_latex(escape=False)

    return latex_table

# For nb201_samp_eff.csv
latex_nb201 = generate_latex_table_from_csv('aggr_arch_abl/nb201_samp_eff.csv')
print(latex_nb201)

# For fbnet_samp_eff.csv
latex_fbnet = generate_latex_table_from_csv('aggr_arch_abl/fbnet_samp_eff.csv')
print(latex_fbnet)
