import pandas as pd
import os
import matplotlib.pyplot as plt

# Define function to format the numbers with subscript and 3 decimal precision
def format_number(spr, spr_std):
    """Format numbers with subscript and 3 decimal precision."""
    spr = 0.0 if pd.isnull(spr) else float(spr)
    spr_std = 0.0 if pd.isnull(spr_std) else float(spr_std)
    return f"${spr:.3f}_{{{spr_std:.3f}}}$"

def generate_graph_from_csv(file_path, space):
    df = pd.read_csv(file_path)
    df = df[["task_index", "transfer_sample_size", "sampling_metric", "spr", "spr_std"]]
    
    task_indices = sorted(list(set(df.task_index.tolist())))

    # Create a new figure with three subplots side by side
    fig, axes = plt.subplots(nrows=1, ncols=len(task_indices), figsize=(18, 6))
    
    allowed_metrics = ["random", "params", "zcp", "a2vcatezcp"]

    color_map = {"random": "#AB3F40", "params": "#3F90BC", "zcp": "#4D4D4D", "a2vcatezcp": "#63BC98", "arch2vec": "red"}
    metric_map = {
        "latency": "Latency (Oracle)", 
        "random": "Random", 
        "params": "Params", 
        "a2vcatezcp": "CAZ", 
        "cate": "CATE", 
        "arch2vec": "Arch2Vec", 
        "zcp": "ZCP"
    }

    for idx, task_ix in enumerate(task_indices):
        df_ = df[df["task_index"] == task_ix]
        ax = axes[idx]

        for metric in allowed_metrics:
            subset = df_[df_['sampling_metric'] == metric].sort_values(by='transfer_sample_size')
            ax.plot(subset['transfer_sample_size'], subset['spr_std'], linewidth=3, label=metric_map[metric], markersize=10, marker="o", color=color_map[metric])

        ax.set_title('Device Set {}{}'.format("N" if space=='nb201' else "F", task_ix), fontsize=20)
        # Change font size of axis numbers
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel('Transfer Sample Size', fontsize=20)
        if idx == 0:  # only set ylabel for the first subplot
            ax.set_ylabel('Standard Deviation Of Rank Correlation', fontsize=20)
        # Add lightly shaded grid
        ax.grid(alpha=0.5)

    # Set a central title for the entire figure
    # fig.suptitle('Std Dev For Samplers vs Transfer Sample Size', fontsize=16, y=1.03)

    # Create a shared legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(allowed_metrics), fontsize=20)

    plt.tight_layout()
    lxs = file_path.split("/")[0]
    if not os.path.exists(f"{lxs}_graphs_{space}/"):
        os.makedirs(f"{lxs}_graphs_{space}/")
    plt.savefig(f"{lxs}_graphs_{space}/combined_tasks.png", bbox_inches='tight')
    plt.savefig(f"{lxs}_graphs_{space}/combined_tasks.pdf", bbox_inches='tight')
    # plt.show()  # display the graph
# def generate_graph_from_csv(file_path, space):
#     # Read the CSV file with the specified columns
#     # df = pd.read_csv(file_path, usecols=["task_index", "gnn_type", "spr", "spr_std"])

#     df = pd.read_csv(file_path)
#     # Filter rows with the specified transfer_sample_size
#     # df = df[df["transfer_sample_size"] == 5]
#     df = df[["task_index", "transfer_sample_size", "sampling_metric", "spr", "spr_std"]]
#     # import pdb; pdb.set_trace()
#     for task_ix in set(df.task_index.tolist()):
#         df_ = df[df["task_index"] == task_ix]
                
#         # Create a new figure and axis
#         fig, ax = plt.subplots(figsize=(6,6))

#         # List of unique sampling metrics
#         sampling_metrics = df_['sampling_metric'].unique()
#         allowed_metrics = ["random", "params", "zcp"]
#         # Fix colors for each metric
#         color_map = {
#             "random": "blue", "params": "orange", "zcp": "green"
#         }

#         metric_map = {
#             "latency": "Latency (Oracle)", "random": "Random", "params": "Params", "a2vcatezcp": "CAZ", "cate": "CATE", "arch2vec": "Arch2Vec", "zcp": "ZCP"
#         }

#         for metric in sampling_metrics:
#             if metric not in allowed_metrics:
#                 continue
#             subset = df_[df_['sampling_metric'] == metric].sort_values(by='transfer_sample_size')
#             ax.plot(subset['transfer_sample_size'], subset['spr_std'], label=metric_map[metric], marker="o", color=color_map[metric])

#         ax.set_title('Std Dev For Samplers vs Transfer Sample Size for Device Set {}{}'.format("N" if space=='nb201' else "F", task_ix))
#         ax.set_xlabel('Transfer Sample Size')
#         ax.set_ylabel('SPR Std Dev')
#         ax.legend()
#         plt.tight_layout()
#         lxs = file_path.split("/")[0]
#         if not os.path.exists(f"{lxs}_graphs_{space}/"):
#             os.makedirs(f"{lxs}_graphs_{space}/")
#         plt.savefig(f"{lxs}_graphs_{space}/task_{task_ix}.png")
#         plt.cla()
#         plt.clf()
# For nb201_samp_eff.csv
generate_graph_from_csv('aggr_study_6_3_1_t2/nb201_samp_eff.csv', space="nb201")

# For fbnet_samp_eff.csv
generate_graph_from_csv('aggr_study_6_3_1_t2/fbnet_samp_eff.csv', space="fbnet")
# def generate_graph_from_csv(file_path, space):
#     df = pd.read_csv(file_path)
#     df = df[["task_index", "transfer_sample_size", "sampling_metric", "spr", "spr_std"]]

#     for task_ix in set(df.task_index.tolist()):
#         df_ = df[df["task_index"] == task_ix]
                
#         # Set figure size to be square
#         fig, ax = plt.subplots(figsize=(8,8))

#         # List of unique sampling metrics
#         sampling_metrics = df_['sampling_metric'].unique()
#         metric_map = {
#             "latency": "Latency (Oracle)", "random": "Random", "params": "Params", "a2vcatezcp": "CAZ", "cate": "CATE", "arch2vec": "Arch2Vec", "zcp": "ZCP"
#         }

#         # Width of a bar in the bar chart
#         width = 0.15
#         positions = list(range(len(df_['transfer_sample_size'].unique())))

#         for idx, metric in enumerate(df_['sampling_metric'].unique()):
#             subset = df_[df_['sampling_metric'] == metric].sort_values(by='transfer_sample_size')
#             ax.bar([p + width * idx for p in positions], subset['spr_std'], width=width, label=metric_map[metric])

#         ax.set_title('Std Dev For Samplers vs Transfer Sample Size for Task {}{}'.format("N" if space=='nb201' else "F", task_ix), fontsize=15)
#         ax.set_xlabel('Transfer Sample Size', fontsize=14)
#         ax.set_xticks([p + width * (len(sampling_metrics) / 2) for p in positions])
#         ax.set_xticklabels(df_['transfer_sample_size'].unique(), fontsize=12)
#         ax.set_ylabel('SPR Std Dev', fontsize=14)
#         ax.legend(loc='upper left', fontsize=11)
#         ax.tick_params(axis='both', which='major', labelsize=12)

#         lxs = file_path.split("/")[0]
#         if not os.path.exists(f"{lxs}_graphs_{space}/"):
#             os.makedirs(f"{lxs}_graphs_{space}/")
#         plt.savefig(f"{lxs}_graphs_{space}/task_{task_ix}.png")
#         plt.cla()
