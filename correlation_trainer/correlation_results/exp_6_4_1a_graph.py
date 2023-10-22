import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
def generate_graphs_from_csv(file_path,space):
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Filter out the required columns
    df = df[["task_index", "sample_sizes", "spr", "sampling_metric"]]

    # Get unique task indices
    task_indices = sorted(list(set(df.task_index.tolist())))

    # Get unique sampling metrics
    sampling_metrics = sorted(list(set(df.sampling_metric.tolist())))

    # Set a color map for each sampling metric for better visualization
    colors = plt.cm.viridis(np.linspace(0, 1, len(sampling_metrics)))
    color_map = {metric: colors[i] for i, metric in enumerate(sampling_metrics)}

    # Set the directory path for saving the graphs
    directory_path = "graphs/" + file_path.split("/")[0] + f'{space}_graphs'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Plot graphs for each unique task index
    for task_ix in task_indices:
        df_task = df[df["task_index"] == task_ix]
        
        plt.figure(figsize=(10, 6))

        for metric in sampling_metrics:
            df_metric = df_task[df_task["sampling_metric"] == metric]
            plt.plot(df_metric['sample_sizes'], df_metric['spr'], marker="o", color=color_map[metric], label=metric)
        
        plt.title(f'Task {task_ix}')
        plt.xlabel('Number Of Samples From Each Source Device')
        plt.ylabel('SPR')
        plt.grid(True)
        plt.legend()
        
        # Save the figure
        plt.savefig(f"{directory_path}/task_{task_ix}.png")
        plt.close()

def generate_combined_graph_from_csv(file_path, space):
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Filter out the required columns
    df = df[["task_index", "sample_sizes", "spr", "sampling_metric", "spr_std"]]

    # Get unique task indices
    task_indices = sorted(list(set(df.task_index.tolist())))
    color_map = {
            "random": "#AB3F40", 
            "params": "#3F90BC", 
            "zcp": "#4D4D4D", 
            "a2vcatezcp": "#63BC98",
            # If there are more metrics, you can continue adding colors in a similar manner
        }
    metric_map = {
            "random": "Random",
            "params": "Params",
            "zcp": "ZCP",
            "a2vcatezcp": "CAZ",
    }
    metrics = ["random", "params", "zcp", "a2vcatezcp"]
    # Get unique sampling metrics
    # only select metrics from metrics
    df = df[df["sampling_metric"].isin(metrics)]
    sampling_metrics = sorted(list(set(df.sampling_metric.tolist())))
    # Set a color map for each sampling metric for better visualization
    colors = plt.cm.tab10(np.linspace(0, 1, len(sampling_metrics)))

    # color_map = {metric: colors[i] for i, metric in enumerate(sampling_metrics)}

    # Create a new figure with multiple subplots side by side
    fig, axes = plt.subplots(nrows=1, ncols=len(task_indices), figsize=(18, 6))

    for idx, task_ix in enumerate(task_indices):
        df_task = df[df["task_index"] == task_ix]
        ax = axes[idx]

        for metric in sampling_metrics:
            df_metric = df_task[df_task["sampling_metric"] == metric]
            error = df_metric['spr_std']
            y = df_metric['spr']
            x = df_metric['sample_sizes']
            ax.plot(df_metric['sample_sizes'], df_metric['spr'], linewidth=2, marker="o", color=color_map[metric], label=metric_map[metric])
            ax.fill_between(x, y-error, y+error, color=color_map[metric], alpha=0.1)  # shaded error region

        
        ax.set_title(f'Task {task_ix}', fontsize=20)
        # Set x axis range 
        # ax.set_xlim([0, 200])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlabel('Number Of Samples From Each Source Device', fontsize=20)
        if idx == 0:  # only set ylabel for the first subplot
            ax.set_ylabel('Spearman Rank Correlation', fontsize=20)
        ax.grid(alpha=0.5)

    # Create a shared legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(sampling_metrics), fontsize=20)

    plt.tight_layout()

    # Set the directory path for saving the graphs
    directory_path = "graphs/" + file_path.split("/")[0] + f'{space}_graphs'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Save the figure
    plt.savefig(f"{directory_path}/combined_tasks.png", bbox_inches='tight')
    plt.savefig(f"{directory_path}/combined_tasks.pdf", bbox_inches='tight')
    plt.close()

def generate_2s_combined_graph_from_csv(file_path_fbnet, file_path_nb201):
    # Read the CSVs
    df_fbnet = pd.read_csv(file_path_fbnet)
    df_nb201 = pd.read_csv(file_path_nb201)
    
    # Filter out the required columns
    df_fbnet = df_fbnet[["task_index", "sample_sizes", "spr", "sampling_metric", "spr_std"]]
    # Add "F" to the task indices to distinguish them from the task indices in the other CSV
    df_fbnet["task_index"] = df_fbnet["task_index"].apply(lambda x: "F" + str(x))
    df_nb201 = df_nb201[["task_index", "sample_sizes", "spr", "sampling_metric", "spr_std"]]
    # Add "N" to the task indices to distinguish them from the task indices in the other CSV
    df_nb201["task_index"] = df_nb201["task_index"].apply(lambda x: "N" + str(x))

    # Get the first two unique task indices
    task_indices_fbnet = sorted(list(set(df_fbnet.task_index.tolist())))[:2]
    task_indices_nb201 = sorted(list(set(df_nb201.task_index.tolist())))[:2]

    # Filter data based on these indices
    df_fbnet = df_fbnet[df_fbnet["task_index"].isin(task_indices_fbnet)]
    df_nb201 = df_nb201[df_nb201["task_index"].isin(task_indices_nb201)]

    # Concatenate the DataFrames
    df = pd.concat([df_fbnet, df_nb201])

    # Rest of the code remains largely unchanged
    # Get unique task indices
    task_indices = sorted(list(set(df.task_index.tolist())))
    color_map = {
            "random": "#AB3F40", 
            "params": "#3F90BC", 
            "zcp": "#4D4D4D", 
            "a2vcatezcp": "#63BC98",
            # If there are more metrics, you can continue adding colors in a similar manner
        }
    metric_map = {
            "random": "Random",
            "params": "Params",
            "zcp": "ZCP",
            "a2vcatezcp": "CAZ",
    }
    metrics = ["random", "params", "zcp", "a2vcatezcp"]
    # Get unique sampling metrics
    # only select metrics from metrics
    df = df[df["sampling_metric"].isin(metrics)]
    # Make sure the labels are in the same order as the metrics
    # sampling_metrics = sorted(list(set(df.sampling_metric.tolist())))
    sampling_metrics = metrics
    # Set a color map for each sampling metric for better visualization
    colors = plt.cm.tab10(np.linspace(0, 1, len(sampling_metrics)))

    # color_map = {metric: colors[i] for i, metric in enumerate(sampling_metrics)}

    # Create a new figure with multiple subplots side by side
    fig, axes = plt.subplots(nrows=1, ncols=len(task_indices), figsize=(18, 5))

    for idx, task_ix in enumerate(task_indices):
        df_task = df[df["task_index"] == task_ix]
        ax = axes[idx]

        for metric in sampling_metrics:
            df_metric = df_task[df_task["sampling_metric"] == metric]
            error = df_metric['spr_std']
            y = df_metric['spr']
            x = df_metric['sample_sizes']
            ax.plot(df_metric['sample_sizes'], df_metric['spr'], linewidth=2, marker="o", color=color_map[metric], label=metric_map[metric])
            ax.fill_between(x, y-error, y+error, color=color_map[metric], alpha=0.1)  # shaded error region

        
        ax.set_title(f'Task {task_ix}', fontsize=20)
        # Set x axis range 
        # ax.set_xlim([0, 200])
        ax.tick_params(axis='both', which='major', labelsize=14)
        # Ensure that tick params are only 1 decimal precision for ONLY y axis
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        # On the y axis, only put ticks at increments of 0.1
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        if task_ix.__contains__("F"):
            # set range
            ax.set_ylim([0.5, 0.85])
        # ax.set_xlabel('Number Of Samples From Each Source Device', fontsize=20)
        if idx == 0:  # only set ylabel for the first subplot
            ax.set_ylabel('Spearman Rank Correlation', fontsize=20)
        ax.grid(alpha=0.5)

    # Create a shared legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(sampling_metrics), fontsize=20)
    fig.text(0.5, -0.01, 'Number Of Samples From Each Source Device', ha='center', va='center', fontsize=20)

    plt.tight_layout()

    # Instead of the original file_path.split("/")[0] use a generic name for directory_path
    directory_path = 'graphs/aggr_study_6_4_1a/combined_graphs/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Save the figure
    plt.savefig(f"{directory_path}/combined_tasks.png", bbox_inches='tight')
    plt.savefig(f"{directory_path}/combined_tasks.pdf", bbox_inches='tight')
    plt.close()

# Generate graphs for the provided CSV
generate_graphs_from_csv('aggr_study_6_4_1a/fbnet_samp_eff.csv', space='fbnet')
generate_graphs_from_csv('aggr_study_6_4_1a/nb201_samp_eff.csv', space='nb201')

generate_combined_graph_from_csv('aggr_study_6_4_1a/fbnet_samp_eff.csv', space='fbnet')
generate_combined_graph_from_csv('aggr_study_6_4_1a/nb201_samp_eff.csv', space='nb201')

generate_2s_combined_graph_from_csv('aggr_study_6_4_1a/fbnet_samp_eff.csv', 'aggr_study_6_4_1a/nb201_samp_eff.csv')
# generate_2s_combined_graph_from_csv('aggr_study_6_4_1a/nb201_samp_eff.csv', space='nb201')
