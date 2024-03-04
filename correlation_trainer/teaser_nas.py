
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


import csv

# help_result = {"asic": [(4.1, 69.8),(5.5, 71.9),(9.1, 73.5)], # constraints 5,7,9
# "fpga": [(4.7,73.4),(5.9,72.4),(7.4,73.5)], # constraints 5,6,7
# "gold_6226": [(7.7,66.6),(11.0,70.6),(13.9,72.1)], # constraints 8,11,14
# "pixel2": [(13, 67.4, 125),(19, 70.6, 125),(34, 73.5, 125)] # constraints 18,21,25
# } 

# brpnas_result = {"asic": [(4.7,71.7),(9.1,73.5),(9.1,73.5)],
# "fpga": [(7.2,73.4),(7.4,73.5),(7.4,73.5)],
# "gold_6226": [(9.5,66.9),(8.7,68.2),(17.0,73.5)],
# "pixel2": [(14, 66.9, 1220), (34, 73.5, 1220), (34, 73.5, 1220)]}

# our_result = {
#     "pixel2": [(14.7,68.71, 4.1),(17.2,70.47,4.1),(22.2,72.08, 4.1),(29.9, 73.2, 4.1),(34,73.5, 4.1)],
# }

# # (21.8, 71.43, 4.1),(25.4,72.25,4.1)
# nas_result = {
#          "Pixel2": {"Ours": results['pixel2'], "HELP": [(13, 67.4, 125),(19, 70.6, 125),(34, 73.5, 125)],"BRPNAS": [(14, 66.9, 1220), (34, 73.5, 1220), (34, 73.5, 1220)]}
#          }
# titanresult = {
#     "Titan RTX 256": {
#         "Ours": results['titan_rtx_256'],
#         "HELP": [(18,69.3), (19,71.6), (25,71.8)]
#     }
# }
# nas_result = {
#          "Pixel2": {"NASFLAT (Ours)": [(14.7,68.71, 4.1),(17.2,70.47,4.1),(22.2,72.08, 4.1),(29.9, 73.2, 4.1),(34,73.5, 4.1)], "HELP": [(13, 67.4, 125),(19, 70.6, 125),(34, 73.5, 125)],"BRPNAS": [(14, 66.9, 1220), (34, 73.5, 1220), (34, 73.5, 1220)]}
#          }

def is_dominated(point, points):
    """Check if a point is dominated by any point in a set of points."""
    x1, y1 = point
    for x2, y2 in points:
        if x2 <= x1 and y2 >= y1:  # Checking for lower latency and higher accuracy
            return True
    return False

def get_pareto_optimal_points(points):
    """Return the Pareto optimal set of points."""
    points = sorted(points, key=lambda x: (x[0], -x[1]))  # Sort by latency, then by accuracy (descending)
    optimal_points = []
    for point in points:
        if not is_dominated(point, optimal_points):  # Compare only with current optimal points
            optimal_points.append(point)
    return optimal_points


def read_csv_and_get_optimal_points(filename):
    """Read the CSV and return a dictionary of optimal points for each device."""
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # print(reader)
        
        # Extract the relevant data
        data = {}
        for row in reader:
            device = row['device']
            accuracy = float(row['accuracy'])
            latency = float(row['latency'])
            if device not in data:
                data[device] = []
            data[device].append((latency, accuracy))
        print(data)
        # Get the Pareto optimal points for each device
        optimal_data = {}
        for device, points in data.items():
            optimal_data[device] = get_pareto_optimal_points(points)

    return optimal_data

# Read the CSV and get the optimal points
results = read_csv_and_get_optimal_points('./HELP/results.csv')
print(results)

nas_result = {
         "Pixel2": {"NASFLAT (Ours)": [(12.2, 65.85599998331705), (14.2, 69.81799994750978), (19.8, 71.92799994995116), (25.6, 72.98666663126627), (34.0, 73.51333332112631)], "HELP": [(13, 67.4, 125),(19, 70.6, 125),(34, 73.5, 125)],"BRPNAS": [(14, 66.9, 1220), (34, 73.5, 1220), (34, 73.5, 1220)]}
         }

corr_result = {
"Ours": [0.959,0.893,0.967,0.857,0.962,0.959,0.961,0.577,0.809,0.871,0.814,0.734],
"HELP": [0.948,0.410,0.604,0.509,0.729,0.746,0.91,0.37,0.793,0.543,0.413,0.799],
"MultiPredict": [0.930,0.820,0.970,0.757,0.947,0.952,0.960,0.45,0.756,0.567,0.434,0.763]
}
datasets = ["ND", "NA", "N1", "N2", "N3", "N4", "FD", "FA", "F1", "F2", "F3", "F4"]


colors = ["#AB3F40", 
            "#3F90BC", 
            "#4D4D4D", 
            "#63BC98"]
from scipy.stats import gmean

fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Create a figure with 2 subplots (side by side)

# Plotting the NAS results
for device, methods in nas_result.items():
    # import pdb; pdb.set_trace()
    for idx, (method, data) in enumerate(methods.items()):
        latencies = [x[0] for x in data]
        accuracies = [x[1] for x in data]
        
        ax[0].scatter(latencies, accuracies, s=150, label=f'{method}', color=colors[idx])
        # ax[0].plot(latencies, accuracies, 'o--', linewidth=4.5)
        # use the colors
        ax[0].plot(latencies, accuracies, 'o--', color=colors[idx], linewidth=4.5)
    
    # ax[0].set_title("Target Device: " + device, fontsize=26)
    ax[0].set_xlabel('Latency (ms)', fontsize=26)
    ax[0].tick_params(axis='both', which='major', labelsize=26)
    ax[0].set_ylabel('Accuracy (%)', fontsize=26)
    # ax[0].legend(loc='upper left', fontsize=22, ncol=1)
    ax[0].legend(loc='lower right', fontsize=24, ncol=1)
    ax[0].grid(True)

# Compute the geometric mean for the datasets starting with 'N' and 'F'
N_means = {k: gmean([v[i] for i, dataset in enumerate(datasets) if dataset.startswith('N')]) for k, v in corr_result.items()}
F_means = {k: gmean([v[i] for i, dataset in enumerate(datasets) if dataset.startswith('F')]) for k, v in corr_result.items()}

bar_width = 0.25
index = np.arange(2)  # Two clusters: NASBench-201 and FBNet

# Get the geometric means for each method
bars1 = [N_means["Ours"], F_means["Ours"]]
bars2 = [N_means["HELP"], F_means["HELP"]]
bars3 = [N_means["MultiPredict"], F_means["MultiPredict"]]

# Create bars
r1 = ax[1].bar(index, bars1, bar_width, color=colors[0], label="Ours")
r2 = ax[1].bar(index + bar_width, bars2, bar_width, color=colors[1], label="HELP")
r3 = ax[1].bar(index + 2*bar_width, bars3, bar_width, color=colors[2], label="MultiPredict")

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax[1].set_xlabel('Datasets', fontsize=26)
ax[1].set_ylabel('Geometric Mean', fontsize=26)
# ax[1].set_title('Geometric Mean of Correlation Coefficients', fontsize=26)
ax[1].set_xticks(index + bar_width)
ax[1].tick_params(axis='both', which='major', labelsize=26)
ax[1].set_ylim([0.5, 1.0])
ax[1].set_xticklabels(['NASBench-201', 'FBNet'])
ax[1].legend(loc='upper right', fontsize=18, ncol=3)
# ax[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig("teaser.pdf")
plt.savefig("teaser.png")





# fig, ax = plt.subplots(1, 2, figsize=(20, 8))  # Create a figure with 2 subplots (side by side)

# # Plotting the NAS results
# for device, methods in nas_result.items():
#     for method, data in methods.items():
#         latencies = [x[0] for x in data]
#         accuracies = [x[1] for x in data]
#         nas_cost = data[0][2]
        
#         ax[0].scatter(latencies, accuracies, s=150, label=f'{method}')
#         ax[0].plot(latencies, accuracies, 'o--', linewidth=4.5)
    
#     ax[0].set_title(device, fontsize=26)
#     ax[0].set_xlabel('Latency (ms)', fontsize=26)
#     # set x and y ticker size
#     ax[0].tick_params(axis='both', which='major', labelsize=26)
#     ax[0].set_ylabel('Accuracy', fontsize=26)
#     ax[0].legend(loc='upper left', fontsize=26)
#     ax[0].grid(True)

# # Plotting the corr_result
# barWidth = 0.2
# r = np.arange(len(datasets))

# colors = ['r', 'g', 'b']

# for idx, (method, values) in enumerate(corr_result.items()):
#     ax[1].hlines(values, xmin=r - barWidth/2 + idx*barWidth, xmax=r + barWidth/2 + idx*barWidth, colors=colors[idx], linewidth=2.5, label=method)

# ax[1].set_xticks(r)
# ax[1].set_xticklabels(datasets, fontsize=26)
# ax[1].tick_params(axis='both', which='major', labelsize=26)
# ax[1].set_title('Correlation Results for Datasets', fontsize=26)
# ax[1].set_xlabel('Datasets', fontsize=26)
# ax[1].set_ylabel('Correlation Value', fontsize=26)
# ax[1].legend(loc='upper left', fontsize=26)
# ax[1].grid(True)

# plt.tight_layout()

# plt.savefig("teaser.pdf")