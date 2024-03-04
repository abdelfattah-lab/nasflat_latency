import pandas as pd
import matplotlib.pyplot as plt

help_results = {"eyeriss": [(4.1, 69.8),(5.5, 71.9),(9.1, 73.5)], # constraints 5,7,9
"fpga": [(4.7,71.8),(5.9,72.4),(7.4,73.5)], # constraints 5,6,7
"gold_6226": [(7.7,66.6),(11.0,70.6),(13.9,72.1)], # constraints 8,11,14
"pixel2": [(13, 67.4),(19, 70.6),(34, 73.5)], # constraints 18,21,25
"titan_rtx_256": [(18,69.3),(19,71.6),(25,71.8)]
} 


brpnas_results = {"eyeriss": [(4.7,71.7),(9.1,73.5),(9.1,73.5)],
"fpga": [(7.2,73.4),(7.4,73.5),(7.4,73.5)],
"gold_6226": [(8.7,68.2),(9.5,66.9),(17.0,73.5)],
"pixel2": [(14, 66.9), (34, 73.5), (34, 73.5)],
"titan_rtx_256": [(19,71.5),(21,67.0),(23,70.7)]}

# layerwise_results = {""}
# "titan_rtx_256"
import pickle
# with open(f'true_latacc_{self.nas_target_device}.pkl', 'wb') as f:
#     pickle.dump((yq_gt, self.data.arch_candidates['true_acc']), f)
# read the pickle files for target_device_list = ['eyeriss', 'fpga', 'gold_6226', 'pixel2', 'titan_rtx_256'] into a dict
trueacclat = {}
for dev in ['eyeriss', 'fpga', 'gold_6226', 'pixel2', 'titan_rtx_256']:
    # read from pickle file
    with open(f'true_latacc_{dev}.pkl', 'rb') as f:
        yq_gt, true_acc = pickle.load(f)
    trueacclat[dev] = (yq_gt.flatten().tolist(), [float(x) for x in true_acc])

# import pdb; pdb.set_trace()
import pandas as pd
import matplotlib.pyplot as plt

def is_dominated(point, points):
    """Check if a point is dominated by any point in a set of points."""
    x1, y1 = point
    for x2, y2 in points:
        if x2 <= x1 and y2 >= y1:
            return True
    return False

def get_pareto_optimal_points(points):
    """Return the Pareto optimal set of points."""
    points = sorted(points, key=lambda x: (x[0], -x[1]))
    optimal_points = []
    for point in points:
        if not is_dominated(point, optimal_points):
            optimal_points.append(point)
    return optimal_points

# Read the CSV
data = pd.read_csv('results2.csv')

# Get the unique devices
devices = data['device'].unique()
# devices = ['gold_6226', 'eye']
devnamemap = {'eyeriss': "Eyeriss (ASIC)", 'fpga': "FPGA", 'gold_6226': "Gold 6226 (CPU)", 'pixel2': "Pixel 2 (mCPU)", 'titan_rtx_256': "Titan RTX (GPU)"}
# Create subplots
fig, axes = plt.subplots(1, len(devices), figsize=(15, 3))
markers = ['*', 'o', 's', "P", "D", "8", "+", "H", '2', '1']  # star, circle, box, triangle respectively

for i, device in enumerate(devices):
    if device in trueacclat:
        true_lat, true_acc = trueacclat[device]
        axes[i].scatter(true_lat, true_acc, color='grey', s=20, alpha=0.2)
    
    device_data = data[data['device'] == device]
    # Only use 10,15,20 samples
    if device == 'pixel2':
        device_data = device_data[device_data['nsamp'].isin([20])]
    elif device == 'titan_rtx_256':
        device_data = device_data[device_data['nsamp'].isin([20])]
    elif device == 'gold_6226':
        device_data = device_data[device_data['nsamp'].isin([20])]
    elif device == 'fpga':
        device_data = device_data[device_data['nsamp'].isin([20])]
    elif device == 'eyeriss':
        device_data = device_data[device_data['nsamp'].isin([20])]
        
    for idx, nsamp in enumerate(device_data['nsamp'].unique()):
        nsamp_data = device_data[device_data['nsamp'] == nsamp]
        optimal_points = get_pareto_optimal_points(list(zip(nsamp_data['latency'], nsamp_data['accuracy'])))
        if device == 'pixel2':
            print(optimal_points)
        if optimal_points:
            x, y = zip(*optimal_points)
            axes[i].plot(x, y, label=f'NASFLAT (S: {nsamp})', linestyle="--",linewidth=1, markersize=10, marker=markers[idx])
    
    # Plot additional data from help_results and brpnas_results
    if device in help_results:
        x, y = zip(*help_results[device])
        axes[i].plot(x, y, label='HELP (S: 20)', linewidth=1, linestyle="--",markersize=10, marker=markers[-2])
    if device in brpnas_results:
        x, y = zip(*brpnas_results[device]) 
        axes[i].plot(x, y, label='BRPNAS (S: 900)', linewidth=1, linestyle="--",markersize=10, marker=markers[-1])
    majfonts = 16
    axes[i].set_title(devnamemap[device], fontsize=majfonts)
    axes[i].set_ylim(66, 74)
    # axes[i].set_xlabel('Latency (ms)')
    if device == 'fpga':
        axes[i].set_xlim(4, 8)
        axes[i].set_ylim(71.5, 73.75)
        axes[i].set_yticks([72, 73])  # setting integer y-ticks manually
        # Only use integers for y axis tickers
        # import MaxNLocator
    from matplotlib.ticker import MaxNLocator
    axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
    if device == 'pixel2':
        axes[i].set_ylim(65, 74)
        axes[i].set_xlim(10, 35)
        axes[i].set_yticks([66, 70, 74])  # setting integer y-ticks manually

    elif device == 'titan_rtx_256':
        axes[i].set_ylim(66, 72.5)
        axes[i].set_xlim(12, 30)
        axes[i].set_yticks([66, 68, 70, 72])  # setting integer y-ticks manually

    elif device == 'gold_6226':
        axes[i].set_ylim(62, 74)
        axes[i].set_xlim(5, 18)
        axes[i].set_yticks([64, 68, 72])  # setting integer y-ticks manually

    elif device == 'eyeriss':
        axes[i].set_ylim(68, 74)
        axes[i].set_xlim(3, 10)
        axes[i].set_yticks([68, 70, 72, 74])  # setting integer y-ticks manually

    if i == 0:
        axes[i].set_ylabel('Accuracy (%)', fontsize=majfonts)
    # axes[i].set_yscale('log')
    # Make ticker size smaller
    axes[i].tick_params(axis='both', which='major', labelsize=majfonts-4)
    # axes[i].yaxis.set_ticks([66,68,70,72,74])
    # Only display accuracy ticker on leftmost 
    # if i != 0:
    #     axes[i].set_yticklabels([])

# Create a common legend
handles, labels = axes[0].get_legend_handles_labels()
# Add a single common xlabel
fig.text(0.5, 0.04, 'Latency (ms)', ha='center', va='center', fontsize=majfonts)
fig.legend(handles, labels, loc='upper center', ncol=9, fontsize=majfonts)

plt.tight_layout()
plt.subplots_adjust(top=0.72, bottom=0.16, wspace=0.17)  # Adjust layout to make space for the legend

plt.savefig('pareto_comparisons.pdf')
plt.show()
    # if device == 'pixel2':
    #     axes[i].set_ylim(65, 74)
    # elif device == 'titan_rtx_256':
    #     axes[i].set_ylim(66, 72.5)
    # elif device == 'gold_6226':
    #     axes[i].set_ylim(62, 74)
    # elif device == 'fpga':
    #     axes[i].set_ylim(71, 74)
    # elif device == 'eyeriss':
        # axes[i].set_ylim(68, 74)