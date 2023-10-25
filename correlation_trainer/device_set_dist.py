import sys, os, random
sys.path.append("..")
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from nas_embedding_suite.nb201_ss import NASBench201
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from scipy.stats import spearmanr
import numpy as np

def convert_matrix_to_graph(matrix):
    G = nx.from_numpy_matrix(matrix)
    return G

nb201_api = NASBench201()

nb2latdict = nb201_api.latency_data
nb2devices = ['1080ti_1', '1080ti_256', '1080ti_32', '2080ti_1', '2080ti_256', '2080ti_32', 'desktop_cpu_core_i7_7820x_fp32', 'desktop_gpu_gtx_1080ti_fp32',      \
                   'embedded_gpu_jetson_nano_fp16', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'essential_ph_1', 'eyeriss', \
                   'fpga', 'gold_6226', 'gold_6240', 'mobile_cpu_snapdragon_450_cortex_a53_int8', 'mobile_cpu_snapdragon_675_kryo_460_int8', 'mobile_cpu_snapdragon_855_kryo_485_int8', \
                   'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_450_adreno_506_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', \
                   'mobile_gpu_snapdragon_855_adreno_640_int8', 'pixel2', 'pixel3', 'raspi4', 'samsung_a50', 'samsung_s7', 'silver_4114', \
                   'silver_4210r', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'titanx_1', 'titanx_256', 'titanx_32', 'titanxp_1', 'titanxp_256', 'titanxp_32']

# filter keys of nb2latdict such that they only contain keys in nb2devices
nb2latdict = {key: nb2latdict[key] for key in nb2latdict if key in nb2devices}

nb2corrs = {}
for key in nb2latdict:
    nb2corrs[key] = {key: 1}
    for otherkey in nb2latdict:
        nb2corrs[key][otherkey] = spearmanr(nb2latdict[key], nb2latdict[otherkey]).correlation
keys = list(nb2corrs.keys())

def create_bipartite_graph(bisect_m, bisect_n, nb2corrs, keys):
    # Create an empty bipartite graph
    B = nx.Graph()
    for node in bisect_m:
        B.add_node(node, bipartite=0)
    for node in bisect_n:
        B.add_node(node, bipartite=1)
    for node_m in bisect_m:
        for node_n in bisect_n:
            weight = nb2corrs[keys[node_m]][keys[node_n]]
            B.add_edge(node_m, node_n, weight=weight)

    return B

m, n = 10, 1
matrix = np.zeros((len(keys), len(keys)))
for i, key in enumerate(keys):
    for j, otherkey in enumerate(keys):
        matrix[i, j] = -nb2corrs[key][otherkey]  
G = convert_matrix_to_graph(matrix)
bisect_m, bisect_n = kernighan_lin_bisection(G, max_iter=10, weight='weight', seed=42)
B = create_bipartite_graph(bisect_m, bisect_n, nb2corrs, keys)
while len(nx.bipartite.sets(B)[0]) > m or len(nx.bipartite.sets(B)[1]) > n:
    left = nx.bipartite.sets(B)[0]
    right = nx.bipartite.sets(B)[1]
    if len(left) > m:
        node_weights = {node: sum([B[node][neighbor]['weight'] for neighbor in B.neighbors(node)]) for node in nx.bipartite.sets(B)[0]}
        min_weight_node = max(node_weights, key=node_weights.get)
        B.remove_node(min_weight_node)
    if len(right) > n:
        node_weights = {node: sum([B[node][neighbor]['weight'] for neighbor in B.neighbors(node)]) for node in nx.bipartite.sets(B)[1]}
        min_weight_node = max(node_weights, key=node_weights.get)
        B.remove_node(min_weight_node)

if True:
    import matplotlib.pyplot as plt
    left, right = nx.bipartite.sets(B)
    pos = dict()
    pos.update((node, (1, index)) for index, node in enumerate(left))
    pos.update((node, (2, index)) for index, node in enumerate(right))
    labels = {node: keys[node] for node in B.nodes()}
    node_colors = ['#1f78b4' if B.nodes[node]['bipartite'] == 0 else '#b2df8a' for node in B.nodes()]
    edge_widths = [0.5 + B[u][v]['weight'] for u, v in B.edges()]
    plt.figure(figsize=(10, 8))
    plt.title("Bipartite Device Set", fontsize=18)
    nx.draw(B, pos=pos, node_size=500, node_color=node_colors, edge_color='#a6a6a6', width=edge_widths, with_labels=True, labels=labels, font_size=10, alpha=0.8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('bipartite_device_set.png', dpi=300)
    plt.show()


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
left, right = nx.bipartite.sets(B)
dev_left = [keys[node] for node in left]
dev_right = [keys[node] for node in right]

corr_chart = {}
for dev_l in dev_left:
    corr_chart[dev_l] = {}
    for dev_r in dev_right:
        corr_chart[dev_l][dev_r] = nb2corrs[dev_l][dev_r]

df = pd.DataFrame(corr_chart)
# Save the dataframe as a confusion matrix style figure
plt.figure(figsize=(10, 8))
plt.title("Device Correlation Matrix", fontsize=18)
sns.heatmap(df, cmap='coolwarm', annot=True, fmt=".2f", vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig('device_corr_matrix.png', dpi=300)





# if True:
#     import matplotlib.pyplot as plt
#     left, right = nx.bipartite.sets(B)
#     pos = dict()
#     pos.update((node, (1, index)) for index, node in enumerate(left))
#     pos.update((node, (2, index)) for index, node in enumerate(right))
#     # Change labels using 'key'
#     labels = dict()
#     for node in left:
#         labels[node] = keys[node]
#     for node in right:
#         labels[node] = keys[node]
#     nx.draw_networkx_labels(B, pos=pos, labels=labels)
#     plt.savefig('bipartite_device_set.png')

# # training_set, test_set = bisection_with_elimination(G, m, n, keys)
# training_set, test_set = [], []
# for tidx in bisect_m:
#     training_set.append(keys[tidx])
# for tidx in bisect_n:
#     test_set.append(keys[tidx])
# print("NASBench201")
# print(training_set)
# print(test_set)

# from nas_embedding_suite.fbnet_ss import FBNet

fbnet_api = FBNet()

fbnetlatdict = fbnet_api.latency_data

# # for each key in fbnetlatdict, calculate spearmanr of fbnetlatdict[key] with every other key
# fbnetcorrs = {}
# for key in fbnetlatdict:
#     fbnetcorrs[key] = {key: 1}
#     for otherkey in fbnetlatdict:
#         fbnetcorrs[key][otherkey] = spearmanr(fbnetlatdict[key], fbnetlatdict[otherkey]).correlation

# # Create an adjacency matrix from your fbnetcorrs dictionary
# keys = list(fbnetcorrs.keys())
# matrix = np.zeros((len(keys), len(keys)))
# for i, key in enumerate(keys):
#     for j, otherkey in enumerate(keys):
#         matrix[i, j] = -fbnetcorrs[key][otherkey]  # Negative because we want to minimize it
        
# G = convert_matrix_to_graph(matrix)
# m = 5
# n = 5
# subset_m, subset_n = get_two_subsets(G, m, n)
# training_set = []
# test_set = []
# for tidx in subset_m:
#     training_set.append(keys[tidx])
# for tidx in subset_n:
#     test_set.append(keys[tidx])
# print("FBNet")
# print(training_set)
# print(test_set)