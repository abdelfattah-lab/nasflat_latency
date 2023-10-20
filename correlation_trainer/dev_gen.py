import sys
import os
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from networkx.algorithms.community import kernighan_lin_bisection

sys.path.append("..")
sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from nas_embedding_suite.nb201_ss import NASBench201
from nas_embedding_suite.fbnet_ss import FBNet

def convert_matrix_to_graph(matrix):
    return nx.from_numpy_matrix(matrix)

def create_correlation_matrix(latency_data):
    corrs = {}
    for key in latency_data:
        corrs[key] = {key: 1}
        for otherkey in latency_data:
            corrs[key][otherkey] = spearmanr(latency_data[key], latency_data[otherkey]).correlation
    keys = list(corrs.keys())
    matrix = np.zeros((len(keys), len(keys)))
    for i, key in enumerate(keys):
        for j, otherkey in enumerate(keys):
            matrix[i, j] = -corrs[key][otherkey]
    return matrix, corrs, keys

def create_bipartite_graph(bisect_m, bisect_n, corrs, keys):
    B = nx.Graph()
    for node in bisect_m:
        B.add_node(node, bipartite=0)
    for node in bisect_n:
        B.add_node(node, bipartite=1)
    for node_m in bisect_m:
        for node_n in bisect_n:
            weight = corrs[keys[node_m]][keys[node_n]]
            B.add_edge(node_m, node_n, weight=weight)
    return B

def plot_bipartite_graph(B, keys, filename='bipartite_device_set.png'):
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
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_correlation_heatmap(B, keys, corrs, filename='device_corr_matrix.png'):
    left, right = nx.bipartite.sets(B)
    dev_left = [keys[node] for node in left]
    dev_right = [keys[node] for node in right]
    print(filename)
    print(dev_left)
    print(dev_right)
    corr_chart = {}
    for dev_l in dev_left:
        corr_chart[dev_l] = {}
        for dev_r in dev_right:
            corr_chart[dev_l][dev_r] = corrs[dev_l][dev_r]
    df = pd.DataFrame(corr_chart)
    plt.figure(figsize=(10, 8))
    plt.title("Device Correlation Matrix", fontsize=18)
    sns.heatmap(df, cmap='coolwarm', annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def main(latency_data, m, n, fname):
    matrix, corrs, keys = create_correlation_matrix(latency_data)
    G = convert_matrix_to_graph(matrix)
    bisect_m, bisect_n = kernighan_lin_bisection(G, max_iter=10, weight='weight', seed=None)
    if False:
        # Create a bipartite graph from the bisect_n and bisect_m
        latency_data_n = {key: latency_data[key] for idx, key in enumerate(latency_data.keys()) if idx in bisect_n}
        latency_data_m = {key: latency_data[key] for idx, key in enumerate(latency_data.keys()) if idx in bisect_m}
        matrix_n, corrs_n, keys_n = create_correlation_matrix(latency_data_n)
        matrix_m, corrs_m, keys_m = create_correlation_matrix(latency_data_m)
        G_n = convert_matrix_to_graph(matrix_n)
        G_m = convert_matrix_to_graph(matrix_m)
        # import pdb; pdb.set_trace()

        bisect_m_n, bisect_n_n = kernighan_lin_bisection(G_n, max_iter=10, weight='weight', seed=None)
        bisect_m_m, bisect_n_m = kernighan_lin_bisection(G_m, max_iter=10, weight='weight', seed=None)
        # choose n elements from bisect_m_n and n elements from bisect_n_n
        bisect_n = list(random.sample(bisect_n_n, n)) + list(random.sample(bisect_m_n, n))
        bisect_m = list(random.sample(bisect_n_m, m)) + list(random.sample(bisect_m_m, m))
    # import pdb; pdb.set_trace()
    B = create_bipartite_graph(bisect_m, bisect_n, corrs, keys)
    while len(nx.bipartite.sets(B)[0]) > m or len(nx.bipartite.sets(B)[1]) > n:
        left = nx.bipartite.sets(B)[0]
        right = nx.bipartite.sets(B)[1]
        if len(left) > m:
            node_weights = {node: sum([B[node][neighbor]['weight'] for neighbor in B.neighbors(node)]) for node in left}
            min_weight_node = max(node_weights, key=node_weights.get)
            B.remove_node(min_weight_node)
        if len(right) > n:
            node_weights = {node: sum([B[node][neighbor]['weight'] for neighbor in B.neighbors(node)]) for node in right}
            min_weight_node = max(node_weights, key=node_weights.get)
            B.remove_node(min_weight_node)
    
    plot_bipartite_graph(B, keys, filename=f'{fname}bpg.png')
    plot_correlation_heatmap(B, keys, corrs, filename=f'{fname}corrs.png')

if __name__ == '__main__':
    nb201_api = NASBench201()
    nb2latdict = nb201_api.latency_data
    nb2devices = ['1080ti_1', '1080ti_256', '1080ti_32', '2080ti_1', '2080ti_256', '2080ti_32', 'desktop_cpu_core_i7_7820x_fp32', 'desktop_gpu_gtx_1080ti_fp32',      \
                    'embedded_gpu_jetson_nano_fp16', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'essential_ph_1', 'eyeriss', \
                    'fpga', 'gold_6226', 'gold_6240', 'mobile_cpu_snapdragon_450_cortex_a53_int8', 'mobile_cpu_snapdragon_675_kryo_460_int8', 'mobile_cpu_snapdragon_855_kryo_485_int8', \
                    'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_450_adreno_506_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', \
                    'mobile_gpu_snapdragon_855_adreno_640_int8', 'pixel2', 'pixel3', 'raspi4', 'samsung_a50', 'samsung_s7', 'silver_4114', \
                    'silver_4210r', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'titanx_1', 'titanx_256', 'titanx_32', 'titanxp_1', 'titanxp_256', 'titanxp_32']
    nb2latdict = {key: nb2latdict[key] for key in nb2latdict if key in nb2devices}

    fbnet_api = FBNet()
    fbnetlatdict = fbnet_api.latency_data
    main(nb2latdict, m=10, n=3, fname='nb2')
    main(fbnetlatdict, m=10, n=3, fname='fbnet')
