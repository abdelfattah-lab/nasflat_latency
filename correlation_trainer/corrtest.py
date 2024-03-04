

import sys, os, torch
import numpy as np
from device_task_list import HardwareDataset

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
nb2_latdata = {}
devices = ['1080ti_1', '1080ti_256', '1080ti_32', '2080ti_1', '2080ti_256', '2080ti_32', 'desktop_cpu_core_i7_7820x_fp32', 'desktop_gpu_gtx_1080ti_fp32',      \
        'embedded_gpu_jetson_nano_fp16', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'essential_ph_1', 'eyeriss', \
        'fpga', 'gold_6226', 'gold_6240', 'mobile_cpu_snapdragon_450_cortex_a53_int8', 'mobile_cpu_snapdragon_675_kryo_460_int8', 'mobile_cpu_snapdragon_855_kryo_485_int8', \
        'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_450_adreno_506_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', \
        'mobile_gpu_snapdragon_855_adreno_640_int8', 'pixel2', 'pixel3', 'raspi4', 'samsung_a50', 'samsung_s7', 'silver_4114', \
        'silver_4210r', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'titanx_1', 'titanx_256', 'titanx_32', 'titanxp_1', 'titanxp_256', 'titanxp_32']
for dev_ in devices:
    nb2_latdata[dev_] = torch.load(BASE_PATH + "/nb201_latency/" + dev_ + ".pt")
 
devices = ["1080ti_1","1080ti_32","1080ti_64","2080ti_1","2080ti_32","2080ti_64","essential_ph_1","eyeriss","fpga",\
                "gold_6226","gold_6240","pixel2","pixel3","raspi4","samsung_a50","samsung_s7","silver_4114","silver_4210r",\
                "titan_rtx_1","titan_rtx_32","titan_rtx_64","titanx_1","titanx_32","titanx_64","titanxp_1","titanxp_32","titanxp_64"]
fbnet_latency_data = {}
for dev_ in devices:
    fbnet_latency_data[dev_] = torch.load(BASE_PATH + "/fbnet_latency/" + dev_ + ".pt")
import pandas as pd
nb2_latframe = pd.DataFrame.from_dict(nb2_latdata)
fbnet_latframe = pd.DataFrame.from_dict(fbnet_latency_data)

# def generate_latex_table(df, name):
#     # Compute the Spearman rank correlation
#     corr_matrix = df.corr(method='spearman')
#     # import pdb; pdb.set_trace()
#     # Format values to 3 decimal places
#     formatted_corr = corr_matrix.applymap(lambda x: "{:.3f}".format(x))

#     # Replace names in headers
#     # Replace names in headers
#     formatted_corr.columns = (formatted_corr.columns.str.replace('mobile', 'm')
#                               .str.replace('snapdragon', 'sd')
#                               .str.replace('embedded', 'e')
#                               .str.replace('desktop', 'd'))
#     formatted_corr.index = (formatted_corr.index.str.replace('mobile', 'm')
#                             .str.replace('snapdragon', 'sd')
#                             .str.replace('embedded', 'e')
#                             .str.replace('desktop', 'd'))
#     # replace _ with \_
#     formatted_corr.columns = formatted_corr.columns.str.replace('_', '\_')
#     formatted_corr.index = formatted_corr.index.str.replace('_', '\_')
#     # Convert DataFrame to LaTeX and print
#     latex_output = formatted_corr.to_latex(escape=False)
#     print(f"\n{name} Latency Correlation:\n")
#     print(latex_output)
def generate_latex_table(df, train_devices, test_devices, name):
    # Compute the Spearman rank correlation
    corr_matrix = df.corr(method='spearman')

    # Extract the sub-matrix
    sub_corr_matrix = corr_matrix.loc[test_devices, train_devices]

    # Format values to 3 decimal places
    formatted_corr = sub_corr_matrix.applymap(lambda x: "{:.3f}".format(x))

    # Replace names in headers and index
    formatted_corr.columns = (formatted_corr.columns.str.replace('mobile', 'm')
                              .str.replace('snapdragon', 'sd')
                              .str.replace('embedded', 'e')
                              .str.replace('desktop', 'd'))
    formatted_corr.index = (formatted_corr.index.str.replace('mobile', 'm')
                            .str.replace('snapdragon', 'sd')
                            .str.replace('embedded', 'e')
                            .str.replace('desktop', 'd'))
    
    # Replace _ with \_
    formatted_corr.columns = formatted_corr.columns.str.replace('_', '\_')
    formatted_corr.index = formatted_corr.index.str.replace('_', '\_')

    # Convert DataFrame to LaTeX and print
    latex_output = formatted_corr.to_latex(escape=False)
    # print(f"\n{name} Latency Correlation between Test and Train devices:\n")
    # print(latex_output)
    num_columns = len(formatted_corr.columns)
    header_line = f"\\multicolumn{{{num_columns + 1}}}{{c}}{{{name} Latency Correlation between Test and Train devices}} \\\\ \hline"
    latex_output = latex_output.replace('\\toprule', header_line)

    print(latex_output)

device_dataset = HardwareDataset()
devmap = {0: "D", 5: "A", 1: "1", 2: "2", 3: "3", 4: "4"}
for ss in ['nb201', 'fbnet']:
    for devtype in [0,1,2,3,4,5]:
        train_devs = device_dataset.data[ss][devtype]['train']
        test_devs = device_dataset.data[ss][devtype]['test']

        if ss == 'nb201':
            generate_latex_table(nb2_latframe, train_devs, test_devs, "NASBench201 %s Train-Test Correlation" % ("N" + devmap[devtype]))
        else:
            generate_latex_table(fbnet_latframe, train_devs, test_devs, "FBNet %s Train-Test Correlation" % ("F" + devmap[devtype]))


# Find correlation between train and test using the nb2_latframe
# generate_latex_table(nb2_latframe, "NB2")
# generate_latex_table(fbnet_latframe, "FBNet")