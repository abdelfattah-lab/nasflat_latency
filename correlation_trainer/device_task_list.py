class HardwareDataset:
    def __init__(self):
        self.data = {
            'nb201': {
                0: {
                    # GPU,CPU,mCPU to GPU,CPU,FPGA,eCPU,ASIC [DEFAULT HELP/MultiPredict]
                    "train": ["1080ti_1","1080ti_32","1080ti_256","silver_4114","silver_4210r","samsung_a50","pixel3","essential_ph_1","samsung_s7"],
                    "test": ["titan_rtx_256","gold_6226","fpga","pixel2","raspi4","eyeriss"]
                },
                1: {
                    # eTPU,ASIC,mGPU,mCPU to GPU
                    "train": ['embedded_tpu_edge_tpu_int8', 'eyeriss', 'mobile_gpu_snapdragon_675_adreno_612_int8', 'mobile_gpu_snapdragon_855_adreno_640_int8', 'pixel3'],
                    "test": ['1080ti_1', 'titan_rtx_32', 'titanxp_1', '2080ti_32', 'titan_rtx_1']
                },
                2: {
                    # GPU to eGPU,eTPU,eCPU,DSP
                    "train": ['1080ti_1', '1080ti_32', 'titanx_32', 'titanxp_1', 'titanxp_32'],
                    "test": ['embedded_gpu_jetson_nano_fp16', 'embedded_tpu_edge_tpu_int8', 'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'pixel3']
                },
                3: {
                    # CPU,eGPU,ASIC,mGPU,mDSP to GPU
                    "train": ['desktop_gpu_gtx_1080ti_fp32', 'embedded_gpu_jetson_nano_fp16', 'eyeriss', 'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_gpu_snapdragon_855_adreno_640_int8'],
                    "test": ['1080ti_1', '2080ti_1', 'titanxp_1', '2080ti_32', 'titanxp_32']
                },
                4: {
                    # CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU to GPUs
                    "train": ['desktop_cpu_core_i7_7820x_fp32', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'eyeriss', 'mobile_cpu_snapdragon_855_kryo_485_int8', 'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', 'mobile_gpu_snapdragon_855_adreno_640_int8', 'pixel2'],
                    "test": ['1080ti_1', '2080ti_1', 'titan_rtx_1']
                },
                5: {
                    # GPU,CPU,mCPU to ASIC,CPU,eTPU
                    "train": ["titan_rtx_1","titan_rtx_32","titanxp_1","2080ti_1","titanx_1","1080ti_1","titanx_32","titanxp_32","2080ti_32","1080ti_32","gold_6226","samsung_s7","silver_4114","gold_6240","silver_4210r","samsung_a50","pixel2"],
                    "test": ["eyeriss","desktop_gpu_gtx_1080ti_fp32","embedded_tpu_edge_tpu_int8"]
                },
                9999: {
                    # For quick testing
                    "train": ["1080ti_1","1080ti_32"],
                    "test": ["titan_rtx_256","gold_6226"]
                },
                8989: {
                    "train": ['1080ti_1','1080ti_256', '1080ti_32', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'essential_ph_1', 'pixel3', 'samsung_s7', 'samsung_a50', 'silver_4210r', 'silver_4114'],
                    "test": ["gold_6226"]
                }
            },
            'fbnet': {
                0: {
                    # GPU,CPU,eCPU to GPU,CPU [DEFAULT HELP/MultiPredict]
                    "train": ["1080ti_1","1080ti_32","1080ti_64","silver_4114","silver_4210r","samsung_a50","pixel3","essential_ph_1","samsung_s7"],
                    "test": ['fpga','raspi4','eyeriss']
                },
                1: {
                    # mCPU,CPU,GPU to ASIC,FPGA,eCPU
                    "train": ['2080ti_1', 'essential_ph_1', 'silver_4114', 'titan_rtx_1', 'titan_rtx_32'],
                    "test": ['eyeriss', 'fpga', 'raspi4', 'samsung_a50', 'samsung_s7']
                },
                2: {
                    # mCPU,CPU,FPGA to GPU
                    "train": ['essential_ph_1', 'gold_6226', 'gold_6240', 'pixel3', 'raspi4'],
                    "test": ['1080ti_1', '1080ti_32', '2080ti_32', 'titan_rtx_1', 'titanxp_1']
                },
                3: {
                    # mCPU,FPGA to GPU
                    "train": ['essential_ph_1', 'pixel2', 'pixel3', 'raspi4', 'samsung_s7'],
                    "test": ['1080ti_1', '1080ti_32', '2080ti_1', 'titan_rtx_1', 'titan_rtx_32']
                },
                4: {
                    # GPU,ASIC,CPU,eCPU,mCPU to GPU,mCPU
                    "train": ['1080ti_64', '2080ti_1', 'eyeriss', 'gold_6226', 'gold_6240', 'raspi4', 'samsung_s7', 'silver_4210r', 'titan_rtx_1', 'titan_rtx_32'],
                    "test": ['1080ti_1', 'pixel2', 'essential_ph_1'],
                },
                5: {
                    # GPU to CPU,mCPU
                    "train": ["1080ti_1","1080ti_32","1080ti_64","2080ti_1","2080ti_32","2080ti_64","titan_rtx_1","titan_rtx_32","titan_rtx_64","titanx_1","titanx_32","titanx_64","titanxp_1","titanxp_32","titanxp_64"],
                    "test": ["gold_6226","essential_ph_1","samsung_s7","pixel2"]
                },
                9999: {
                    # For quick testing
                    "train": ["1080ti_1","1080ti_32"],
                    "test": ["titanx_1","titanx_32"]
                },
            }
        }

    def get_data(self, space, index):
        return self.data.get(space, {}).get(index, None)


    def format_device_list(self, device_list):
        # Insert line breaks after every 3 devices
        grouped_devices = [", ".join(device_list[i:i+3]) for i in range(0, len(device_list), 3)]
        formatted_str = " \\\\\n".join(["& & \\texttt{" + device.replace('_', '\\_') + "}" for group in grouped_devices for device in group.split(', ')])
        return formatted_str

    def convert_to_latex(self, space):
        # LaTeX table header
        latex_str = """\\begin{table*}[h]
\\centering
\\begin{tabular}{|c|c|l|}
\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\addlinespace[0.5ex]
\\textbf{Task Index} & \\textbf{Type} & \\textbf{Devices} \\\\
\\cmidrule(lr){1-1} \\cmidrule(lr){2-2} \\cmidrule(lr){3-3} \\addlinespace[0.5ex]
"""

        space_data = self.data[space]
        for task_index, task_data in space_data.items():
            # Adding the 'Train' data
            if task_index != 9999:
                train_devices = self.format_device_list(task_data["train"])
                latex_str += "{0} & Train & {1} \\\\ \n".format(task_index, train_devices)
                
                # Adding the 'Test' data
                test_devices = self.format_device_list(task_data["test"])
                latex_str += "& Test & {0} \\\\ \\cmidrule(lr){{1-1}} \\cmidrule(lr){{2-2}} \\cmidrule(lr){{3-3}} \\addlinespace[0.5ex]\n".format(test_devices)

        # LaTeX table footer
        latex_str += """\\end{tabular}
\\caption{Hardware devices for space: %s}
\\end{table*}""" % space

        return latex_str


# # Example usage
# dataset = HardwareDataset()

# # NB201 table
# nb201_table = dataset.convert_to_latex('nb201')
# print(nb201_table)

# # FBNet table
# fbnet_table = dataset.convert_to_latex('fbnet')
# print(fbnet_table)
# exit(0)
# devlist = {"nb201" : ["GPU,CPU,mCPU|GPU,CPU,FPGA,eCPU,ASIC","eTPU,ASIC,mGPU,mCPU|GPU","GPU|eGPU,eTPU,eCPU,DSP","CPU,eGPU,ASIC,mGPU,mDSP|GPU","CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU","GPU,CPU,mCPU|ASIC,CPU,eTPU"],
#             "fbnet" : ["GPU,CPU,eCPU|GPU,CPU", "mCPU,CPU,GPU|ASIC,FPGA,eCPU","mCPU,CPU,FPGA|GPU","mCPU,FPGA|GPU","GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU","GPU|CPU,mCPU"]}
# devices = devlist['nb201'] + devlist['fbnet']
# devices = set([item for sublist in devices for item in sublist.replace("|", ",").split(',')])
# def generate_latex_table(devices, devlist):
#     header = " & ".join(["Device"] + [f"N{i}" for i in range(len(devlist['nb201']))] + [f"F{i}" for i in range(len(devlist['fbnet']))])
#     header += " \\\\ \\hline \n"

#     table_content = ""

#     for device in devices:
#         row_content = [device]
#         for space, lists in devlist.items():
#             for l in lists:
#                 train, test = l.split('|')
#                 if device in train.split(","):
#                     row_content.append("S")
#                 elif device in test.split(","):
#                     row_content.append("T")
#                 else:
#                     row_content.append("-")
#                 if device in train.split(",") and device in test.split(","):
#                     # remove previous append
#                     row_content.pop()
#                     row_content.append("ST")

#         table_content += " & ".join(row_content) + " \\\\ \n"

#     table = "\\begin{tabular}{" + "|".join(["c"] * (len(devlist['nb201']) + len(devlist['fbnet']) + 1)) + "}\n"
#     table += header
#     table += table_content
#     table += "\\end{tabular}"

#     return table

# # # if __name__ == "__main__":
# #     # Usage:
# # dataset = HardwareDataset()
# # print(dataset.get_data('nb201', 1))
# # print(dataset.get_data('fbnet', 3))
# # # # devices ={'mDSP', 'GPUs', 'GPU', 'mCPU', 'eTPU', 'FPGA', 'eCPU', 'eGPU', 'mGPU', 'ASIC', 'mCPU', 'DSP', 'CPU'}
# # lz = [[item for sublist in list(x.values()) for item in sublist] for x in list(dataset.data['nb201'].values())]
# # # flatten list of list
# # devices = set([item for sublist in lz for item in sublist])



# print(generate_latex_table(devices, devlist))
# if True:
#     import sys
#     import pandas as pd
#     from scipy.stats import spearmanr
#     import numpy as np
#     sys.path.append("..")
#     from nas_embedding_suite.fbnet_ss import FBNet as fbnetclass
#     fbnet_embgen = fbnetclass(normalize_zcp=True, log_synflow=True)
#     from nas_embedding_suite.nb201_ss import NASBench201  as nbclass
#     nb201_embgen = nbclass(normalize_zcp=True, log_synflow=True)
#     arch2vec_fbnet = pd.DataFrame({i: fbnet_embgen.get_arch2vec(i) for i in range(fbnet_embgen.get_numitems())})
#     cate_fbnet = pd.DataFrame({i: fbnet_embgen.get_cate(i) for i in range(fbnet_embgen.get_numitems())})
#     zcp_fbnet = pd.DataFrame({i: fbnet_embgen.get_zcp(i) for i in range(fbnet_embgen.get_numitems())})
#     arch2vec_nb2 = pd.DataFrame({i: nb201_embgen.get_arch2vec(i) for i in range(nb201_embgen.get_numitems())})
#     cate_nb2 = pd.DataFrame({i: nb201_embgen.get_cate(i) for i in range(nb201_embgen.get_numitems())})
#     zcp_nb2 = pd.DataFrame({i: nb201_embgen.get_zcp(i) for i in range(nb201_embgen.get_numitems())})
#     test_idx = 0
#     for test_idx in [0,1,2,3,4,5]:
#         dataset = HardwareDataset()
#         nb2devs = dataset.get_data('nb201', test_idx)['test']
#         fbdevs = dataset.get_data('fbnet', test_idx)['test']
#         nb2devlats = {dev: {i: nb201_embgen.get_latency(i) for i in range(nb201_embgen.get_numitems())} for dev in nb2devs}
#         fbdevlats = {dev: {i: fbnet_embgen.get_latency(i) for i in range(fbnet_embgen.get_numitems())} for dev in fbdevs}
#         nb2devmaxcorr = {dev: {"cate": -1, "zcp": -1, "arch2vec": -1} for dev in nb2devs}
#         for dev in nb2devs:
#             devlatency = np.asarray(list(nb2devlats[dev].values()))
#             maxcorr = -1
#             for idx in list(cate_nb2.index):
#                 spr = spearmanr(devlatency, np.asarray(cate_nb2.loc[idx].tolist())).correlation
#                 if spr > maxcorr:
#                     maxcorr = spr
#             nb2devmaxcorr[dev]["cate"] = maxcorr
#             maxcorr = -1
#             for idx in list(zcp_nb2.index):
#                 spr = spearmanr(devlatency, np.asarray(zcp_nb2.loc[idx].tolist())).correlation
#                 if spr > maxcorr:
#                     maxcorr = spr
#             nb2devmaxcorr[dev]["zcp"] = maxcorr
#             maxcorr = -1
#             for idx in list(arch2vec_nb2.index):
#                 spr = spearmanr(devlatency, np.asarray(arch2vec_nb2.loc[idx].tolist())).correlation
#                 if spr > maxcorr:
#                     maxcorr = spr
#             nb2devmaxcorr[dev]["arch2vec"] = maxcorr
#         fbdevmaxcorr = {dev: {"cate": -1, "zcp": -1, "arch2vec": -1} for dev in fbdevs}
#         for dev in fbdevs:
#             devlatency = np.asarray(list(fbdevlats[dev].values()))
#             maxcorr = -1
#             for idx in list(cate_fbnet.index):
#                 spr = spearmanr(devlatency, np.asarray(cate_fbnet.loc[idx].tolist())).correlation
#                 if spr > maxcorr:
#                     maxcorr = spr
#             fbdevmaxcorr[dev]["cate"] = maxcorr
#             maxcorr = -1
#             for idx in list(zcp_fbnet.index):
#                 spr = spearmanr(devlatency, np.asarray(zcp_fbnet.loc[idx].tolist())).correlation
#                 if spr > maxcorr:
#                     maxcorr = spr
#             fbdevmaxcorr[dev]["zcp"] = maxcorr
#             maxcorr = -1
#             for idx in list(arch2vec_fbnet.index):
#                 spr = spearmanr(devlatency, np.asarray(arch2vec_fbnet.loc[idx].tolist())).correlation
#                 if spr > maxcorr:
#                     maxcorr = spr
#             fbdevmaxcorr[dev]["arch2vec"] = maxcorr
#         print("Test set correlation with each of the arch2vec, cate, zcp encodings for task index: ", test_idx)
#         print("NB201")
#         print(nb2devmaxcorr)
#         print("FBNet")
#         print(fbdevmaxcorr)
#         # now, find the correlation BETWEEN devices in the training set
#         nb2_traindevs = dataset.get_data('nb201', test_idx)['train']
#         fb_traindevs = dataset.get_data('fbnet', test_idx)['train']
#         nb2_traindevlats = {dev: {i: nb201_embgen.get_latency(i) for i in range(nb201_embgen.get_numitems())} for dev in nb2_traindevs}
#         fb_traindevlats = {dev: {i: fbnet_embgen.get_latency(i) for i in range(fbnet_embgen.get_numitems())} for dev in fb_traindevs}
#         nb2_traindevmaxcorr = {dev: -1 for dev in nb2_traindevs}
#         for dev in nb2_traindevs:
#             devlatency = np.asarray(list(nb2_traindevlats[dev].values()))
#             maxcorr = -1
#             for dev2 in list(pd.DataFrame(nb2_traindevlats).columns):
#                 if dev != dev2:
#                     spr = spearmanr(devlatency, np.asarray(nb2_traindevlats[dev2].tolist())).correlation
#                     if spr > maxcorr:
#                         maxcorr = spr
#             nb2_traindevmaxcorr[dev] = maxcorr
#         fb_traindevmaxcorr = {dev: -1 for dev in fb_traindevs}
#         for dev in fb_traindevs:
#             devlatency = np.asarray(list(fb_traindevlats[dev].values()))
#             maxcorr = -1
#             for dev2 in list(pd.DataFrame(fb_traindevlats).columns):
#                 if dev != dev2:
#                     spr = spearmanr(devlatency, np.asarray(fb_traindevlats[dev2].tolist())).correlation
#                     if spr > maxcorr:
#                         maxcorr = spr
#             fb_traindevmaxcorr[dev] = maxcorr
#         print("Correlation within devices in the training set for task index: ", test_idx)
#         print(nb2_traindevmaxcorr)
#         print(fb_traindevmaxcorr)
                    
#         # for cateaxis in range(cate_nb2[0].shape[1]):
#         #     spearmanr(devlatency, cate_nb2[0][:,cateaxis]).correlation
#         #     # if spearmanr is greater than max_cate_corr, then max_cate_corr = spearmanr
#         # spearmanr(devlatency, cate_nb2[0])
#         # Then, find the arch2vec_nb2 that is closest to the devlatency in spearmanr
#         # Then, find the zcp_nb2 that is closest to the devlatency in spearmanr
        

# # flatten list of list called 'dl'
# # dl = [item for sublist in devlist for item in sublist.split('|')]
# # I have 4 different 'device sets', each device set has a 'train' and 'test'. The train and test set is demarcated by '|', and the type of devices in each set is separated by ','.
# # I have this information for two data-sets, NASBench-201 and FBNet. They are provided below:
# # NASBench201 =[
# # "eTPU,ASIC,mGPU,mCPU|GPU",
# # "GPU|eGPU,eTPU,eCPU,DSP",
# # "CPU,eGPU,ASIC,mGPU,mDSP|GPU",
# # "CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",
# # ]
# # FBNet = [
# # "mCPU,CPU,GPU|ASIC,FPGA,eCPU",
# # "mCPU,CPU,FPGA|GPU",
# # "mCPU,FPGA|GPU",
# # "GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",
# # ]
# # devices =[
# # "eTPU,ASIC,mGPU,mCPU,GPU",
# # "GPU,eGPU,eTPU,eCPU,DSP",
# # "CPU,eGPU,ASIC,mGPU,mDSP,GPU",
# # "CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU,GPUs",
# # "mCPU,CPU,GPU,ASIC,FPGA,eCPU",
# # "mCPU,CPU,FPGA,GPU",
# # "mCPU,FPGA,GPU",
# # "GPU,ASIC,CPU,eCPU,mCPU,GPU,mCPU",
# # ]
# # # flatten list of list
# # devices = set([item for sublist in devices for item in sublist.split(',')])

# # # I want to present this information in a neat LaTeX table. How can I present it? (ChatGPT Temporary prompt.)
# # devlist = {"nb201" : ["eTPU,ASIC,mGPU,mCPU|GPU","GPU|eGPU,eTPU,eCPU,DSP","CPU,eGPU,ASIC,mGPU,mDSP|GPU","CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",]
# #            "fbnet" : ["mCPU,CPU,GPU|ASIC,FPGA,eCPU","mCPU,CPU,FPGA|GPU","mCPU,FPGA|GPU","GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",]}

# #            I have a set of devices
# # devices ={'mDSP', 'GPUs', 'GPU', 'mCPU', 'eTPU', 'FPGA', 'eCPU', 'eGPU', 'mGPU', 'ASIC', 'mCPU', 'DSP', 'CPU'}

# # I also have a dictionary of two 'spaces' nb201 and fbnet as follows:
# # devlist = {"nb201" : ["eTPU,ASIC,mGPU,mCPU|GPU","GPU|eGPU,eTPU,eCPU,DSP","CPU,eGPU,ASIC,mGPU,mDSP|GPU","CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",]
# #            "fbnet" : ["mCPU,CPU,GPU|ASIC,FPGA,eCPU","mCPU,CPU,FPGA|GPU","mCPU,FPGA|GPU","GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",]}

# # I want to create a LaTeX table where devices are listed on the rows. There is a new column for every item in the 'nb201' and 'fbnet' lists. For each 'device' that is in the string before "|", mark that entry as 'S', the 'device' that is in the string after "|" mark that entry as "T", mark everything else as "-"

# # Provide python code that can produce such a table by taking as the input the 'devices' set and the 'devlist'