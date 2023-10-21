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
                9999: {
                    # For quick testing
                    "train": ["1080ti_1","1080ti_32"],
                    "test": ["titan_rtx_256","gold_6226"]
                },
            },
            'fbnet': {
                0: {
                    # GPU,CPU,eCPU to GPU,CPU [DEFAULT HELP/MultiPredict]
                    "train": ["1080ti_1","1080ti_32","1080ti_64","silver_4114","silver_4210r","samsung_a50","pixel3","essential_ph_1","samsung_s7"],
                    "test": ["titanx_1","titanx_32","titanx_64","gold_6240"]
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
                9999: {
                    # For quick testing
                    "train": ["1080ti_1","1080ti_32"],
                    "test": ["titanx_1","titanx_32"]
                },
            }
        }

    def get_data(self, space, index):
        return self.data.get(space, {}).get(index, None)

def generate_latex_table(devices, devlist):
    header = " & ".join(["Device"] + [f"nb201-{i+1}" for i in range(len(devlist['nb201']))] + [f"fbnet-{i+1}" for i in range(len(devlist['fbnet']))])
    header += " \\\\ \\hline \n"

    table_content = ""

    for device in devices:
        row_content = [device]
        for space, lists in devlist.items():
            for l in lists:
                train, test = l.split('|')
                if device in train.split(","):
                    row_content.append("S")
                elif device in test.split(","):
                    row_content.append("T")
                else:
                    row_content.append("-")

        table_content += " & ".join(row_content) + " \\\\ \n"

    table = "\\begin{tabular}{" + "|".join(["c"] * (len(devlist['nb201']) + len(devlist['fbnet']) + 1)) + "}\n"
    table += header
    table += table_content
    table += "\\end{tabular}"

    return table

# if __name__ == "__main__":
    # Usage:
dataset = HardwareDataset()
print(dataset.get_data('nb201', 1))
print(dataset.get_data('fbnet', 3))
# devices ={'mDSP', 'GPUs', 'GPU', 'mCPU', 'eTPU', 'FPGA', 'eCPU', 'eGPU', 'mGPU', 'ASIC', 'mCPU', 'DSP', 'CPU'}

devlist = {"nb201" : ["eTPU,ASIC,mGPU,mCPU|GPU","GPU|eGPU,eTPU,eCPU,DSP","CPU,eGPU,ASIC,mGPU,mDSP|GPU","CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",],
            "fbnet" : ["mCPU,CPU,GPU|ASIC,FPGA,eCPU","mCPU,CPU,FPGA|GPU","mCPU,FPGA|GPU","GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",]}
devices = devlist['nb201'] + devlist['fbnet']
devices = set([item for sublist in devices for item in sublist.replace("|", ",").split(',')])

print(generate_latex_table(devices, devlist))
# flatten list of list called 'dl'
# dl = [item for sublist in devlist for item in sublist.split('|')]
# I have 4 different 'device sets', each device set has a 'train' and 'test'. The train and test set is demarcated by '|', and the type of devices in each set is separated by ','.
# I have this information for two data-sets, NASBench-201 and FBNet. They are provided below:
# NASBench201 =[
# "eTPU,ASIC,mGPU,mCPU|GPU",
# "GPU|eGPU,eTPU,eCPU,DSP",
# "CPU,eGPU,ASIC,mGPU,mDSP|GPU",
# "CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",
# ]
# FBNet = [
# "mCPU,CPU,GPU|ASIC,FPGA,eCPU",
# "mCPU,CPU,FPGA|GPU",
# "mCPU,FPGA|GPU",
# "GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",
# ]
# devices =[
# "eTPU,ASIC,mGPU,mCPU,GPU",
# "GPU,eGPU,eTPU,eCPU,DSP",
# "CPU,eGPU,ASIC,mGPU,mDSP,GPU",
# "CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU,GPUs",
# "mCPU,CPU,GPU,ASIC,FPGA,eCPU",
# "mCPU,CPU,FPGA,GPU",
# "mCPU,FPGA,GPU",
# "GPU,ASIC,CPU,eCPU,mCPU,GPU,mCPU",
# ]
# # flatten list of list
# devices = set([item for sublist in devices for item in sublist.split(',')])

# # I want to present this information in a neat LaTeX table. How can I present it? (ChatGPT Temporary prompt.)
# devlist = {"nb201" : ["eTPU,ASIC,mGPU,mCPU|GPU","GPU|eGPU,eTPU,eCPU,DSP","CPU,eGPU,ASIC,mGPU,mDSP|GPU","CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",]
#            "fbnet" : ["mCPU,CPU,GPU|ASIC,FPGA,eCPU","mCPU,CPU,FPGA|GPU","mCPU,FPGA|GPU","GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",]}

#            I have a set of devices
# devices ={'mDSP', 'GPUs', 'GPU', 'mCPU', 'eTPU', 'FPGA', 'eCPU', 'eGPU', 'mGPU', 'ASIC', 'mCPU', 'DSP', 'CPU'}

# I also have a dictionary of two 'spaces' nb201 and fbnet as follows:
# devlist = {"nb201" : ["eTPU,ASIC,mGPU,mCPU|GPU","GPU|eGPU,eTPU,eCPU,DSP","CPU,eGPU,ASIC,mGPU,mDSP|GPU","CPU,eGPU,eTPU,ASIC,mCPU,mDSP,mCPU,mGPU|GPU",]
#            "fbnet" : ["mCPU,CPU,GPU|ASIC,FPGA,eCPU","mCPU,CPU,FPGA|GPU","mCPU,FPGA|GPU","GPU,ASIC,CPU,eCPU,mCPU|GPU,mCPU",]}

# I want to create a LaTeX table where devices are listed on the rows. There is a new column for every item in the 'nb201' and 'fbnet' lists. For each 'device' that is in the string before "|", mark that entry as 'S', the 'device' that is in the string after "|" mark that entry as "T", mark everything else as "-"

# Provide python code that can produce such a table by taking as the input the 'devices' set and the 'devlist'