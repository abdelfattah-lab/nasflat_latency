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
                    # mCPUs,FPGA to GPU
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


if __name__ == "__main__":
    # Usage:
    dataset = HardwareDataset()
    print(dataset.get_data('nb201', 1))
    print(dataset.get_data('fbnet', 3))