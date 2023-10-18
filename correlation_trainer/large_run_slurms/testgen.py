if False:
    import itertools
    timesteps = [1,2,3]
    residual = [True, False]
    leaky_rely = [True, False]
    unique_attention_projection = [True, False]
    opattention = [True, False]
    spaces = ['PNAS', 'nb201', 'Amoeba']
    configurations = list(itertools.product(timesteps, residual, leaky_rely, unique_attention_projection, opattention, spaces))
    base_command = (
        'False,arch_abl,True,30000,4,python new_main.py --seed 42 '
        '--name_desc arch_abl --gnn_type ensemble --sample_sizes 8 16 32 --batch_size 8  '
        '--representation adj_gin  --space {space} --timesteps {timesteps} '
    )
    for config in configurations:
        timesteps, residual, leaky_rely, unique_attention_projection, opattention, space = config
        command = base_command.format(space=space, timesteps=timesteps)
        if not residual: 
            command += ' --no_residual'
        if not leaky_rely: 
            command += ' --no_leaky_rely'
        if not unique_attention_projection: 
            command += ' --no_unique_attention_projection'
        if not opattention: 
            command += ' --no_opattention'
        print(command)

if False:
    import itertools
    timesteps = [1,2,3]
    back_mlp = [True, False]
    back_opemb = [True, False]
    back_y_info = [True, False]
    spaces = ['ENAS', 'nb201']
    configurations = list(itertools.product(spaces, timesteps, back_mlp, back_opemb, back_y_info))
    base_command = (
        'True,archabl{idx},True,30000,4,python main_abl.py --seed 42 --name_desc archabl '
        '--sample_sizes 64 128 --batch_size 8 --space {space} --timesteps {timesteps} --representation adj_gin '
        '--gnn_type dense --forward_gcn_out_dims 128 128 128 --backward_gcn_out_dims 128 128 128 --replace_bgcn_mlp_dims 128 128 128'
    )
    for idx, config in enumerate(configurations):
        space, timesteps, back_mlp, back_opemb, back_y_info = config
        command = base_command.format(idx=idx, space=space, timesteps=timesteps)
        if back_mlp: 
            command += ' --back_mlp'
        if back_opemb: 
            command += ' --back_opemb'
        if back_y_info: 
            command += ' --back_y_info'
        print(command)
if False:
    import itertools
    timesteps = [1,2,3,4,5,6,7,8,9,10]
    back_mlp = [True]
    back_opemb = [True]
    back_y_info = [False]
    spaces = ['ENAS', 'nb201', 'nb101', 'PNAS']
    configurations = list(itertools.product(spaces, timesteps, back_mlp, back_opemb, back_y_info))
    base_command = (
        'True,archabl{idx},True,30000,4,python main_abl.py --seed 42 --name_desc timeabl '
        '--sample_sizes 64 128 --batch_size 8 --space {space} --timesteps {timesteps} --representation adj_gin '
        '--gnn_type dense --forward_gcn_out_dims 128 128 128 --backward_gcn_out_dims 128 128 128 --replace_bgcn_mlp_dims 128 128 128'
    )
    for idx, config in enumerate(configurations):
        space, timesteps, back_mlp, back_opemb, back_y_info = config
        command = base_command.format(idx=idx, space=space, timesteps=timesteps)
        if back_mlp: 
            command += ' --back_mlp'
        if back_opemb: 
            command += ' --back_opemb'
        if back_y_info: 
            command += ' --back_y_info'
        print(command)
# True,opRptb,True,30000,4,python main_abl.py --seed 42 --name_desc opRptb --sample_sizes 64 128 --batch_size 8 --space nb101 --timesteps 2 --representation adj_gin --gnn_type dense --forward_gcn_out_dims 128 128 128 --backward_gcn_out_dims 128 128 128 --replace_bgcn_mlp_dims 128 128 128 --num_trials 5  --ensemble_fuse_method add --randopupdate
if False:
    import itertools
    timesteps = [2]
    back_mlp = [True]
    back_opemb = [True, False]
    back_y_info = [True, False]
    detach_modes = ['default', 'detach_all', 'detach_none']
    spaces = ['ENAS', 'nb201', 'nb101', 'PNAS']
    configurations = list(itertools.product(spaces, timesteps, back_mlp, back_opemb, back_y_info, detach_modes))
    base_command = (
        'True,archabl{idx},True,30000,4,python main_abl.py --seed 42 --name_desc timeabl '
        '--sample_sizes 64 128 --batch_size 8 --space {space} --timesteps {timesteps} --representation adj_gin '
        '--gnn_type dense --forward_gcn_out_dims 128 128 128 --backward_gcn_out_dims 128 128 128 --replace_bgcn_mlp_dims 128 128 128'
    )
    for idx, config in enumerate(configurations):
        space, timesteps, back_mlp, back_opemb, back_y_info, detach_mode = config
        command = base_command.format(idx=idx, space=space, timesteps=timesteps)
        if back_mlp: 
            command += ' --back_mlp'
        if back_opemb: 
            command += ' --back_opemb'
        if back_y_info: 
            command += ' --back_y_info'
        command += ' --detach_mode %s' % (detach_mode,)
        print(command)

if True:
    import itertools
    # devices = ['1080ti_1', '1080ti_256', '1080ti_32', '2080ti_1', '2080ti_256', '2080ti_32', 'desktop_cpu_core_i7_7820x_fp32', 'desktop_gpu_gtx_1080ti_fp32',      \
    #                'embedded_gpu_jetson_nano_fp16', 'embedded_gpu_jetson_nano_fp32', 'embedded_tpu_edge_tpu_int8', 'essential_ph_1', 'eyeriss', 'flops_nb201_cifar10', \
    #                'fpga', 'gold_6226', 'gold_6240', 'mobile_cpu_snapdragon_450_cortex_a53_int8', 'mobile_cpu_snapdragon_675_kryo_460_int8', 'mobile_cpu_snapdragon_855_kryo_485_int8', \
    #                'mobile_dsp_snapdragon_675_hexagon_685_int8', 'mobile_dsp_snapdragon_855_hexagon_690_int8', 'mobile_gpu_snapdragon_450_adreno_506_int8', 'mobile_gpu_snapdragon_675_adreno_612_int8', \
    #                'mobile_gpu_snapdragon_855_adreno_640_int8', 'nwot_nb201_cifar10', 'params_nb201_cifar10', 'pixel2', 'pixel3', 'raspi4', 'samsung_a50', 'samsung_s7', 'silver_4114', \
    #                'silver_4210r', 'titan_rtx_1', 'titan_rtx_256', 'titan_rtx_32', 'titanx_1', 'titanx_256', 'titanx_32', 'titanxp_1', 'titanxp_256', 'titanxp_32']
    devices = ['titan_rtx_256', 'gold_6226', 'pixel2', 'raspi4', 'eyeriss', 'fpga']
    representations = [ 'adj_mlp', 'zcp', 'cate', 'arch2vec', 'adj_gin', 'adj_gin_zcp', 'adj_gin_cate', 'adj_gin_arch2vec']
    configurations = list(itertools.product(devices, representations))
    base_command = (
        'True,lat_{idx},True,30000,4,python main_abl.py --seed 42 --name_desc lattest --sample_sizes 4 8 12 16 24 32 64 128 '
        '--batch_size 8 --space nb201 --representation {repr_} --gnn_type dense --separate_op_fp --device {dev_} --num_trials 5'
    )
    for idx, config in enumerate(configurations):
        dev_, repr_ = config
        command = base_command.format(idx=idx, dev_=dev_, repr_=repr_)
        print(command)