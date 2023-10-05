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
if True:
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