import optuna
import os
import subprocess

def objective(trial):
    # Defining the range of the parameters to be optimized
    representation = trial.suggest_categorical('representation', ['adj_gin', 'adj_gin_cate', 'adj_gin_a2vcatezcp', 'adj_gin_arch2vec', 'adj_gin_zcp'])
    transfer_lr = trial.suggest_float('transfer_lr', 0.00001, 0.01)
    transfer_epochs = trial.suggest_categorical('transfer_epochs', [10, 20, 30, 40, 50])
    # gcn_dims = [trial.suggest_int('gcn_dim_{}'.format(i), 16, 512, log=True) for i in range(3)]  # Here 3 layers as per your given python command
    ensemble_fuse_method = trial.suggest_categorical('ensemble_fuse_method', ['add', 'mlp'])

    # Constructing the command
    cmd = [
        'python', 'optuna_main_trf.py',
        '--seed', '42',
        '--name_desc', 'optuna_run',
        '--sample_sizes', '256',
        '--representation', representation,
        '--num_trials', '3',
        '--transfer_sample_sizes', '20',
        '--transfer_lr', str(transfer_lr),
        '--transfer_epochs', str(transfer_epochs),
        '--transfer_hwemb',
        '--task_index', '2',
        '--space', 'nb201',
        '--gnn_type', 'ensemble',
        '--sampling_metric', 'params',
        # '--op_fp_gcn_out_dims'] + list(map(str, gcn_dims[:2])) + [
        # '--forward_gcn_out_dims'] + list(map(str, gcn_dims)) + [
        '--ensemble_fuse_method', ensemble_fuse_method,
        '--optuna_run_id', str(trial.number)
    ]

    # Running the command
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return 0
        # raise RuntimeError(f"Command failed with error: {result.stderr}")

    # Extracting the avg_spr from the saved file
    with open(f'optuna_run_nb201/{trial.number}.txt', 'r') as f:
        avg_spr = float(f.read().strip())

    # Logging the result for the current trial
    with open('optuna_nb2.log', 'a') as log_file:
        log_file.write(f"Trial {trial.number}, Params: {trial.params}, Value: {trial.value}\n")

    return avg_spr

# if __name__ == "__main__":
study = optuna.create_study(direction='maximize', study_name='optuna_nb2')
study.optimize(objective, n_trials=300)

# Once all trials are done, log the top 5 trials
with open('optuna_nb2.log', 'a') as log_file:
    log_file.write("\nTop 5 Trials:\n")
    sorted_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)
    for trial in sorted_trials[:5]:
        log_file.write(f"Trial {trial.number}, Params: {trial.params}, Value: {trial.value}\n")
# # if __name__ == "__main__":
# study = optuna.create_study(direction='maximize', study_name='optuna_nb2')
# study.optimize(objective, n_trials=300)

# # Saving the results to optuna_nb2.log
# with open('optuna_nb2.log', 'w') as log_file:
#     for trial in study.trials:
#         log_file.write(f"Trial {trial.number}, Params: {trial.params}, Value: {trial.value}\n")

#     # Sort trials based on their values and write the top 5
#     sorted_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)  # assuming you want to maximize
#     log_file.write("\nTop 5 Trials:\n")
#     for trial in sorted_trials[:5]:
#         log_file.write(f"Trial {trial.number}, Params: {trial.params}, Value: {trial.value}\n")