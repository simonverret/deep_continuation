import os
import yaml

import wandb
import pandas as pd 
import torch
import torch.nn as nn

from deep_continuation import utils
from deep_continuation import train
from deep_continuation.train import MLP

# HERE = os.path.dirname(os.path.realpath(__file__))

data_path = "/Users/Simon/codes/deep_continuation/deep_continuation/"
# data_path = "deep_continuation/"

path_dict = {
    'F': f'{data_path}data/Fournier/valid/',
    'G': f'{data_path}data/G1/valid/',
    'B': f'{data_path}data/B1/valid/',
}

loss_dict = {
    'mse': nn.MSELoss(), 
    'dcs': nn.L1Loss(), 
    'mae': train.dc_square_error, 
    'dca': train.dc_absolute_error,
}

scale_dict = {
    'N': False,
    'R': True
}

noise_dict = {
    '0': 0,
    '5': 1e-5,
    '3': 1e-3,
    '2': 1e-2,
}

beta_dict = {
    'T10': [10.0],
    'T20': [20.0],
    'T30': [30.0],
    'l3T': [15.0, 20.0, 25.0], 
    'l5T': [10.0, 15.0, 20.0, 25.0, 30.0],
}


def data_args_from_name(metric_name):
    metric_loss, metric_key = metric_name.split('_')
    p, n, b, s = metric_key[0], metric_key[1], metric_key[2:5], metric_key[5]
    return dict(
        path=path_dict[p],
        beta=beta_dict[b],
        noise=noise_dict[n],
        rescaled=scale_dict[s],
        base_scale=15 if path_dict[p]=="F" else 20
    )


def metric_args_from_name(metric_name):
    metric_loss, metric_key = metric_name.split('_')
    p, n, b, s = metric_key[0], metric_key[1], metric_key[2:5], metric_key[5]
    return dict(
        name=f"{p+n+b+s}",
        loss_dict=loss_dict,
        noise=noise_dict[n],
        beta=beta_dict[b],
        scale=scale_dict[s],
    )


def download_wandb_table(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    summary_df = pd.DataFrame.from_records([
        {k:v for k,v in run.summary.items() if not k.startswith('gradients/')}
        for run in runs
    ]) 
    config_df = pd.DataFrame.from_records([
        {k:v for k,v in run.config.items() if not k.startswith('_')}
        for run in runs 
    ])  
    wandb_df = pd.DataFrame({
        'wandb_name': [run.name for run in runs],
        'wandb_id': [run.id for run in runs],
        'wandb_state': [run.state for run in runs]
    }) 
    return pd.concat([wandb_df, config_df, summary_df], axis=1)


def get_wandb_model(model_id, device,
    local_path="best_models",
    wandb_project="deep_continuation/beta_and_scale/",
    remote_model_file="best_valid_loss_model.pt"
):
    api = wandb.Api()
    model_filename = os.path.join(local_path,f"{model_id}.pt")
    config_filename = os.path.join(local_path,f"{model_id}.yaml")

    if (not os.path.exists(model_filename)) or (not os.path.exists(config_filename)):
        run = api.run(f"{wandb_project}{model_id}")
        run.file(remote_model_file).download(replace=True)
        os.rename(remote_model_file, model_filename)
        run.file("config.yaml").download(replace=True)
        os.rename("config.yaml", config_filename)

    with open(config_filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config.pop("wandb_version")
        config.pop("_wandb")
        for k, v in config.items():
            config[k] = v['value']

    args = utils.ObjectView(config)
    model = MLP(args).to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    return model, args
