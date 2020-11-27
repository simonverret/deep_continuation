import os
import time
import yaml
import json


import random
import numpy as np
import torch
import torch.nn as nn

import wandb

from deep_continuation import utils
from deep_continuation import data
from deep_continuation import train

def main():
    api = wandb.Api()
    best_folder = "plots/best_models/"
    model_id = "isotekwh"

    if not os.path.exists(f"{best_folder}{model_id}.pt"):
        print("downloading...")
        run = api.run(f"deep_continuation/beta_and_scale/{model_id}")
        run.file("best_valid_loss_model.pt").download(replace=True)
        os.rename("best_valid_loss_model.pt", f"{best_folder}{model_id}.pt")
        run.file("config.yaml").download(replace=True)
        os.rename("config.yaml", f"{best_folder}{model_id}.yaml")
        print("done")
    else:
        print("already downloaded")

    with open(f"{best_folder}{model_id}.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config.pop("wandb_version")
        config.pop("_wandb")
        for k, v in config.items():
            config[k] = v['value']

    for k, v in config.items():
        print(k, v) 

    args = utils.parse_file_and_command(config, {})
    args.plot = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
        print('using GPU')
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('no GPU available')

    path_dict = {
        'F': 'data/Fournier/valid/',
        'G': 'data/G1/valid/',
        'B': 'data/B1/valid/',
    }

    train_set = data.ContinuationData(
        path_dict[args.data],
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="F" else 20
    )
    valid_set = data.ContinuationData(
        path_dict[args.data],
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="F" else 20
    )

    loss_dict = {
        'mse': nn.MSELoss(), 
        'dcs': nn.L1Loss(), 
        'mae': train.dc_square_error, 
        'dca': train.dc_absolute_error,
    }
    
    loss = loss_dict[args.loss]

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

    # metric_list = []
    # for p, path in path_dict.items():
    #     dataset = data.ContinuationData(
    #         path,
    #         base_scale=15 if p=="F" else 20
    #     )
    #     for n, noise in noise_dict.items():
    #         for b, beta, in beta_dict.items():
    #             for s, scale in scale_dict.items():
    #                 print(f"loading metric {p+s+n+b}")
    #                 metric_list.append(data.EvaluationMetric(
    #                     name = f"{p+n+b+s}",
    #                     dataset=dataset,
    #                     loss_dict=loss_dict,
    #                     noise=noise,
    #                     beta=beta,
    #                     scale=scale,
    #                     std=args.standardize,
    #                     bs=args.metric_batch_size,
    #                     num_workers=args.num_workers,
    #                 ))
    
    train.train(args, device, train_set, valid_set, loss, metric_list=[])
    

if __name__ == "__main__":
    main()

