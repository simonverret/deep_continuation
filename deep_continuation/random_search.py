#!/usr/bin/env python3
#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from deep_continuation import utils
from deep_continuation.utils import ObjectView
from deep_continuation import data
from deep_continuation import train

default_dict = {
    'data': 'B',
    'noise': 1e-5,
    'loss': 'mse',
    'batch_size': 300,
    'epochs': 500,
    'layers': [
        128,
        2000,
        2000,
        2000,
        512
    ],
    'out_unit': 'None',
    'lr': 0.0008,
    'initw': True,
    'dropout': 0,
    'batchnorm': True,
    'optimizer': "adam",
    'weight_decay': 0,
    'smoothing': 1.0,
    'stop': 20,
    'warmup': True,
    'schedule': True,
    'factor': 0.4,
    'patience': 6,
    'seed': int(time.time()),
    'num_workers': 2,
    'cuda': True,
    'valid_fraction': 0.05,
    'metric_batch_size': 64,
    'rescale': False,
    'beta': [20.0],
    'plot': False,
    'standardize': False,
}

# the random pick is recursive:
#   a list of (lists/tuples) will return a list of random picks
#   a tuple of (lists/tuples) will pick one list/tuple to choose from
# the recursion ends when it finds:
#   a list of two elements = defines a range
#   a tuple of many elements = defines a set to random.choice from
#   a standalone value will be returned as is
search_space = {
    "layers": (
        [128, [500,3000], [500,3000], 512],
        [128, [500,3000], [500,3000], [500,3000], 512],
        [128, [500,3000], [500,3000], [500,3000], 512],
        [128, [500,3000], [500,3000], [500,3000], 512],
        [128, [500,3000], [500,3000], [500,3000], [500,3000], 512],
        [128, [500,3000], [500,3000], [500,3000], [500,3000], [500,3000], 512],
    ),
    "loss": ("mse", "mae", "mse", "mae", "dcs", "dca"),
    "noise": (0.0 , [0.01,0.000001]),
    "batch_size": ([100,500],[200,1000]),
    "lr": [0.0005, 0.00001],
    "dropout": (0, [0.0,0.8]),
    'initw': (True,True,False),
    "out_unit": ('None', 'Softmax'), #('None', 'ReLU', 'Softmax', 'Normalizer'),
    "batchnorm": (True,True,False),
    "factor": [0.2,0.8], 
    "patience": [4,10],
    'standardize': (True, False),
}

def pick_from(entity):
    is_list  = type(entity) is list
    is_tuple = type(entity) is tuple
    if not (is_list or is_tuple):
        return entity # because then it is the desired value itself
    
    is_nested = any([(type(item) is list or type(item) is tuple) for item in entity])
    if is_tuple and is_nested:
        return pick_from( random.choice(entity) )
    elif is_tuple:
        return random.choice(entity)
    if is_list and is_nested:
        return [ pick_from(nested) for nested in entity ]
    else:
        a = entity[0]
        b = entity[1]
        if type(a) is int and type(b) is int:
            return random.randint(a,b)
        else :
            return random.uniform(a,b)

def new_args_dict_from(search_space, template_dict = default_dict):
    new_args_dict = template_dict
    for parameter, range_def in search_space.items():
        value = pick_from(range_def)
        new_args_dict[parameter] = value   
    return new_args_dict


args = utils.parse_file_and_command(default_dict, help_dict={})
default_dict = vars(args)

for i in range(30):
    new_args_dict = new_args_dict_from(search_space, default_dict)
    args = ObjectView(new_args_dict)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = args.cuda and torch.cuda.is_available()
    if cuda: 
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda") 
        print('using GPU')
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('no GPU available')

    path_dict = {
        'F': 'data/Fournier/',
        'G': 'data/G1/',
        'B': 'data/B1/',
    }

    train_set = data.ContinuationData(
        path_dict[args.data]+"train/",
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="F" else 20
    )
    valid_set = data.ContinuationData(
        path_dict[args.data]+"valid/",
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

    metric_list = []
    for p, path in path_dict.items():
        dataset = data.ContinuationData(
            path+"valid/",
            base_scale=15 if p=="F" else 20
        )
        for n, noise in noise_dict.items():
            for b, beta, in beta_dict.items():
                for s, scale in scale_dict.items():
                    print(f"loading metric {p+s+n+b}")
                    metric_list.append(data.EvaluationMetric(
                        name = f"{p+n+b+s}",
                        dataset=dataset,
                        loss_dict=loss_dict,
                        noise=noise,
                        beta=beta,
                        scale=scale,
                        std=args.standardize,
                        bs=args.metric_batch_size,
                        num_workers=args.num_workers,
                    ))
    
    train.train(args, device, train_set, valid_set, loss, metric_list=metric_list)
