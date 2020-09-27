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

from deep_continuation.utils import ObjectView
from deep_continuation import data
from deep_continuation import train

default_dict = {
    'data': 'B1',
    'noise': 1e-4,
    'loss': 'mse',
    'batch_size': 300,
    'epochs': 1000,
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
    'weight_decay': 0,
    'smoothing': 1.0,
    'stop': 40,
    'warmup': True,
    'schedule': True,
    'factor': 0.4,
    'patience': 6,
    'seed': int(time.time()),
    'num_workers': 8,
    'cuda': True,
    'valid_fraction': 0.3,
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
        [128, [40,2500], 512],
        [128, [30,2000], [40,2000], 512],
        [128, [30,1500], [40,2000], [30,1500], 512],
        [128, [1500,3000], [1500,3000], [1500,3000], 512],
        [128, [30,1000], [40,2000], [40,2000], [30,1000], 512],
        [128, [30,800], [40,1000], [40,1000], [40,800], [30,800], 512],
        [128, [100,600], [100,600], [100,600], [100,600], [100,600], [100,600], [100,600], [100,600], [100,600], 512]
    ),
    "loss": ("mse", "mae", "dcs", "dca"),
    # "data": ("G1", "Fournier", "B1", "FournierB"),
    "noise": (0.0 , [0.0,0.001]),
    "batch_size": ([10,500],[200,1000]),
    "lr": [0.001, 0.00001],
    "weight_decay": (0, [0.0,0.8]),
    "dropout": (0, [0.0,0.8]),
    'initw': (True,False),
    "out_unit": ('None', 'ReLU', 'Softmax', 'Normalizer'),
    "batchnorm": (True,False),
    "factor": [0.05,1], 
    "patience": [4,10],
    'rescale': False,
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

for i in range(100):

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


    train_set = data.ContinuationData(f'data/{args.data}/train/', noise=args.noise)

    train_set = data.ContinuationData(
        f'data/{args.data}/train/',
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="Fournier" else 20
    )
    valid_set = data.ContinuationData(
        f'data/{args.data}/valid/',
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="Fournier" else 20
    )

    # VALID LIST
    path_dict = {
        # 'F': 'data/Fournier/valid/',
        # 'G': 'data/G1/valid/',
        'B': 'data/B1/valid/',
    }
    scale_dict = {
        'N': False,
        # 'R': True
    }
    noise_dict = {
        '0': 0,
        '5': 1e-5,
        '3': 1e-3,
        # '2': 1e-2,
    }
    beta_dict = {
        # 'T10': [10.0],
        'T20': [20.0],
        # 'T30': [30.0],
        # 'T35': [35.0],
        # 'l3T': [15.0, 20.0, 25.0], 
        # 'l5T': [10.0, 15.0, 20.0, 25.0, 30.0],
    }

    metrics_dict = {}
    for p, path in path_dict.items():
        for s, scale in scale_dict.items():
            for b, beta, in beta_dict.items():
                for n, noise in noise_dict.items():
                    metrics_dict[p+n+b+s] = data.ContinuationData(
                        path,
                        noise=noise,
                        beta=beta,
                        rescaled=scale,
                        standardize=args.standardize,
                        base_scale=15 if p=="F" else 20
                    )

    train.train(args, device, train_set, valid_set, metrics=metrics_dict)
