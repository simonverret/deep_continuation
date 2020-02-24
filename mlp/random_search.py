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

from utils import ObjectView
import data
import deep_continuation as dc

default_dict = {
    "data": "G1",
    "noise": 0.01,
    "loss": "MSELoss",
    "batch_size": 1500,
    "epochs": 200,
    "layers": [
        128,
        512,
        1024,
        512
    ],
    "out_unit": "None",
    "dropout": 0,
    "batchnorm": True,
    "lr": 0.001,
    "weight_decay": 0,
    "stop": 40,
    "warmup": True,
    "schedule": True,
    "factor": 0.5,
    "patience": 4,
    "seed": int(time.time()),
    "measure": "Normal",
    "normalize": False,
    "num_workers": 0,
    "cuda": False
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
        [128, [40,800], 512],
        [128, [30,200], [40,800], 512],
        [128, [30,200], [40,800], [30,200], 512],
        [128, [30,200], [40,800], [40,800], [30,200], 512],
        [128, [30,800], [40,800], [40,800], [40,800], [30,800], 512]
    ), # x10 implicit
    "loss": ("L1Loss", "MSELoss", "expL1Loss", "invL1Loss", "expMSELoss", "invMSELoss"),
    "data": ("G1", "G2", "G3", "G4"),
    "noise": (0.0 , [0.0,0.1]),
    "batch_size": [5,200], # x10 implicit
    "lr": [0.001, 0.00001],
    "weight_decay": (0, [0.0,0.8]),
    "dropout": (0, [0.0,0.8]),
    "out_unit": ('ReLU','None'),
    "batchnorm": (True,False),
    "factor": [0.05,1], 
    "patience": [4,10],
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

previous_batch_size = 0
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

    # if previous_batch_size != args.batch_size: 
    #     train, val = make_loaders(args.path, args.batch_size, args.num_workers)
    #     previous_batch_size = args.batch_size
    
    train_set = data.ContinuationData(f'data/{args.data}/train/', noise=args.noise)

    ### VALID LIST
    metrics_dict = {
        'G1bse': data.ContinuationData('data/G1/valid/', noise=0.0),
        'G1n01': data.ContinuationData('data/G1/valid/', noise=0.01),
        'G1n05': data.ContinuationData('data/G1/valid/', noise=0.05),
        'G1n10': data.ContinuationData('data/G1/valid/', noise=0.10),
        'G2bse': data.ContinuationData('data/G2/valid/', noise=0.0),
        'G2n01': data.ContinuationData('data/G2/valid/', noise=0.01),
        'G2n05': data.ContinuationData('data/G2/valid/', noise=0.05),
        'G2n10': data.ContinuationData('data/G2/valid/', noise=0.10),
        'G3bse': data.ContinuationData('data/G3/valid/', noise=0.0),
        'G3n01': data.ContinuationData('data/G3/valid/', noise=0.01),
        'G3n05': data.ContinuationData('data/G3/valid/', noise=0.05),
        'G3n10': data.ContinuationData('data/G3/valid/', noise=0.10),
        'G4bse': data.ContinuationData('data/G4/valid/', noise=0.0),
        'G4n01': data.ContinuationData('data/G4/valid/', noise=0.01),
        'G4n05': data.ContinuationData('data/G4/valid/', noise=0.05),
        'G4n10': data.ContinuationData('data/G4/valid/', noise=0.10)
    }

    if not os.path.exists('results'):
        os.mkdir('results')

    for metric in metrics_dict:
        if not os.path.exists(f'results/BEST_{metric}'):
            os.mkdir(f'results/BEST_{metric}')
    ##############
    dc.dump_params(args)    
    dc.train(args, device, train_set, metrics=metrics_dict)
    
    ## CHECKPOINTING (WHEN on SLURM)
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        os.system('''
                DATE=$(date -u +%Y%m%d)
                cp -r ./ $SLURM_SUBMIT_DIR/deep_cont_$DATE-id$SLURM_JOB_ID
                ''')
