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
import deep_continuation as model

default_dict = {
    "path": "../sdata/part/",
    "measure": "Normal",
    "normalize": False,
    "batch_size": 1500,
    "epochs": 1,
    "layers": [
        128,
        512,
        1024,
        512,
        512
    ],
    "out_unit": "None",
    "loss": "MSELoss",
    "lr": 0.01,
    "weight_decay": 0,
    "stop": 40,
    "warmup": True,
    "schedule": True,
    "factor": 0.5,
    "patience": 4,
    "dropout": 0,
    "batchnorm": True,
    "seed": 1579012834,
    "num_workers": 0,
    "cuda": False
}

# the random pick is recursive:
#   a list of (lists/tuples) will return a list of random picks
#   a tuple of (lists/tuples) will pick one list/tuple to choose from
# at the end level
# a list defines a range
# a tuple defines a set to random.choice from
# a standalone value will be returned
search_space = {
    "layers": (
        [128, [40,800], 512],
        [128, [30,200], [40,800], 512],
        [128, [30,200], [40,800], [30,200], 512],
        [128, [30,200], [40,800], [40,800], [30,200], 512],
        [128, [30,800], [40,800], [40,800], [40,800], [30,800], 512]
    ), # x10 implicit
    "lr": [0.001,0.00001],
    "batch_size": [5,200], # x10 implicit
    "factor": [0.05,1], 
    "patience": [4,10],
    "weight_decay": (0, [0.0,0.8]),
    "dropout": (0, [0.0,0.8]),
    "batchnorm": (True,False),
    "out_unit": ('ReLU','None'),
    "loss": ("L1Loss", "MSELoss", "expL1Loss", "invL1Loss", "expMSELoss", "invMSELoss")
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
    dataset = data.ContinuationData(args.path, measure=args.measure, normalize=args.normalize)
    model.train(args, device, dataset)
    
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        os.system('''
                DATE=$(date -u +%Y%m%d)
                cp -r ./ $SLURM_SUBMIT_DIR/deep_cont_$DATE-id$SLURM_JOB_ID
                ''')
