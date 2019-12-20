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
import deep_continuation as dcont
import numpy as np
import torch
import random

args_dict = {
    "path": "../sdata/",
    "no_cuda": False,
    "seed": int(time.time()),
    "num_workers": 0,
    "epochs": 20,
    "in_size": 128,
    "h1": 512,
    "h2": 512,
    "out_size":512,
    "loss":"L1Loss",
    "lr": 0.01,
    "batch_size": 1500,
    "stop": 40,
    "weight_decay": 0,
    "dropout": 0,
    "schedule": True,
    "factor": 0.5,
    "patience": 8,
    "warmup": True,
    "overwrite": True,
    "save": False
}
search_ranges = {
    "h1": [2,20], #x10 implicit
    "h2": [2,20], #x10 implicit
    "lr": [0.001,0.00001],
    "batch_size": [5,200], #x10 implicit
    "factor": [0.05,1], 
    "patience": [4,10],
    "weight_decay": [0.0,0.8],
    "dropout": [0.0,0.8],
}

class ObjectView():
    def __init__(self,dict):
        self.__dict__.update(dict)

for i in range(2):
    print()
    for key, ran in search_ranges.items():
        if len(ran)>2:
            value = random.choice(ran)
        elif type(ran[0])==int and type(ran[1])==int:
            value = random.randint(ran[0],ran[1])
        else:
            value = random.uniform(ran[0],ran[1])
        args_dict[key] = value   
        print(key, value)
    

    args = ObjectView(args_dict)
    args.h1 = 10*args.h1
    args.h2 = 10*args.h2
    args.batch_size = 10*args.batch_size
    print(dcont.name(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda") 
        print('using GPU')
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('no GPU available')

    train, val = dcont.load_data(args)
    dcont.train(args, device, train, val)
    
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        os.system('''
                DATE=$(date -u +%Y%m%d)
                cp -r ./ $SLURM_SUBMIT_DIR/deep_cont_$DATE-id$SLURM_JOB_ID
                ''')
