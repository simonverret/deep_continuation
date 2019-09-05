#
#   deep_continuation
#
#   © Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

import deep_continuation as dcont
import numpy as np
import torch
import random

args_dict = {
    "path": "../data/",
    "no_cuda": False,
    "seed": 20,
    "num_workers": 0,
    "epochs": 1000,
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
    "h1": [1,150], #x10 implicit
    "h2": [1,150], #x10 implicit
    "lr": [0.1,0.0001],
    "batch_size": [3,50], #x10 implicit
    "factor": [0.1,1], 
    "patience": [4,20],
}

class ObjectView():
    def __init__(self,dict):
        self.__dict__.update(dict)

for i in range(72):
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

    pyctmo_train, pyctmo_val = dcont.load_data(args)
    dcont.train(args, device, pyctmo_train, pyctmo_val)
