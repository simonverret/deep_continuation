#!/usr/bin/env python3
#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

import time
import os
import json
from glob import glob
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from data import make_loaders
from utils import parse_file_and_command



# GLOBAL PARAMETERS & PARSING

default_parameters = {
    "path"          : "../sdata/",
    "batch_size"    : 1500,
    "epochs"        : 2,
    "layers"        : [128,1024,1024,1024,1024,512],
    "out_unit"      : "ReLU",
    "loss"          : "L1Loss",
    "lr"            : 0.01,
    "weight_decay"  : 0.0,
    "stop"          : 40,
    "warmup"        : True,
    "schedule"      : True,
    "factor"        : 0.5,
    "patience"      : 12,
    "dropout"       : 0.0,
    "batchnorm"     : True,
    "seed"          : int(time.time()),
    "num_workers"   : 0,
    "cuda"          : False,
}

help_strings = {
    'file'          : 'defines the name of the .json file from which to take the default parameters',
    'path'          : 'path to the SigmaRe.csv and Pi.csv files',
    'batch_size'    : 'batch size for dataloaders',
    'epochs'        : 'number of epochs to train.',
    'layers'        : 'sequence of dimensions for the neural net, includes input and output, e.g. --layers 128 400 600 512',
    'out_unit'      : 'select the output unit; "None", "ReLU"',
    'loss'          : 'loss function to be used (see the code to find all possibilities)',
    'lr'            : 'initial learning rate',
    'weight_decay'  : 'L2 regularization factor passed to the Adam optimizer',
    'stop'          : 'early stopping limit (number of epochs allowed without improvement)',
    'warmup'        : 'activate linear increase of the learning rate in the first epoch', 
    'schedule'      : 'Turn on the learning rate scheduler (plateau,',
    'factor'        : 'scheduler factor at plateau',
    'patience'      : 'scheduler plateau (number of epochs without improvement triggering reduction of lr)',
    'dropout'       : 'dropout probability on all layers but the last one',
    'batchnorm'     : 'apply batchnorm (after ReLU) on all layers but the last one',
    'seed'          : 'seed for the random generator number (time.time() if unspecified)',
    'num_workers'   : 'number of workers used in the dataloaders',
    'cuda'          : 'enables CUDA',
}

'''
The next function allows to call the current script with arguments and fills the
help option. In other words, this will work:
    $ deep_continutation.py --no_cuda --layers 128 256 256 512 -lr 0.001
    $ deep_continutation.py --help
The default_parameters dictionary above serves as a template, so you can add
parameters (float, int, str, bool, or [int]) and the parsing should adapt.
The function, when possible, will:
    1. replace the default value with the one found in 'params.json', then
    2. replace this value with the one specified by command arguments, and then 
    3. return an argparse.Namespace object (argparse is standard, see its doc)
Thus, from here, all parameters should be accessed as:
    args.parameter
note: for every bool flag, an additional --no_flag is defined to turn it off.
'''
args = parse_file_and_command(default_parameters, help_strings)


# TODO: 
# move these two function out to utils.py and have a more general treatment 
# ideas: 
#   name(args, naming_dict)  # where naming dict specifies parameters to use 

def name(args):
    layers_str = str(args.layers[0])
    for size in args.layers[1:]:
        layers_str = layers_str + '-' + str(size)

    name = 'mlp{}_bs{}_lr{}_wd{}_drop{}{}{}{}{}'.format(
                layers_str,
                args.batch_size, round(args.lr,5), round(args.weight_decay,3), round(args.dropout,3),
                f'_{args.out_unit}',
                '_bn' if args.batchnorm else '',
                '_wup' if args.warmup else '',
                '_scheduled{}-{}'.format(round(args.factor,3), round(args.patience,3)) if args.schedule else '')
    return name

def dump_params(args):
    with open(f'results/params_{args.loss}_{name(args)}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


# MODEL

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        sizeA = args.layers[0]
        for sizeB in args.layers[1:]:
            self.layers.append( nn.Linear(sizeA, sizeB) )
            if args.dropout > 0:
                self.layers.append( nn.Dropout(args.dropout) )
            if args.batchnorm:
                self.layers.append( nn.BatchNorm1d(sizeB) )
            self.layers.append( nn.ReLU() )
            sizeA = sizeB
        
        # last layer
        self.layers.append( nn.Linear( sizeA, args.layers[-1] ) )
        if args.out_unit == 'ReLU': 
            self.layers.append( nn.ReLU() )
        elif args.out_unit == 'None': pass
        else: raise ValueError('out_unit unknown')

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)



# CUSTOM LOSS FUNCTIONS

def weightedL1Loss(outputs, targets):
    if not hasattr(weightedL1Loss, 'weights'):
        output_size = outputs.shape[1]
        weightedL1Loss.weights = torch.exp(-torch.arange(output_size, dtype=torch.float)/100)
        print('loss weights =', weightedL1Loss.weights)
    out = torch.abs(outputs-targets) * weightedL1Loss.weights
    out = torch.mean(out)
    return out

def weightedMSELoss(outputs, targets):
    if not hasattr(weightedMSELoss, 'weights'):
        output_size = outputs.shape[1]
        weightedMSELoss.weights = torch.exp(-10*torch.arange(output_size, dtype=torch.float)/100)
        print('loss weights =', weightedMSELoss.weights)
    out = (outputs-targets)**2 * weightedMSELoss.weights
    out = torch.mean(out)
    return out



# CUSTOM SCORES & SAVING TOOLS

def mse(outputs, targets):
    ''' mean square error '''
    return torch.mean((outputs-targets)**2)

def dc_error(outputs, targets):
    ''' computes the 0th component difference (DC conductivity)'''
    return torch.mean(torch.abs(outputs[:,0]-targets[:,0]))

def save_best(criteria_str, model, args):
    for filename in glob(f'results/BEST_{criteria_str}*_epoch*{name(args)}*'): 
        os.remove(filename)
    if criteria_str == 'mse':
        score = model.avg_mse
    elif criteria_str == 'dc_error':
        score = model.avg_dc_error
    else:
        score = model.avg_val_loss
    torch.save(model.state_dict(), f'results/BEST_{criteria_str}{score:.9f}_epoch{model.epoch}_{name(args)}.pt')



def train(args, device, train_loader, valid_loader): 
    model = MLP(args).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    ## standard loss functions
    if args.loss == "L1Loss":
        criterion = nn.L1Loss()
    elif args.loss == "KLDivLoss":
        criterion = nn.KLDivLoss()
    elif args.loss == "MSELoss":
        criterion = nn.MSELoss()
    ## non standard loss functions
    elif args.loss == "expWeightL1Loss":
        criterion = weightedL1Loss
    elif args.loss == "invWeightL1Loss":
        criterion = weightedL1Loss
        weightedL1Loss.weights = 1/torch.arange(1,args.out_size+1, dtype=torch.float)
    elif args.loss == "expWeightMSELoss":
        criterion = weightedMSELoss
    elif args.loss == "invWeightMSELoss":
        criterion = weightedMSELoss
        weightedMSELoss.weights = 1/torch.arange(1,args.out_size+1, dtype=torch.float)
    elif args.loss == "hotDC_MSELoss":
        criterion = weightedMSELoss
        loss_weights = torch.ones(args.out_size, dtype=torch.float)
        loss_weights[0] *= 100
        weightedMSELoss.weights = loss_weights
    else:
        raise ValueError('Unknown loss function "'+args.loss+'"')
    
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=args.factor, patience=args.patience, 
                verbose=True, min_lr=1e-6
            )

    best_val_loss = 1e6
    best_mse = 1e6
    best_dc_error = 1e6

    with open('results/training_'+f'{args.loss}_'+name(args)+'.csv', 'w') as f:
        f.write('epoch')
        f.write('\ttrain_loss')
        f.write('\tval_loss')
        f.write('\tmse')
        f.write('\tdc_error')
        f.write('\tlr\n')

        print('training',name(args))
        for epoch in range(1,args.epochs+1):
            
            print(f' epoch {epoch}')
            f.write(f'{epoch}\t')
            model.epoch = epoch
            
            model.train()
            model.avg_train_loss = 0
            train_n_iter = 0

            if args.warmup and epoch==1:
                print('   linear warm-up of learning rate')
            for batch_number, (inputs, targets)  in enumerate(train_loader):
                if args.warmup and epoch==1:
                    tmp_lr = batch_number*args.lr/len(train_loader)
                    for g in optimizer.param_groups:
                        g['lr'] = tmp_lr

                inputs = inputs.to(device).float()
                targets = targets.to(device).float()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                model.avg_train_loss += loss.item()
                train_n_iter += 1
            model.avg_train_loss = model.avg_train_loss/train_n_iter
            print(f'   train loss: {model.avg_train_loss:.9f}')
            f.write(f'{model.avg_train_loss:.9f}\t')

            model.eval()
            model.avg_val_loss = 0
            model.avg_mse = 0
            model.avg_dc_error = 0
            val_n_iter = 0
            for batch_number, (inputs, targets)  in enumerate(valid_loader):
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                model.avg_val_loss += loss.item()
                model.avg_mse      += mse(outputs,targets).item()
                model.avg_dc_error += dc_error(outputs,targets).item()
                val_n_iter += 1
                
            model.avg_val_loss = model.avg_val_loss/val_n_iter
            model.avg_mse      = model.avg_mse     /val_n_iter
            model.avg_dc_error = model.avg_dc_error/val_n_iter
            print(f'   valid loss: {model.avg_val_loss:.9f}')
            print(f'          MSE: {model.avg_mse:.9f}')
            print(f'     DC error: {model.avg_dc_error:.9f}')
            
            f.write(f'{model.avg_val_loss:.9f}\t')
            f.write(f'{model.avg_mse:.9f}\t')
            f.write(f'{model.avg_dc_error:.9f}\t')
            f.write(f'{optimizer.param_groups[0]["lr"]:.9f}\t')
            f.write('\n')
            f.flush()

            if args.schedule:
                scheduler.step(model.avg_train_loss)
            
            is_best_loss     = model.avg_val_loss < best_val_loss
            is_best_mse      = model.avg_mse      < best_mse
            is_best_dc_error = model.avg_dc_error < best_dc_error
            
            if is_best_loss:
                best_loss_model = deepcopy(model)
                save_best(args.loss, model, args)
                best_val_loss = model.avg_val_loss
            if is_best_mse:
                best_mse_model = deepcopy(model)
                save_best('mse', model, args)
                best_mse = model.avg_mse
            if is_best_dc_error:
                best_dc_model = deepcopy(model)
                save_best('dc_error', model, args)
                best_dc_error = model.avg_dc_error
            if is_best_loss or is_best_mse or is_best_mse: 
                early_stop_count = args.stop
            else: 
                early_stop_count -= 1
            
            if early_stop_count==0:
                print('early stopping limit reached!!')
                break
    
    for i, criterion in enumerate([args.loss, 'mse', 'dc_error']):
        best_model = [best_loss_model, best_mse_model, best_dc_model][i]
        
        results_filename = f'results/all_bests_{criterion}.csv'
        if not os.path.exists(results_filename):
            with open(results_filename,'w') as f:
                f.write('\t'.join([s for s in [
                            'val_loss',
                            'train_loss',
                            'mse',
                            'dc_error',
                            'epoch'
                        ]]))
                f.write('\t')            
                f.write('\t'.join(vars(args).keys()))
                f.write('\n')
        with open(results_filename,'a') as f:
            f.write('\t'.join([str(s) for s in [
                        best_model.avg_val_loss,
                        best_model.avg_train_loss,
                        best_model.avg_mse,
                        best_model.avg_dc_error,
                        best_model.epoch
                    ]]))
            f.write('\t')
            f.write('\t'.join([str(val) for val in vars(args).values()]))
            f.write('\n')

    return model





if __name__=="__main__":
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

    if not os.path.exists('results'):
        os.mkdir('results')
    dump_params(args)
    train_loader, valid_loader = make_loaders(args.path, args.batch_size, args.num_workers)
    model = train(args, device, train_loader, valid_loader)
    