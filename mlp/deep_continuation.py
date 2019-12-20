#!/usr/bin/env python3
#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#
'''
TODO:
2. refactor (make the train function smaller)
'''
#%% INITIALIZATION
import os
import time
import json
import argparse
from glob import glob
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_reader import RezaDataset

# PARSING ARGUMENTS AND PARAMETERS FILE

default_parameters = {
    "file"          : "params.json",
    "path"          : "../sdata/",
    "batch_size"    : 1500,
    "epochs"        : 20,
    "layers"        : [128,512,512,512],
    "loss"          :"L1Loss",
    "lr"            : 0.01,
    "weight_decay"  : 0.0,
    "stop"          : 40,
    "warmup"        : True,
    "schedule"      : True,
    "factor"        : 0.5,
    "patience"      : 8,
    "dropout"       : 0.0,
    "seed"          : int(time.time()),
    "num_workers"   : 0,
    "no_cuda"       : False,
    "overwrite"     : True,
}

help_str = {
    'file'         : 'defines the name of the .json file from which to take the default parameters',
    'path'         : 'path to the SigmaRe.csv and Pi.csv files',
    'batch_size'   : 'batch size for dataloaders',
    'epochs'       : 'Number of epochs to train.',
    "layers"       : 'Sequence of dimension for the neural net, e.g. --layers 128 400 600 512',
    'loss'         : 'path to the SigmaRe.csv and Pi.csv files',
    'lr'           : 'Initial learning rate',
    'weight_decay' : 'L2 regularizer factor of the Adam optimizer',
    'stop'         : 'Early stopping limit',
    'warmup'       : 'linear increase of the learning rate in the first epoch', 
    'schedule'     : 'Turn on the learning rate scheduler (plateau,',
    'factor'       : 'scheduler factor',
    'patience'     : 'scheduler plateau size',
    'dropout'      : 'Dropout factor on every layer',
    'seed'         : 'Random seed',
    'num_workers'  : 'number of workers in the dataloaders',
    'no_cuda'      : 'Disables CUDA',
    'overwrite'    : 'overwrite results file, otherwise appends new results'
}

def get_json_dict(argv=None):
    parser = argparse.ArgumentParser(argv)
    parser.add_argument('--file', type=str, default=default_parameters['file'], help=help_str['file'])
    
    json_filename = parser.parse_known_args()[0].file
    if os.path.exists(json_filename):
        with open(json_filename) as f:
            json_dict = json.load(f)
    else:
        print("warning: input file '"+json_filename+"' not found") 
        json_dict = {}
    return json_dict

def get_args(default_dict, json_dict={}, argv=None):
    parser = argparse.ArgumentParser(argv)
    for name, default in default_parameters.items():
        try: default = json_dict['name']
        except KeyError: pass
        if type(default) is list:
            parser.add_argument('--'+name, nargs='+', type=type(default[0]), default=default, help=help_str[name])
        else:
            parser.add_argument('--'+name, type=type(default), default=default, help=help_str[name])
    return parser.parse_known_args()[0]

args = get_args(default_parameters, get_json_dict())

def name(args):
    layers_str = str(args.layers[0])
    for size in args.layers[1:]:
        layers_str = layers_str + '-' + str(size)

    name = 'mlp{}_bs{}_lr{}_wd{}_drop{}{}{}'.format(
                layers_str,
                args.batch_size, round(args.lr,3), round(args.weight_decay,3), round(args.dropout,3),
                '_wup' if args.warmup else '',
                '_scheduled{}-{}'.format(round(args.factor,3), round(args.patience,3)) if args.schedule else '')
    return name

def dump_params(args):
    with open(f'results/params_{args.loss}_{name(args)}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

# FUNCTIONS

def load_data(args):
    print("Loading data")
    dataset = RezaDataset(args.path)

    validation_split = .1
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split*len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=validation_sampler)
    return train_loader,valid_loader

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        sizeA = args.layers[0]
        for sizeB in args.layers[1:]:
            self.layers.append( nn.Linear(sizeA, sizeB) )
            if args.dropout > 0:
                self.layers.append( nn.Dropout(args.dropout) )
            self.layers.append( nn.ReLU() )
            sizeA = sizeB

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

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
    print(model)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    factor=args.factor, patience=args.patience, verbose=True, min_lr=1e-6)

    best_val_loss = 1e6
    best_mse = 1e6
    best_dc_error = 1e6

    with open('results/training_'+f'{args.loss}_'+name(args)+'.csv', 'w' if args.overwrite else 'a') as f:
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
            print(f'   training   loss: {model.avg_train_loss:.9f}')
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
            print(f'   validation loss: {model.avg_val_loss:.9f}')
            print(f'               MSE: {model.avg_mse:.9f}')
            print(f'          DC error: {model.avg_dc_error:.9f}')
            
            f.write(f'{model.avg_val_loss:.9f}\t')
            f.write(f'{model.avg_mse:.9f}\t')
            f.write(f'{model.avg_dc_error:.9f}\t')
            f.write(f'{optimizer.param_groups[0]["lr"]:.9f}\t')
            f.write('\n')
            f.flush()

            if args.schedule:
                scheduler.step(model.avg_train_loss)
            
            is_best_loss = model.avg_val_loss < best_val_loss
            is_best_mse = model.avg_mse < best_mse
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
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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
    train_loader, valid_loader = load_data(args)
    model = train(args, device, train_loader, valid_loader)
    