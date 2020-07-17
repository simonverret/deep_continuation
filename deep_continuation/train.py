#!/usr/bin/env python3
#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#
#%%
import os
import time
import json
from glob import glob
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from deep_continuation import data
from deep_continuation import utils

import wandb
wandb.init(project="mlp")


TORCH_MAX = torch.finfo(torch.float64).max

# GLOBAL PARAMETERS & PARSING

default_parameters = {
    'data': 'G1',
    'noise': 0.001,
    'loss': 'MSELoss',
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
    'dropout': 0,
    'batchnorm': True,
    'lr': 0.0008,
    'weight_decay': 0,
    'stop': 40,
    'warmup': True,
    'schedule': True,
    'factor': 0.4,
    'patience': 6,
    'seed': int(time.time()),
    'measure': 'Normal',
    'normalize': False,
    'num_workers': 4,
    'cuda': True,
    'valid_fraction':0.3,
    'resampling': 'default'
}

help_strings = {
    'measure'       : 'Resa uses "squared" measure to enhance low frequencies resolution',
    'normalize'     : 'multiplies each target spectrum by the value at the first Matsubara frequency',
    'path'          : 'path to the SigmaRe.csv and Pi.csv files for the training set',
    'noise'         : 'noise to the matsubara spectra',
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
    'resampling'    : 'changes the data file for the either "cbrt" or "sqrt"'
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
args = utils.parse_file_and_command(default_parameters, help_strings)


# TODO: 
# move these two function out to utils.py and have a more general treatment
# ideas: 
#   name(args, naming_dict)  # where naming dict specifies parameters to use

def name(a = args): ## a = args
    layers_str = str(a.layers[0])
    for size in a.layers[1:]:
        layers_str = layers_str + '-' + str(size)

    name  = f'mlp{layers_str}_{a.loss}_{a.data}n{round(a.noise,3)}_bs{a.batch_size}'
    name += f'_lr{round(a.lr, 5)}_wd{round(a.weight_decay, 3)}'
    name += f'_{round(a.dropout, 3)}_{a.out_unit}'
    name += f'_bn' if a.batchnorm else ''
    name += f'_wup' if a.warmup else ''
    name += f'_sch{round(a.factor, 3)}-{round(a.patience, 3)}' if a.schedule else ''
    return name


def dump_params(args):
    with open(f'results/params_{name(args)}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.name = name(args)
        self.epoch = 0

        self.layers = nn.ModuleList()
        sizeA = args.layers[0]
        for sizeB in args.layers[1:]:
            self.layers.append(nn.Linear(sizeA, sizeB))
            if args.dropout > 0:
                self.layers.append(nn.Dropout(args.dropout))
            if args.batchnorm:
                self.layers.append(nn.BatchNorm1d(sizeB))
            self.layers.append(nn.ReLU())
            sizeA = sizeB

        # last layer
        self.layers.append(nn.Linear(sizeA, args.layers[-1]) )

        if args.out_unit == 'None':
            pass
        elif args.out_unit == 'ReLU':
            self.layers.append(nn.ReLU())
        elif args.out_unit == 'Softmax':
            self.layers.append(nn.Softmax(dim=-1))
        ## Here would be a place for our custom Softmax
        else: 
            raise ValueError('out_unit unknown')

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)


def mse(outputs, targets):
    ''' mean square error '''
    return torch.mean((outputs-targets)**2)


def dc_error(outputs, targets):
    ''' computes the 0th component difference (DC conductivity)'''
    return torch.mean(torch.abs(outputs[:, 0]-targets[:, 0]))


class Metric():
    def __init__(self, name, data_set, loss_list=['mse', 'dc_error'], batch_size=512):
        self.valid_loader = DataLoader(data_set, batch_size=batch_size, drop_last=True, shuffle=False)
        self.name = name
        self.batch_size = batch_size

        self.loss_list = loss_list
        self.loss_value = {lname: 0 for lname in loss_list}
        self.best_loss = {lname: TORCH_MAX for lname in loss_list}
        self.best_model = {lname: None for lname in loss_list}

        self.loss = {}
        for lname in self.loss_list:
            if lname == "L1Loss":
                self.loss[lname] = nn.L1Loss()
            elif lname == "KLDivLoss":
                self.loss[lname] = nn.KLDivLoss()
            elif lname == "MSELoss" or lname == 'mse':
                self.loss[lname] = nn.MSELoss()
            elif lname == 'dc_error':
                self.loss[lname] = dc_error
            elif hasattr(train_set, 'custom_loss'):
                self.loss[lname] = data_set.custom_loss(lname)
            else: 
                raise ValueError(f'Unknown loss function "{lname}"')

    def evaluate(self, model, device, save_best=False, fraction=0.3):
        for lname in self.loss_list:
            self.loss_value[lname] = 0

        stop_at_batch = fraction*len(self.valid_loader)//self.batch_size+1
        batch_count = 0
        for batch_number, (inputs, targets)  in enumerate(self.valid_loader):
            if batch_number == stop_at_batch: break
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            for lname in self.loss_list:
                loss = self.loss[lname](outputs,targets).item()
                self.loss_value[lname] += loss
            batch_count += 1

        for lname in self.loss_list:
            self.loss_value[lname] /= batch_count

        is_best = False
        for lname in self.loss_list:
            score = self.loss_value[lname]
            if score < self.best_loss[lname]:
                self.best_loss[lname] = score
                self.best_model[lname] = deepcopy(model)
                if save_best:
                    for filename in glob(f'results/BEST_{self.name}/{lname}*_epoch*{model.name}*'): 
                        os.remove(filename)
                    torch.save(model.state_dict(), f'results/BEST_{self.name}/{lname}{score:.9f}_epoch{model.epoch}_{model.name}.pt')
                is_best = True
        return is_best

    def print_results(self):
        print(f'      {self.name}:  ', end = '')
        for lname in self.loss_list:
            print(f' {self.loss_value[lname]:.9f} ({lname}) ', end = '')
        print()

    def write_header(self, file):
        for lname in self.loss_list:
            file.write(f'\t{self.name}_{lname}')

    def write_results(self, file):
        for lname in self.loss_list:
            file.write(f'\t{self.loss_value[lname]:.9f}')



def train(args, device, train_set, metrics=None):
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

    loss_list=['mse', 'dc_error']
    metric_list = []
    for name, dataset in metrics.items():
        metric_list.append( Metric(name, dataset, loss_list=loss_list) )

    model = MLP(args).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    wandb.watch(model)

    if args.loss == "L1Loss":
        criterion = nn.L1Loss()
    elif args.loss == "KLDivLoss":
        criterion = nn.KLDivLoss()
    elif args.loss == "MSELoss":
        criterion = nn.MSELoss()
    elif hasattr(train_set, 'custom_loss'):
        criterion = train_set.custom_loss(args.loss)
    else: 
        raise ValueError(f'Unknown loss function "{args.loss}"')
    
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=args.factor, patience=args.patience, 
                verbose=True, min_lr=1e-6
            )

    with open(f'results/training_{model.name}.csv', 'w') as f:
        f.write('epoch')
        f.write('\tlr')
        f.write('\ttrain_loss')
        for metric in metric_list:
            metric.write_header(f)
        f.write('\n')

        print('training', model.name)
        early_stop_count = args.stop
        for epoch in range(1,args.epochs+1):
            print(f' epoch {epoch}')
            f.write(f'{epoch}\t')
            f.write(f'{optimizer.param_groups[0]["lr"]:.9f}\t')
            model.epoch = epoch
            
            model.train()
            model.avg_train_loss = 0
            train_n_iter = 0

            if args.warmup and epoch==1:
                print('   linear warm-up of learning rate')
            
            for batch_number, (inputs, targets)  in enumerate(train_loader):
                if args.warmup and epoch==1:
                    tmp_lr = (batch_number+1)*args.lr/len(train_loader)
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
            f.write(f'{model.avg_train_loss:.9f}')
         
            model.eval()   
            found_one_best = False
            for metric in metric_list:
                is_best = metric.evaluate(model, device, save_best=True, fraction=args.valid_fraction)
                if is_best: found_one_best = True
                metric.print_results()
                metric.write_results(f)
            
            f.write('\n')
            f.flush()

            if found_one_best: 
                early_stop_count = args.stop
            else: 
                early_stop_count -= 1

            if args.schedule:
                scheduler.step(model.avg_train_loss)
                        
            if early_stop_count==0:
                print('early stopping limit reached!!')
                break
    
    print('final_evaluation')
    all_best_file = f'results/all_bests.csv'
    if not os.path.exists(all_best_file): # header
        with open(all_best_file,'w') as f:
            for metric in metric_list:
                for lname in loss_list:
                    f.write( f'{metric.name}_{lname}\t' )
            f.write('\t'.join(vars(args).keys())+'\t')
            f.write('best_epochs\n')
    with open(all_best_file,'a') as f:
        epoch_list = []
        for metric in metric_list:
            for lname in loss_list:
                tmp_model = metric.best_model[lname]
                tmp_model.eval()
                epoch_list.append(tmp_model.epoch)
                metric.evaluate(tmp_model, device, save_best=False, fraction=1.0)
                metric.print_results()
                f.write(f'{metric.loss_value[lname]}\t')
                
        f.write('\t'.join([str(v) for v in vars(args).values()])+'\t')
        f.write(f'{str(epoch_list)}\n')

    all_losses_dict = {}
    for metric in metric_list:
        for lname in loss_list:
            all_losses_dict[f"{metric.name}_{lname}"] = metric.loss_value[lname]
    wandb.log(all_losses_dict)

    return model


if __name__=="__main__":

    args.cuda = True
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

    train_set = data.ContinuationData(f'data/{args.data}/train/', noise=args.noise, resampling=args.resampling)

    ### VALID LIST
    metrics_dict = {
        'G1bse': data.ContinuationData('data/G1/valid/', noise=0.0  , resampling=args.resampling),
        'G1ne2': data.ContinuationData('data/G1/valid/', noise=1e-2 , resampling=args.resampling),
        'G1ne3': data.ContinuationData('data/G1/valid/', noise=1e-3 , resampling=args.resampling),
        'G1ne5': data.ContinuationData('data/G1/valid/', noise=1e-5 , resampling=args.resampling),
        'G2bse': data.ContinuationData('data/G2/valid/', noise=0.0  , resampling=args.resampling),
        'G2ne2': data.ContinuationData('data/G2/valid/', noise=1e-2 , resampling=args.resampling),
        'G2ne3': data.ContinuationData('data/G2/valid/', noise=1e-3 , resampling=args.resampling),
        'G2ne5': data.ContinuationData('data/G2/valid/', noise=1e-5 , resampling=args.resampling),
        'G3bse': data.ContinuationData('data/G3/valid/', noise=0.0  , resampling=args.resampling),
        'G3ne2': data.ContinuationData('data/G3/valid/', noise=1e-2 , resampling=args.resampling),
        'G3ne3': data.ContinuationData('data/G3/valid/', noise=1e-3 , resampling=args.resampling),
        'G3ne5': data.ContinuationData('data/G3/valid/', noise=1e-5 , resampling=args.resampling),
        'G4bse': data.ContinuationData('data/G4/valid/', noise=0.0  , resampling=args.resampling),
        'G4ne2': data.ContinuationData('data/G4/valid/', noise=1e-2 , resampling=args.resampling),
        'G4ne3': data.ContinuationData('data/G4/valid/', noise=1e-3 , resampling=args.resampling),
        'G4ne5': data.ContinuationData('data/G4/valid/', noise=1e-5 , resampling=args.resampling)
    }
    for metric in metrics_dict:
        if not os.path.exists(f'results/BEST_{metric}'):
            os.mkdir(f'results/BEST_{metric}')
    ##############

    model = train(args, device, train_set, metrics=metrics_dict)
        