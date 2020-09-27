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
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import wandb
    USE_WANDB = True
except ModuleNotFoundError:
    USE_WANDB = False

from deep_continuation import data
from deep_continuation import utils

TORCH_MAX = torch.finfo(torch.float64).max


default_parameters = {
    'data': 'Fournier',
    'noise': 1e-4,
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
    'weight_decay': 0,
    'smoothing': 1.0,
    'stop': 40,
    'warmup': True,
    'schedule': True,
    'factor': 0.4,
    'patience': 6,
    'seed': int(time.time()),
    'num_workers': 4,
    'cuda': True,
    'valid_fraction': 0.3,
    'rescale': False,
    'beta': [20.0],
    'plot': False,
    'standardize': False,
    'wandb': False,
}

help_strings = {
    'data': 'path to the SigmaRe.csv and Pi.csv will be "data/{data}/train/"',
    'noise': 'noise to the matsubara spectra',
    'loss': 'loss function to be used (see the code to find all possibilities)',
    'batch_size': 'batch size for dataloaders',
    'epochs': 'number of epochs to train.',
    'layers': 'sequence of dimensions for the neural net, includes input and output, e.g. --layers 128 400 600 512',
    'out_unit': 'select the output unit; "None", "ReLU"',
    'lr': 'initial learning rate',
    'dropout': 'dropout probability on all layers but the last one',
    'batchnorm': 'apply batchnorm (after ReLU) on all layers but the last one',
    'weight_decay': 'L2 regularization factor passed to the Adam optimizer',
    'stop': 'early stopping limit (number of epochs allowed without improvement)',
    'warmup': 'activate linear increase of the learning rate in the first epoch',
    'schedule': 'Turn on the learning rate scheduler (plateau,',
    'factor': 'scheduler factor at plateau',
    'patience': 'scheduler plateau (number of epochs without improvement triggering reduction of lr)',
    'seed': 'seed for the random generator number (time.time() if unspecified)',
    'num_workers': 'number of workers used in the dataloaders',
    'cuda': 'enables CUDA',
    'rescale': 'use rescaled output (requires extra target file with suffix)',
    'betas': "list of extra temperatures to use randomly (requires extra input data files with suffix)"
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

USE_WANDB = (args.wandb and USE_WANDB) 


class Normalizer(nn.Module):
    def __init__(self, dim=-1, norm=np.pi/40):
        super().__init__()
        self.dim = dim
        self.norm = norm
    def forward(self, x):
        N = x.shape[self.dim]
        return torch.renorm(x, p=1, dim=self.dim, maxnorm=N*self.norm)

class RenormSoftmax(nn.Module):
    def __init__(self, dim=-1, norm=np.pi/40):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.dim = dim
        self.norm = norm
    def forward(self, x):
        N = x.shape[self.dim]
        return self.softmax(x) *N*self.norm
    

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
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
        self.layers.append(nn.Linear(sizeA, args.layers[-1]))

        if args.out_unit == 'None':
            pass
        elif args.out_unit in ['ReLU', 'relu']:
            self.layers.append(nn.ReLU())
        elif args.out_unit in ['Softmax', 'softmax']:
            self.layers.append(RenormSoftmax())
        elif args.out_unit in ['Normalizer', 'normalizer']:
            self.layers.append(Normalizer())
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


def mse_smooth(outputs, targets, factor=1):
    ''' mean square error '''
    return torch.mean((outputs-targets)**2) + torch.mean(factor*torch.abs(outputs[:,1:]-outputs[:,:-1]))


def dc_absolute_error(outputs, targets):
    ''' computes the 0th component difference (DC conductivity)'''
    return torch.mean(torch.abs(outputs[:, 0]-targets[:, 0]))


def dc_square_error(outputs, targets):
    ''' computes the 0th component square difference (DC conductivity)'''
    return torch.mean((outputs[:, 0]-targets[:, 0])**2)


class Metric():
    def __init__(self, name, data_set, loss_list=['mse', 'mae', 'dcs', 'dca'], batch_size=512):
        self.valid_loader = torch.utils.data.DataLoader(
            data_set, batch_size=batch_size, drop_last=True, shuffle=False)
        self.name = name
        self.batch_size = batch_size

        self.loss_list = loss_list
        self.loss_value = {lname: 0 for lname in loss_list}
        self.best_loss = {lname: TORCH_MAX for lname in loss_list}
        self.best_model = {lname: None for lname in loss_list}

        self.loss = {}
        for lname in self.loss_list:
            if lname == "mse":
                self.loss[lname] = nn.MSELoss()
            elif lname == "mae":
                self.loss[lname] = nn.L1Loss()
            elif lname == 'dcs':
                self.loss[lname] = dc_square_error
            elif lname == 'dca':
                self.loss[lname] = dc_absolute_error
            elif lname == "kld":
                self.loss[lname] = nn.KLDivLoss()
            else:
                raise ValueError(f'Unknown loss function "{lname}"')

    def evaluate(self, model, device, save_best=False, fraction=0.3):
        for lname in self.loss_list:
            self.loss_value[lname] = 0

        stop_at_batch = fraction*len(self.valid_loader)//self.batch_size+1
        batch_count = 0
        for batch_number, (inputs, targets) in enumerate(self.valid_loader):
            if batch_number == stop_at_batch:
                break
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            for lname in self.loss_list:
                loss = self.loss[lname](outputs, targets).item()
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
                if save_best and USE_WANDB:
                    torch.save(
                        model.state_dict(),
                        os.path.join(wandb.run.dir, f"{lname}_{self.name}.pt")
                    )
                is_best = True
        
        return is_best

    def print_results(self):
        print(f'      {self.name}: ', end='')
        for lname in self.loss_list:
            print(f' {self.loss_value[lname]:.9f} ({lname})  ', end='')
        print()


def train(args, device, train_set, valid_set, metrics=None):
    if USE_WANDB: 
        run = wandb.init(project="mlp", entity="deep_continuation", reinit=True)
        wandb.config.update(args)
        # wandb.save("*.pt")  # will sync .pt files as they are saved

    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True, 
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True, 
        drop_last=True
    )

    # metrics (validation losses)
    loss_list = ['mse', 'dcs', 'mae', 'dca']
    metric_list = [
        Metric(name, dataset, loss_list=loss_list)
        for name, dataset in metrics.items()
    ]

    # model
    model = MLP(args).to(device)
    if args.initw:
        model.apply(init_weights)
    if USE_WANDB: 
        wandb.watch(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loss
    if  args.loss == "mse":
        criterion = nn.MSELoss()
    elif  args.loss == "mae":
        criterion = nn.L1Loss()
    elif  args.loss == 'dcs':
        criterion = dc_square_error
    elif  args.loss == 'dca':
        criterion = dc_absolute_error
    elif  args.loss == 'mss':
        criterion = lambda o, t: mse_smooth(o, t, factor=args.smoothing)
    elif  args.loss == "kld":
        criterion = nn.KLDivLoss()
    elif hasattr(train_set, 'custom_loss'):
        criterion = train_set.custom_loss(args.loss)
    else:
        raise ValueError(f'Unknown loss function "{args.loss}"')
    
    # lr scheduler
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=args.factor, 
            patience=args.patience,
            verbose=True,
            min_lr=1e-10
        )

    early_stop_count = args.stop
    best_valid_loss = TORCH_MAX
    for epoch in range(1, args.epochs+1):
        print(f' epoch {epoch}')
        model.epoch = epoch
        model.train()
        model.avg_train_loss = 0
        train_n_iter = 0

        if args.warmup and epoch == 1:
            print('   linear warm-up of learning rate')
        for batch_number, (inputs, targets) in enumerate(train_loader):
            if args.warmup and epoch == 1:
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

        model.eval()
        model.avg_valid_loss = 0
        valid_n_iter = 0
        for batch_number, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            model.avg_valid_loss += loss.item()
            valid_n_iter += 1

            if batch_number == 0 and args.plot:
                plt.clf()
                plt.plot(outputs[0].detach().numpy())
                plt.plot(targets[0].detach().numpy())
                plt.title(f"o:{outputs[0].sum().detach().numpy()}, t:{targets[0].sum().detach().numpy()}")
                plt.pause(0.001)
                
        
        model.avg_valid_loss = model.avg_valid_loss/valid_n_iter
        print(f'   valid loss: {model.avg_valid_loss:.9f}')
        
        early_stop_count -= 1
        if model.avg_valid_loss < best_valid_loss:
            avg_valid_loss = model.avg_valid_loss
            early_stop_count = args.stop
            if USE_WANDB:
                torch.save(
                    model.state_dict(),
                    os.path.join(wandb.run.dir, f"best_valid_loss_model.pt")
                )
        for metric in metric_list:
            is_best = metric.evaluate(model, device, save_best=False, fraction=args.valid_fraction)
            if is_best:
                early_stop_count = args.stop
            metric.print_results()
        
        if args.schedule:
            scheduler.step(model.avg_train_loss)

        if USE_WANDB: 
            dict_to_log = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]['lr'],
                "train_loss": model.avg_train_loss,
                "valid_loss": model.avg_valid_loss,
            }
            for metric in metric_list:
                for lname in loss_list:
                    m_name = f"{lname}_{metric.name}"
                    dict_to_log[m_name] = metric.loss_value[lname]
                    wandb.run.summary[m_name] = metric.best_loss[lname]
                    wandb.run.summary[f"epoch_{m_name}"] = metric.best_model[lname].epoch
            wandb.log(dict_to_log)

        if early_stop_count == 0:
            print('early stopping limit reached!!')
            break

    print('final_evaluation')
    best_epoch_dict = []
    for metric in metric_list:
        for lname in loss_list:
            tmp_model = metric.best_model[lname]
            tmp_model.eval()
            metric.evaluate(tmp_model, device, save_best=False, fraction=1.0)
            metric.print_results()
            
            if USE_WANDB:
                rec_name = f"{lname}_{metric.name}"
                wandb.run.summary[rec_name] = metric.loss_value[lname]
                wandb.run.summary[f"epoch_{rec_name}"] = tmp_model.epoch

    if USE_WANDB:
        run.join()

    return model


def main():
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
        'F': 'data/Fournier/valid/',
        # 'G': 'data/G1/valid/',
        # 'B': 'data/B1/valid/',
    }
    scale_dict = {
        'N': False,
        # 'R': True
    }
    noise_dict = {
        '0': 0,
        '5': 1e-5,
        '3': 1e-3,
        '2': 1e-2,
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
                    print(f"loading metric: {p+n+b+s}")
                    metrics_dict[p+n+b+s] = data.ContinuationData(
                        path,
                        noise=noise,
                        beta=beta,
                        rescaled=scale,
                        standardize=args.standardize,
                        base_scale=15 if p=="F" else 20
                    )

    model = train(args, device, train_set, valid_set, metrics=metrics_dict)
    return model


if __name__ == "__main__":
    main()
