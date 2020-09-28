#!/usr/bin/env python3
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
    'data': 'B',
    'noise': 1e-5,
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
    'optimizer': "adam",
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
    'valid_fraction': 0.05,
    'metric_batch_size': 64,
    'rescale': False,
    'beta': [20.0],
    'plot': False,
    'standardize': False,
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


def train(args, device, train_set, valid_set, loss, metric_list=None):
    if USE_WANDB: 
        run = wandb.init(project="temperature_rescaling", entity="deep_continuation", reinit=True)
        wandb.config.update(args)

    # datasets
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

    # model
    model = MLP(args).to(device)
    if args.initw:
        model.apply(init_weights)
    if USE_WANDB: 
        wandb.watch(model)
        model_insights = {
            'mlp_depth': len(args.layers) - 1,
            'mlp_width': max(args.layers[1:-1]),
            'mlp_narrow': min(args.layers[1:-1]),
            'mlp_neck': np.argmin(args.layers[1:-1]),
            'mlp_belly': np.argmax(args.layers[1:-1]),
        }
        wandb.config.update(model_insights)


    # optimizer
    if args.optimizer in ["adam", "Adam"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer in ["sgd", "SGD"]:
        optimizer = torch.optim.SGD(model.parameters(), lr=100*args.lr, weight_decay=args.weight_decay)
    else:
        ValueError(f"Unknown {args.optimizer} optimizer")

    # training loss
    criterion = loss

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
        if args.schedule:
            scheduler.step(model.avg_train_loss)
        
        early_stop_count -= 1
        if model.avg_valid_loss < best_valid_loss:
            early_stop_count = args.stop
            
            best_valid_loss = model.avg_valid_loss
            if USE_WANDB:
                torch.save(
                    model.state_dict(),
                    os.path.join(wandb.run.dir, f"best_valid_loss_model.pt")
                )
        
        model_chkpt = deepcopy(model)
        for metric in metric_list:
            is_best = metric.evaluate(model_chkpt, device, fraction=args.valid_fraction)
            if is_best:
                early_stop_count = args.stop
            metric.print_results()
         

        if USE_WANDB: 
            dict_to_log = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]['lr'],
                "train_loss": model.avg_train_loss,
                "valid_loss": model.avg_valid_loss,
            }
            for metric in metric_list:
                for lname, lvalue in metric.loss_values.items():
                    m_name = f"{lname}_{metric.name}"
                    dict_to_log[m_name] = lvalue
                    wandb.run.summary[m_name] = metric.best_losses[lname]
                    wandb.run.summary[f"epoch_{m_name}"] = metric.best_models[lname].epoch
            wandb.log(dict_to_log)

        if early_stop_count == 0:
            print('early stopping limit reached!!')
            break

    print('final_evaluation')
    for metric in metric_list:
        for lname, lvalue in metric.loss_values.items():
            tmp_model = metric.best_models[lname]
            tmp_model.eval()
            metric.evaluate(tmp_model, device, fraction=1.0)
            metric.print_results()
            
            if USE_WANDB:
                m_name = f"{lname}_{metric.name}"
                wandb.run.summary[m_name] = metric.loss_values[m_name]
                wandb.run.summary[f"epoch_{m_name}"] = tmp_model.epoch

    if USE_WANDB:
        run.join()

    return model


def main():
    args = utils.parse_file_and_command(default_parameters, help_strings)

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

    path_dict = {
        'F': 'data/Fournier/valid/',
        'G': 'data/G1/valid/',
        'B': 'data/B1/valid/',
    }

    train_set = data.ContinuationData(
        path_dict[args.data],
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="F" else 20
    )
    valid_set = data.ContinuationData(
        path_dict[args.data],
        beta=args.beta,
        noise=args.noise,
        rescaled=args.rescale,
        standardize=args.standardize,
        base_scale=15 if args.data=="F" else 20
    )

    loss_dict = {
        'mse': nn.MSELoss(), 
        'dcs': nn.L1Loss(), 
        'mae': dc_square_error, 
        'dca': dc_absolute_error,
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
        dataset = data.ContinuationData(path, base_scale=15 if p=="F" else 20)
        for n, noise in noise_dict.items():
            for b, beta, in beta_dict.items():
                for s, scale in scale_dict.items():
                    print(f"loading metric {p+s+n+b}")
                    metric_list.append(data.Metric(
                        name = f"{p+n+b+s}",
                        dataset=dataset,
                        loss_dict=loss_dict,
                        noise=noise,
                        beta=beta,
                        scale=scale,
                        std=args.standardize,
                        bs=args.metric_batch_size
                    ))
    
    train(args, device, train_set, valid_set, loss, metric_list=metric_list)
    

if __name__ == "__main__":
    main()
