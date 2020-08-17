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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    use_wandb = True
except ModuleNotFoundError:
    use_wandb = False

from deep_continuation import data
from deep_continuation import utils

TORCH_MAX = torch.finfo(torch.float64).max


default_parameters = {
    'data': 'P1',
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
    'lr': 0.0008,
    'initw': True,
    'stop': 40,
    'warmup': True,
    'schedule': True,
    'factor': 0.4,
    'patience': 6,
    'seed': int(time.time()),
    'num_workers': 0,
    'cuda': True,
}

help_strings = {
    'data': "chooses which dataset to use for training",
    'noise': 'noise to the matsubara spectra',
    'loss': 'loss function to be used (see the code to find all possibilities)',
    'batch_size': 'batch size for dataloaders',
    'epochs': 'number of epochs to train.',
    'layers': 'sequence of dimensions for the neural net, includes input and output, e.g. --layers 128 400 600 512',
    'lr': 'initial learning rate',
    'stop': 'early stopping limit (number of epochs allowed without improvement)',
    'schedule': 'Turn on the learning rate scheduler (plateau,',
    'factor': 'scheduler factor at plateau',
    'patience': 'scheduler plateau (number of epochs without improvement triggering reduction of lr)',
    'seed': 'seed for the random generator number (time.time() if unspecified)',
    'num_workers': 'number of workers used in the dataloaders',
    'cuda': 'enables CUDA',
}


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        sizeA = args.layers[0]
        for sizeB in args.layers[1:]:
            self.layers.append(nn.Linear(sizeA, sizeB))
            self.layers.append(nn.BatchNorm1d(sizeB))
            self.layers.append(nn.ReLU())
            sizeA = sizeB
        # last layer
        self.layers.append(nn.Linear(sizeA, args.layers[-1]))

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


# GLOBAL PARAMETERS 
# Can be parsed from command line 
# for example:
#
#   python simpler_mlp.py --lr 0.001
#
def main():
    
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

    if use_wandb: 
        wandb.init(project="nrm_smpl_mlp", entity="deep_continuation")
        wandb.config.update(args)
        wandb.save("*.pt")  # will sync .pt files as they are saved

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


    train_set = data.ContinuationData(
        f'data/{args.data}/train/', 
        noise=args.noise, 
    )

    valid_set = data.ContinuationData(
        f'data/{args.data}/valid/',
        noise=args.noise,
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_set, 
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False
    )


    model = MLP(args).to(device)
    if args.initw:
        model.apply(init_weights)
    if use_wandb: 
        wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=args.factor, 
        patience=args.patience,
        verbose=True,
        min_lr=1e-6
    )

    print('training')
    early_stop_count = args.stop
    best = TORCH_MAX
    for epoch in range(1, args.epochs+1):
        print(f' epoch {epoch}')

        model.train()
        avg_train_loss = 0
        train_n_iter = 0
        
        if args.warmup and epoch==1:
            print('   linear warm-up of learning rate')
        for batch_number, (inputs, targets) in enumerate(train_loader):
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

            avg_train_loss += loss.item()
            train_n_iter += 1
        avg_train_loss = avg_train_loss/train_n_iter
        print(f'   train loss: {avg_train_loss:.9f}')

        model.eval()
        avg_valid_loss = 0
        valid_n_iter = 0
        for batch_number, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            avg_valid_loss += loss.item()
            valid_n_iter += 1
        avg_valid_loss = avg_valid_loss/valid_n_iter
        print(f'   validation loss: {avg_valid_loss:.9f}')

        model.eval()
        if use_wandb: 
            wandb.log({
                "epoch":epoch,
                "train loss": avg_train_loss,
                "valid loss": avg_valid_loss
            })

        scheduler.step(avg_train_loss)
        
        if avg_valid_loss < best:
            if use_wandb: 
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_weights.pt"))
                wandb.run.summary["train loss"] = avg_train_loss
                wandb.run.summary["valid loss"] = avg_valid_loss
            else:
                torch.save(os.path.join("results", "simpler_mlp_latest_weights.pt"))
            early_stop_count = args.stop
            best = avg_valid_loss
        else: 
            early_stop_count -= 1
        if early_stop_count==0:
            print('early stopping limit reached!!')
            break


if __name__ == "__main__":
    main()