import os
import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from fire import Fire

from deep_continuation import dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TORCH_MAX = torch.finfo(torch.float64).max
MODELS_PATH = os.path.join(dataset.HERE, "saved_models")

class ContinuationData(torch.utils.data.Dataset):
    def __init__(
        self, pi_path, sigma_path, noise=0.0, standardize=True, avg=None, std=None,
    ):
        self.x_data = np.load(pi_path)
        self.y_data = np.load(sigma_path)

        self.noise = noise
        self.standardize = standardize
        if avg is not None and std is not None:
            self.avg = avg
            self.std = std
        else:
            self.avg = self.x_data.mean(axis=-2)
            self.std = self.x_data.std(axis=-2)
        
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        x += np.random.normal(0,1, size=x.shape)*self.noise
        if self.standardize:
            x = (x - self.avg)/self.std

        y = self.y_data[index]
        return x, y


class MLP(nn.Module):
    def __init__(self, layers=[128, 952, 1343, 1673, 1722, 512]):
        super(MLP, self).__init__()
        self.epoch = 0
        
        self.layers = nn.ModuleList()
        sizeA = layers[0]
        for sizeB in layers[1:]:
            self.layers.append(nn.Linear(sizeA, sizeB))
            self.layers.append(nn.ReLU())
            sizeA = sizeB

        self.layers.append(nn.Linear(sizeA, layers[-1]))
        self.layers.append(nn.Softmax(dim=-1))
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def initialize_weights(module, gain=1):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight, gain=gain)
        torch.nn.init.zeros_(module.bias)


def train_mlp(
    # dataset options
    noise=0.0, 
    standardize=True, 
    # datafile options
    name=None,
    path=os.path.join(dataset.DATAPATH, "default"),
    num_std=1,
    num_beta=1,
    Nwn=128,
    beta=30,
    Nw=512,
    wmax=20,
    fixstd=False,
    # dataloader options
    lr = 8e-5,
    batch_size = 523,
    # model options
    layers=[128, 952, 1343, 1673, 1722, 512],
    # scheduler options
    factor=0.216, 
    patience=5, 
    min_lr=1e-10,
    # training loop options
    n_epochs = 200,
    early_stop_limit = 40,
    warmup = True,
):
    config_dict = locals()
    train_pi_path, train_sigma_path, train_set_id = dataset.get_dataset(
        size=100000, seed=55555, 
        name=name, path=path, num_std=num_std, num_beta=num_beta,
        Nwn=Nwn, beta=beta, Nw=Nw, wmax=wmax, fixstd=fixstd,
    )
    valid_pi_path, valid_sigma_path, valid_set_id = dataset.get_dataset(
        size=10000, seed=555, 
        name=name, path=path, num_std=num_std, num_beta=num_beta,
        Nwn=Nwn, beta=beta, Nw=Nw, wmax=wmax, fixstd=fixstd,
    )

    config_id = f"n{noise}"

    train_set = ContinuationData(
        pi_path=train_pi_path, sigma_path=train_sigma_path, noise=noise,
        standardize=standardize,
    )
    valid_set = ContinuationData(
        pi_path=valid_pi_path, sigma_path=valid_sigma_path, noise=noise,
        standardize=standardize,
        avg=train_set.avg, std=train_set.std,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    # model, loss, optimizer, scheduler
    model = MLP(layers=layers).to(device)
    model.apply(initialize_weights)
    loss_function = nn.L1Loss()  
    mse_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, min_lr=min_lr, verbose=True
    )

    # training loop
    best_valid_loss = TORCH_MAX
    loss_id = None
    early_stop_count = early_stop_limit
    for epoch in range(1, n_epochs+1):
        print(f' epoch {epoch}')
        avg_train_loss = 0
        avg_train_mse = 0

        model.train()
        for batch_number, (inputs, targets) in enumerate(train_loader):
            
            # learning rate warmup
            if warmup and epoch == 1: 
                tmp_lr = (batch_number+1)*lr/len(train_loader)
                for g in optimizer.param_groups:
                    g['lr'] = tmp_lr

            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            mse = mse_function(outputs, targets)
            avg_train_mse += mse.item()
            avg_train_loss += loss.item()
        
        avg_train_loss = avg_train_loss/len(train_loader)
        print(f'   train loss: {avg_train_loss:.9f}, mse: {avg_train_mse:.9f}')

        model.eval()
        avg_valid_loss = 0
        avg_valid_mse = 0
        for batch_number, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            mse = mse_function(outputs, targets)
            avg_valid_loss += loss.item()
            avg_valid_mse += mse.item()

        avg_valid_loss = avg_valid_loss/len(valid_loader)
        print(f'   valid loss: {avg_valid_loss:.9f}, mse:{avg_valid_mse:.9f}')
        
        scheduler.step(avg_train_loss)
        
        # saving and 
        early_stop_count -= 1
        if avg_valid_loss < best_valid_loss:
            early_stop_count = early_stop_limit
            best_valid_loss = avg_valid_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            
            if loss_id is not None:
                os.remove(model_path)
                os.remove(config_path)
            loss_id = f"noise{noise}_epoch{best_epoch}_mse{avg_valid_mse}_loss{best_valid_loss}"
            model_dir = os.path.join(MODELS_PATH, f"trained_on_{name}_{train_set_id}")
            model_dir = os.path.join(model_dir, f"validated_on_{name}_{valid_set_id}")
            model_path = os.path.join(model_dir, f"model_{loss_id}.pt")
            config_path = os.path.join(model_dir, f"config_{loss_id}.json")

            os.makedirs(model_dir, exist_ok=True)
            torch.save(best_model.state_dict(), model_path)
            with open(config_path, 'w') as fp:
                json.dump(config_dict, fp, indent=4) 
        
        if early_stop_count == 0:
            print('early stopping limit reached!!')
            print(f'best epoch was {best_epoch}')
            break


if __name__ == "__main__":
    Fire(train_mlp)