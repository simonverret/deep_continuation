#!/usr/bin/env python
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
TORCH_MAX = torch.finfo(torch.float64).max

# TODO: make something more general purpose; this dataset is very specific for the temperature exploration

class ContinuationData(torch.utils.data.Dataset):
    def __init__(self, path, noise=0.0, beta=[20.0], rescaled=False, standardize=False, base_scale=20.0):
        self.noise = noise
        self.beta = beta
        self.rescaled = rescaled
        self.standardize = standardize
        self.base_scale = base_scale

        self.fullBetaList = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 50.0]        
        self.x_data = {2.0: np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")}
        for b in self.fullBetaList :
            self.x_data[b] = np.loadtxt(open(path+f"Pi_beta_{b}.csv", "rb"), delimiter=",")
        self.avg = {b: self.x_data[b].mean(axis=-2) for b in self.fullBetaList}
        self.std = {b: self.x_data[b].std(axis=-2) for b in self.fullBetaList}

        self.y_data = {
            'N': np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=","),
            'R': np.loadtxt(open(path+"SigmaRe_scaled_4.0.csv", "rb"), delimiter=",")
        }
        self.N = self.y_data['N'].shape[-1]
        self.wmaxs = np.loadtxt(open(path+"wmaxs.csv", "rb"), delimiter=",")

    def __len__(self):
        return len(self.x_data[20.0])

    def __getitem__(self, index):
        b = np.random.choice(self.beta)
        x = self.x_data[b][index]
        x += np.random.normal(0,1, size=x.shape)*self.noise
        if self.standardize:
            x = (x - self.avg[b])/self.std[b]
        
        if self.rescaled:
            y = self.wmaxs[index] * self.y_data['R'][index] / 20.0
        else: 
            y = self.base_scale * self.y_data['N'][index] / 20.0
        return x, y


class EvaluationMetric():
    def __init__(self, name, dataset, loss_dict, noise, beta, scale, std, bs, num_workers):
        self.name = name
        
        self.batch_size = bs
        self.dataset = dataset
        self.noise = noise
        self.beta = beta
        self.scale = scale
        self.std = std
        self.bs = bs
        self.num_workers = num_workers
        
        self.loss_dict = loss_dict
        self.loss_values = {lname: TORCH_MAX for lname in loss_dict.keys()}
        self.best_losses = {lname: TORCH_MAX for lname in loss_dict.keys()}
        self.best_models = {lname: None for lname in loss_dict.keys()}

    def evaluate(self, model, device, fraction=0.3):
        self.dataset.noise = self.noise
        self.dataset.beta = self.beta
        self.dataset.rescaled = self.scale
        self.dataset.standardize = self.std
        loader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.bs,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )

        for lname in self.loss_dict.keys():
            self.loss_values[lname] = 0
        batch_count = 0
        stop_at_batch = max(int(fraction*len(loader)),1)
        for batch_number, (inputs, targets) in enumerate(loader):
            if batch_number == stop_at_batch: break
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
        
            for lname, loss in self.loss_dict.items():
                self.loss_values[lname] += loss(outputs, targets).item()
            batch_count += 1
        
        for lname in self.loss_dict.keys():
            self.loss_values[lname] = self.loss_values[lname]/ batch_count

        is_best = False
        for lname in self.loss_dict.keys():
            if self.loss_values[lname] < self.best_losses[lname]:
                self.best_losses[lname] = self.loss_values[lname]
                self.best_models[lname] = model
                is_best = True
        return self.loss_values, is_best

    def print_results(self):
        print(f'      {self.name}: ', end='')
        for lname in self.loss_dict.keys():
            print(f' {self.loss_values[lname]:.9f} ({lname})  ', end='')
        print()


def main():
    pass
    
if __name__ == "__main__":
    main()
