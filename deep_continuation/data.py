#!/usr/bin/env python
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent


class ContinuationData(torch.utils.data.Dataset):
    def __init__(self, path, noise=0.0, beta=[], rescaled=False, standardize=False, base_scale=20.0):
        self.x_data = np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")
        self.y_data = np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=",")
        self.N = self.y_data.shape[-1]
        self.noise = noise
        self.standardize = standardize
        
        self.beta = beta
        if self.beta:
            self.xT_data = [self.x_data]
            for t in beta[1:]:
                self.xT_data.append(
                    np.loadtxt(open(path+f"Pi_beta_{t}.csv", "rb"), delimiter=",")
                )
            self.xT_data = np.stack(self.xT_data)
            self.nT = len(self.xT_data)

        self.base_scale = base_scale
        self.rescaled = rescaled
        if self.rescaled:
            scaled_path = path+"SigmaRe_scaled_4.0.csv"
            self.scaled_y_data = np.loadtxt(open(scaled_path, "rb"), delimiter=",")
            self.wmaxs = np.loadtxt(open(path+"wmaxs.csv", "rb"), delimiter=",")

        if self.standardize:
            self.avg = self.x_data.mean(axis=-2)
            self.std = self.x_data.std(axis=-2)
            self.xTavg = self.xT_data.mean(axis=-2)
            self.xTstd = self.xT_data.std(axis=-2)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        if self.beta:
            b = np.random.randint(self.nT)
            x = self.xT_data[b, index]
            x += np.random.normal(0,1, size=x.shape)*self.noise
            if self.standardize:
                x = (x - self.xTavg[b])/self.xTstd[b]
        else:
            x = self.x_data[index] 
            x += np.random.normal(0,1, size=x.shape)*self.noise
            if self.standardize:
                x = (x - self.avg)/self.std

        if self.rescaled:
            y = self.wmaxs[index] * self.scaled_y_data[index] / 20.0
        else: 
            y = self.base_scale * self.y_data[index] / 20.0
        return x, y


def main():
    pass
    
if __name__ == "__main__":
    main()
