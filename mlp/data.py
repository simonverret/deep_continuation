#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

#%%

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class RezaDataset(Dataset):
    def __init__(self, path):
        self.x_data = np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")
        self.y_data = np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=",")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

def make_loaders(path, batch_size, num_workers=0):
    print("Loading data")
    dataset = RezaDataset(path)

    validation_split = .1
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split*len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=validation_sampler)
    return train_loader,valid_loader

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    
    dataset = RezaDataset('../sdata/')
    
    for ii in range(10):
        plt.plot(dataset[ii][0])
    plt.show()
    
    for ii in range(10):
        plt.plot(dataset[ii][1])
    plt.show()