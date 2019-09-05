#
#   deep_continuation
#
#   © Simon Verret
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

class RezaDataset(Dataset):
    def __init__(self, path):
        self.x_data = np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")
        self.y_data = np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=",")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    
    dataset = RezaDataset('../Database/Training/')
    
    for ii in range(10):
        plt.plot(dataset[ii][0])
    plt.show()
    
    for ii in range(10):
        plt.plot(dataset[ii][1])
    plt.show()