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

### Reza's data (see the C++ code to generate it)

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

def mesh(N=512,L=20):
    ints = torch.arange(N).float()
    freqs = (ints/(N-1))**2*L
    return freqs

def mesh_deltas(N=512,L=20, analytic=True):
    ints = torch.arange(N).float()
    if analytic:
        deltas = 2*ints/((N-1)**2)*L #
    else:    
        prev_freqs = ((ints-1)/(N-1))**2*L
        next_freqs = ((ints+1)/(N-1))**2*L
        deltas = (next_freqs - prev_freqs)/2    

    deltas[0] = deltas[0]/2
    deltas[-1] = deltas[-1]/2
    return deltas


if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    # %% test squared mesh
    freqs  = mesh()
    deltas = mesh_deltas()
    deltas_f = mesh_deltas(analytic=False)
    plt.plot(freqs.numpy(), np.zeros(512), '.')
    plt.plot(freqs.numpy(), deltas.numpy()-deltas_f.numpy(), '.')
    plt.show()
    print(deltas.sum())

    #%% test RezaDataset
    dataset = RezaDataset('../sdata/')

    #%%
    for ii in range(10):
        plt.plot(dataset[ii][0])
    plt.show()

    for ii in range(10):
        chi0 = dataset[ii][0][0]
        plt.plot(freqs.numpy(),dataset[ii][1]*(2/chi0*np.pi))
    plt.show()

    #%% test makeloaders

    train_loader, valid_loader = make_loaders('../sdata/', 64)
    #%%
    for batch_number, (inputs, targets) in enumerate(train_loader):
        chi0s = inputs[:,0].float().view(-1,1)
        spectra = targets.float()
        print(torch.sum((2/chi0s/np.pi)*(spectra*deltas),dim=-1))

    #%%
    for batch_number, (inputs, targets) in enumerate(train_loader):
        batch_size = len(inputs)
        for i in range(10,batch_size):
            # spectra are normalized at chi0
            matsubara = inputs[i].float()
            plt.plot(matsubara.numpy())
            plt.show()
            chi0 = matsubara[0]
            print(chi0)
            spectrum = targets[i].float()
            plt.plot(freqs.numpy(), (deltas*spectrum*(2/chi0/np.pi)).numpy())
            plt.show()
            
            print(torch.sum(deltas*spectrum*(2/chi0/np.pi)))
            break
        break
