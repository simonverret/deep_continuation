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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt 

### Reza's data (see the C++ code to generate it)


# TODO: the dataset should have the "make_loader" function, and different
# datasets would lead to different weighted loss functions and measures (mesh)
# this should be handled internally by the dataset.
# the mainscript could simply ask the dataset if additional options are
# available as loss functions / 

class ContinuationData(Dataset):
    def __init__(self, path, measure=None, N=512, L=10, normalize=False):
        self.x_data = np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")
        self.y_data = np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=",")
        
        self.measure_name = measure
        self.N = N
        self.L = L
        self.normalize = normalize
        self.mesh = self.make_mesh()
        self.measure = self.make_measure()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        if self.normalize:
            norm = x[0] 
        else:
            norm = 1.
        return x/norm, y/norm

    def make_loaders(self, batch_size, num_workers=0, split=0.1):
        """ make pytorch dataloaders """

        print("Loading data")
        validation_split = split
        indices = list(range(len(self)))
        split = int(np.floor(validation_split*len(self)))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
        valid_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=validation_sampler)
        return train_loader,valid_loader

    def make_mesh(self):
        """ Returns the x at to which the data corresponds """
        
        ints = torch.arange(self.N).float()
        if self.measure_name == 'squared':
            mesh = (ints/(self.N-1))**2*self.L
        else:
            mesh = (ints/(self.N-1))*self.L
        return mesh

    def make_measure(self, analytic=True): 
        """ Returns the lenghts dx on which the data is defined """

        ints = torch.arange(self.N).float()
        if self.measure_name == 'squared':
            if analytic:
                meas = 2*ints/((self.N-1)**2)*self.L #
            else:    
                prev_freq = ((ints-1)/(self.N-1))**2*self.L
                next_freq = ((ints+1)/(self.N-1))**2*self.L
                meas = (next_freq - prev_freq)/2    
        else:
            if analytic:
                meas = torch.ones_like(ints)*self.L/(self.N-1)
            else:
                prev_freq = ((ints-1)/(self.N-1))*self.L
                next_freq = ((ints+1)/(self.N-1))*self.L
                meas = (next_freq - prev_freq)/2 
        meas[ 0] = meas[ 0]/2
        meas[-1] = meas[-1]/2
        return meas

    def custom_loss(self, loss_name):
        """ Select among custom weigted loss functions """
        
        # this functino could take additional parameters
        param=1e-3

        if loss_name == "expL1Loss":
            expL1Loss = WgtLoss()
            expL1Loss.weights = torch.exp(-self.mesh)
            expL1Loss.loss = lambda y1,y2: torch.abs(y1-y2)
            return expL1Loss
        
        elif loss_name == "invL1Loss":
            invL1Loss = WgtLoss()
            invL1Loss.weights = 1/(self.mesh + param)
            invL1Loss.loss = lambda y1,y2: torch.abs(y1-y2)
            return invL1Loss
        
        elif loss_name == "expMSELoss":
            expMSELoss = WgtLoss()
            expMSELoss.weights = torch.exp(-self.mesh)
            expMSELoss.loss = lambda y1,y2: (y1-y2)**2
            return expMSELoss

        elif loss_name == "invMSELoss":
            invMSELoss = WgtLoss()
            invMSELoss.weights = 1/(self.mesh + param)
            invMSELoss.loss = lambda y1,y2: (y1-y2)**2
            return invMSELoss

        else:
            raise ValueError('Unknown loss function "'+loss_name+'"')


class WgtLoss(nn.Module):
    """ Base class for weighted loss function """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        out = self.weights * self.loss(outputs, targets)
        out = torch.mean(out)
        return out


        
class Normalizer(nn.Module):
    """ Module intended as a custom non exponential Softmax that uses the proper 
    measure to normalize the data. Not working because needs to take the Measure
    into account."""

    def __init__(self, dim=-1):
        super(Normalizer, self).__init__()
        self.relu = nn.ReLU()
        self.dim = dim

    def forward(self, q):
        q = self.relu(q)
        return q/torch.sum(q, dim=self.dim, keepdim=True).detach()

    # to use it
    # elif args.out_unit == 'Normalize': 
        # self.layers.append( data.Normalizer() )




if __name__ == '__main__':

    #%% test dataset with ContinuationData
    my_dataset = ContinuationData('../sdata/short/')

    # %% test squared mesh
    meas   = my_dataset.measure
    meas_f = my_dataset.make_measure(analytic=False)
    plt.plot(my_dataset.mesh.numpy(), np.zeros(512), '.')
    plt.plot(my_dataset.mesh.numpy(), meas.numpy()-meas_f.numpy(), '.')
    plt.show()
    print(meas.sum())
    print(meas_f.sum())

    #%% plot some examples
    for ii in range(10):
        y = my_dataset[ii][0]
        plt.plot(y)
    plt.show()

    for ii in range(10):
        chi0 = my_dataset[ii][0][0]
        x = my_dataset.mesh.numpy()
        y = my_dataset[ii][1]*(2/chi0*np.pi)
        plt.plot(x,y)
    plt.show()

    #%% test makeloaders & normalization
    train_loader, valid_loader = my_dataset.make_loaders(batch_size=64)
    for batch_number, (inputs, targets) in enumerate(train_loader):
        spectra = targets.float()
        print(torch.sum(2*(spectra*my_dataset.measure),dim=-1)[0] )

    # NOTE: as one can see, the dataset spectra are roughtly normalized to one, note
    # also that this normalization is not guaranteed, because we do not use the full
    # domain of omega used to generate the spectrum and because the tails are very 
    # important


    #%% test Reza with ContinuationData
    reza_dataset = ContinuationData('../rdata/part/', measure='squared', L=20)

    # %% test measure for squared
    meas   = reza_dataset.measure
    meas_f = reza_dataset.make_measure(analytic=False)
    plt.plot(reza_dataset.mesh.numpy(), np.zeros(512), '.')
    plt.plot(reza_dataset.mesh.numpy(), meas.numpy()-meas_f.numpy(), '.')
    plt.show()
    print(meas.sum())

    #%%
    for ii in range(1,20):
        y = reza_dataset[ii][0]
        plt.plot(y)
    plt.show()

    for ii in range(1,20):
        x = reza_dataset.mesh.numpy()
        y = reza_dataset[ii][1]
        # chi0 = reza_dataset[ii][0][0]
        # y *= (2/chi0*np.pi)
        plt.plot(x,y)
    plt.show()

    #%% test makeloaders
    r_train_loader, r_valid_loader = reza_dataset.make_loaders(batch_size=64)
    for batch_number, (inputs, targets) in enumerate(r_train_loader):
        chi0s = inputs[:,0].float().view(-1,1)
        spectra = targets.float()
        print(torch.sum((2/chi0s/np.pi)*(spectra*meas),dim=-1)[0] )

    # NOTE: as one can see, the dataset spectra are roughly normalized 
    # to 2*chi(w_n=0)/pi

    #%% 
    for batch_number, (inputs, targets) in enumerate(r_train_loader):
        batch_size = len(inputs)
        for i in range(10,batch_size):
            # spectra are normalized at chi0
            matsubara = inputs[i].float()
            plt.plot(matsubara.numpy())
            plt.show()
            chi0 = matsubara[0]
            print(chi0)
            spectrum = targets[i].float()
            
            x = reza_dataset.mesh.numpy()
            y = (reza_dataset.measure*spectrum*(2/chi0/np.pi)).numpy()
            plt.plot(x,y)
            plt.show()
            
            print(torch.sum(meas*spectrum*(2/chi0/np.pi)))
            break
        break
