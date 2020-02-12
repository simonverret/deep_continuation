#!/usr/bin/env python
#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

#%%
import os
import json
import random
import numpy as np
from scipy import integrate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt 

import utils

np.set_printoptions(precision=4)
SMALL = 1e-10



### Reza's data (see the C++ code to generate it)

# TODO: the dataset should have the "make_loader" function, and different
# datasets would lead to different weighted loss functions and measures (mesh)
# this should be handled internally by the dataset.
# the mainscript could simply ask the dataset if additional options are
# available as loss functions / 

class WgtLoss(nn.Module):
    """ Base class for weighted loss function """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        out = self.weights * self.loss(outputs, targets)
        out = torch.mean(out)
        return out

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

        if self.normalize:
            print('WARNING normalization untested')
            if self.normalize:
                self.y_data = self.y_data[:,:]/self.y_data[:,0]
                self.x_data = self.x_data[:,:]/self.y_data[:,0]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

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

    def plot(self, N):
        #%% plot some examples from the loaded dataset
        print('\nnormalization')
        print('sum =', sig_of_w_array.sum(axis=-1)*(2*10/512))
        print('pi0 =', pi_of_wn_array[:,0].real, '\n')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[4,4])
        ax1.set_xlabel('iwn')
        ax1.set_title('input (matsubara freqs)',loc='right', pad=-12)
        ax2.set_title('target (real freqs)',loc='right', pad=-12)

        for ii in range(N):
            ax1.plot( self[ii][0] )

            chi0 = self[ii][0][0]
            x = self.mesh.numpy()
            y = self[ii][1]*(2/chi0*np.pi)
            ax2.plot(x,y)
        plt.show()


class DataGenerator():
    def __init__(self, args):
        self.N_wn = args.in_size
        self.N_w  = args.out_size
        
        self.wn_list, self.w_list = self.set_freq_grid( args.w_max )
        self.w_grid, self.wn_grid = self.set_integration_grid( args.N_tail, args.power )
        
        self.scale = args.scale
        # self.w_list *= self.scale

        # default peaks characteristics
        self.gauss               = args.gauss
        self.drude_width_range   = np.array(args.drude_width) * self.scale
        self.peak_position_range = np.array(args.posit) * self.scale
        self.peak_width_range    = np.array(args.peak_width) * self.scale
        self.max_drude           = args.max_drude
        self.max_peaks           = args.max_peaks
        self.weight_ratio        = args.weight_ratio

    def peak(self, omega, center=0, width=1, height=1):
        if self.gauss:
            return (height/np.sqrt(np.pi)/width) * np.exp(-(omega-center)**2/width**2)
        else:
            return (height/np.pi) * width/( (omega-center)**2 + (width)**2 )

    def grid_integrand(self, omega, omega_n, c, w, h):
        spectralw = self.peak(omega, c, w, h).sum(axis=0)
        return omega**2 * spectralw/ (omega**2+omega_n**2)

    def set_freq_grid(self, w_max):
        self.wn_list = np.arange(0.0, self.N_wn, dtype=float)
        self.w_list  = np.arange(0.0, w_max, w_max/self.N_w, dtype=float)
        return self.wn_list, self.w_list

    def set_integration_grid(self, N_tail, tail_power):
        pos_w_list = self.w_list
        neg_w_list  = -np.flip(pos_w_list[1:])
        pos_tail = np.logspace(1, tail_power, N_tail)
        neg_tail = -np.flip(pos_tail)[:-1]
        
        full_w_list = [ neg_tail, neg_w_list, pos_w_list, pos_tail ]
        full_w_list = np.concatenate(full_w_list) + SMALL
        self.w_grid, self.wn_grid = np.meshgrid(full_w_list, self.wn_list)
        return  self.w_grid, self.wn_grid 

    def generate_batch(self, N):

        sig_of_w_array = np.zeros([ N, self.N_w ])
        pi_of_wn_array = np.zeros([ N, self.N_wn])

        for i in range(N):
            if (i==0 or (i+1)%(max(1,N//100))==0): print(f"sample {i+1}")

            # random spectrum characteristics
            num_drude    = np.random.randint( 0,     self.max_drude)
            num_peak     = np.random.randint( 1,     self.max_peaks)
            weight_ratio = np.random.uniform( SMALL, self.weight_ratio)
            
            # random initialization (center, width, height) of peaks
            min_c = self.peak_position_range[0]
            max_c = self.peak_position_range[1]
            c  = np.random.uniform( min_c, max_c, size=num_peak )
            w  = np.random.uniform( 0.0  , 1.000, size=num_peak )
            h  = np.random.uniform( 0.0  , 1.000, size=num_peak )
            
            # Drude peaks adjustments
            c[:num_drude]  = 0.0
            w[:num_drude] *= (self.drude_width_range[1] - self.drude_width_range[0])
            w[:num_drude] += self.drude_width_range[0] #min
            h[:num_drude] *= weight_ratio/( h[:num_drude].sum() + SMALL )
            
            # other peaks adjustments
            w[num_drude:] *= (self.peak_width_range[1] - self.peak_width_range[0])
            w[num_drude:] += self.peak_width_range[0] #min
            h[num_drude:] *= (1-weight_ratio)/( h[num_drude:].sum() + SMALL )
            
            #symmetrize and normalize
            c = np.hstack([c,-c])
            w = np.hstack([w, w])
            h = np.hstack([h, h])
            h *= 1/h.sum(axis=-1, keepdims=True)

            # compute spectra
            sig_of_w_array[i] = self.peak(
                                    self.w_list[np.newaxis,:], 
                                    c[:,np.newaxis], 
                                    w[:,np.newaxis], 
                                    h[:,np.newaxis] 
                                ).sum(axis=0)

            matsubaraGrid = self.grid_integrand( 
                                self.w_grid [ np.newaxis,:,: ], 
                                self.wn_grid[ np.newaxis,:,: ], 
                                c[ :, np.newaxis, np.newaxis ],
                                w[ :, np.newaxis, np.newaxis ], 
                                h[ :, np.newaxis, np.newaxis ] 
                            )

            pi_of_wn_array[i] = integrate.simps( matsubaraGrid[0], self.w_grid, axis=1)
        return pi_of_wn_array, sig_of_w_array

    def plot(self, pi_of_wn_array, sig_of_w_array):
        print('\nnormalization')
        print('sum =', sig_of_w_array.sum(axis=-1)*(2*10/512))
        print('pi0 =', pi_of_wn_array[:,0].real, '\n')
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[4,4])
        ax1.set_xlabel('iwn')
        ax1.set_title('input (matsubara freqs)',loc='right', pad=-12)
        ax2.set_title('target (real freqs)',loc='right', pad=-12)
        for i in range(len(pi_of_wn_array)):
            ax1.plot( self.wn_list, pi_of_wn_array[i] )
            ax2.plot( self.w_list, sig_of_w_array[i] )
        plt.show()

    def generate_dataset(self, N, path='./'):
        pi_wn_array, sig_of_w_array = self.generate_batch(N)
        np.savetxt( path + 'SigmaRe.csv', sig_of_w_array, delimiter=',')
        np.savetxt( path + 'Pi.csv'     , pi_wn_array   , delimiter=',')


if __name__ == '__main__':
    default_args = {
        # script parameters
        'test'         : False,
        'plot'         : 0,
        'generate'     : 0,
        'path'         : './',
        # data generation parameters
        'in_size'      : 128,
        'out_size'     : 512,
        'w_max'        : 10,
        'N_tail'       : 128,
        'power'        : 5,
        'scale'        : 1,
        # spectrum parameters
        'gauss'        : False,
        'max_drude'    : 4,
        'max_peaks'    : 6,
        'weight_ratio' : 0.20,
        'drude_width'  : [0.1, 0.5],
        'posit'        : [2.0, 6.0],
        'peak_width'   : [0.2, 1.0],
        'seed'         : 139874
    }
    args = utils.parse_file_and_command(default_args, {}, )

    np.random.seed(args.seed)
    sigma_path = args.path+'SigmaRe.csv'
    pi_path = args.path+'Pi.csv'

    print(sigma_path)
    print(pi_path)

    if not (os.path.exists(sigma_path) or os.path.exists(pi_path)):
        generator = DataGenerator(args)
        
        if args.generate > 0:
            generator.generate_dataset(N=args.generate, path=args.path)
            if args.plot > 0:
                print('WARNING: examples printed are not part of the dataset')

        if args.plot > 0:
            pi, sigma = generator.generate_batch(N=args.plot)
            generator.plot(pi, sigma)

    else:
        if args.generate > 0:
            raise ValueError('ABORT GENERATION: there is already a dataset on this path')

        #%% test dataset with ContinuationData
        my_dataset = ContinuationData(args.path)

        if args.plot > 0:
            my_dataset.plot(args.plot)


    if args.test:
        # %% test squared mesh
        meas   = my_dataset.measure
        meas_f = my_dataset.make_measure(analytic=False)
        plt.plot(my_dataset.mesh.numpy(), np.zeros(512), '.')
        plt.plot(my_dataset.mesh.numpy(), meas.numpy()-meas_f.numpy(), '.')
        plt.show()
        print(meas.sum())
        print(meas_f.sum())

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
