#!/usr/bin/env python
#
#   deep_continuation
#
#   Simon Verret
#   Quinton Weyrich
#   Reza Nourafkan
#   Andre-Marie Tremblay
#

#%%
import os
import time
import random
import numpy as np
from scipy import integrate
from scipy.special import binom, erf


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt 
from matplotlib import rc, rcParams
# rc('text', usetex=True)
# rc('axes', linewidth=0.5)
# rc('xtick.major', width=0.5)
# rc('ytick.major', width=0.5)
rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}',
        r'\usepackage{amssym}',
    ]

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
    def __init__(self, path, N=512, L=10, noise=0.0, measure=None, normalize=False, resampling='default'):
        self.x_data = np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")
        if resampling == 'cbrt':
            self.y_data = np.loadtxt(open(path+"SigmaRe_cbrtScale.csv", "rb"), delimiter=",")
        elif resampling == 'sqrt':
            self.y_data = np.loadtxt(open(path+"SigmaRe_sqrtScale.csv", "rb"), delimiter=",")
        elif resampling == 'default':
            self.y_data = np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=",")
        else:
            raise ValueError(f'resampling value {resampling} could not be read must be either "cbrt", "sqrt" or "default"')
        self.noise = noise

        self.measure_name = measure
        self.N = N
        self.L = L
        self.normalize = normalize
        self.mesh = self.make_mesh()
        self.measure = self.make_measure()

        if self.normalize:
            print('WARNING normalization untested')
            if self.normalize:
                self.y_data = self.y_data[:,:]/self.x_data[:,:1]
                self.x_data = self.x_data[:,:]/self.x_data[:,:1]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = self.x_data[index] 
        x += np.random.normal(0,1, size=x.shape)*self.noise
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

        train_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, shuffle=True)
        valid_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers, sampler=validation_sampler, shuffle=True)
        return train_loader,valid_loader

    def single_loader(self, batch_size=0, num_workers=0):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    def make_mesh(self):
        """ Returns the x at to which the data corresponds """
        ints = torch.arange(self.N).float()
        if self.measure_name == 'squared':
            mesh = (ints/(self.N-1))**2*self.L
        else:
            mesh = (ints/(self.N-1))*self.L
        return mesh

    def make_measure(self, analytic=True): 
        """ Returns the lengths dx on which the data is defined """
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
        print('sum =', self.y_data[:N].sum(axis=-1)*(2*10/512))
        print('pi0 =', self.x_data[:N,0].real, '\n')

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


def gaussian(x, c, w, h):
    return (h/(np.sqrt(np.pi)*w))*np.exp(-((x-c)/w)**2)


def lorentzian(x, c, w, h):
    return (h/np.pi)*w/((x-c)**2+w**2)


def bernstein(x, m, n):
    return binom(m,n) * (x**n) * ((1-x)**(m-n)) * (x>=0) * (x<=1)


def free_bernstein(x, m, n, c=0, w=1, h=1):
    sq = np.sqrt(m+1)
    xx = (x-c)/(w*sq) + n/m
    return (h*sq/w)*bernstein(xx, m, n)


def peak(omega, center=0, width=1, height=1, type_m=0, type_n=0):
    out = 0
    out += (type_m == 0) * lorentzian(omega, center, width, height)
    out += (type_m == 1) * gaussian(omega, center, width, height)
    out += (type_m >= 2) * free_bernstein(omega, type_m, type_n, center, width, height)
    return out


def peak_sum(omega, c, w, h, m, n):
        return peak(
            x[np.newaxis, :],
            c[:, np.newaxis],
            w[:, np.newaxis],
            h[:, np.newaxis],
            m[:, np.newaxis],
            n[:, np.newaxis]
        ).sum(axis=0)


def test_peak():
    x = np.linspace(-1,2,1000)
    c = np.array([-0.2, 0.3, 0.6, -0.9])
    w = np.array([ 0.5, 0.5, 1.2,  0.8])
    h = np.array([ 1/10, 2/10, 3/10, 4/10])
    m = np.array([ 99, 99, 5, 5 ])
    n = np.array([98, 2, 2, 3])

    bs = peak_sum(x, c, w, h, m, n)
    plt.plot(x, bs, lw=2, color='black')


class DataGenerator():
    def __init__(self, args):
        self.N_wn = args.in_size        # Number of input datapoints
        self.N_w  = args.out_size       # Number of output datapoints

        self.wn_list, self.w_list = self.set_freq_grid( args.w_max, args.beta )
        self.w_grid, self.wn_grid = self.set_integration_grid( args.w_max, args.tail_power, args.N_tail )
        
        self.w_max               = args.w_max       # These are all hyperparameters
        self.Pi0                 = args.Pi0
        self.beta                = args.beta
        self.sqrt_ratio          = args.sqrt_ratio
        self.cbrt_ratio          = args.cbrt_ratio
        self.normalize           = args.normalize   # True or false flag for normalization
        # default peaks characteristics
        self.bernstein           = args.bernstein     # True or false flag for the use of Bernstein polynomials instead of Gaussians
        self.max_drude           = args.max_drude
        self.max_peaks           = args.max_peaks
        self.weight_ratio        = args.weight_ratio
        self.w_volume            = 2*self.w_max/(np.pi*self.N_w)            
        self.drude_width_range   = np.array(args.drude_width)*args.w_max
        self.peak_position_range = np.array(args.peak_pos)*args.w_max
        self.peak_width_range    = np.array(args.peak_width)*args.w_max
        # Lorentzian-specific characteristics
        self.lorentz             = args.lorentz          # True or false flag for the use of Lorentzians instead of Gaussians
        self.lor_peaks           = int(1000)             # Number of Lorentzian peaks
        self.lor_width           = 0.5                   # Width of Lorentzian peaks 
        self.N_seg               = 8                     # The number of terms to include in the peak distribution functions

    def peak(self, omega, center=0, width=1, height=1, type_m=0, type_n=0):
        if self.lorentz:
            return ((height/(np.pi * omega + SMALL)) * (width/( (omega-center)**2 + (width)**2) - width/( (omega+center)**2 + (width)**2))) + (omega == 0) * 4 * height * width * center / (np.pi * (center**2 + width**2)**2) 
            # Define peak function to be a Lorentzian if flag set to true. These Lorentzians are by design symmetrical, and have the
            # centers in the denominator to cancel out the omega in the numerator of the integrand.
            # The function will normally go to zero at 0 due to the subtraction of Lorentzians. This is undesirable behaviour, so the term at the 
            # end will add in the correct value (found by taking the limit) in this specific case.
        else:
            out = 0
            # out += (type_m == 0) * lorentzian(omega, center, width, height)
            out += (type_m == 1) * gaussian(omega, center, width, height)
            out += (type_m >= 2) * free_bernstein(omega, type_m, type_n, center, width, height)
            return out
    
    def grid_integrand(self, omega, omega_n, c, w, h, m, n):
        spectralw = self.peak(omega, c, w, h, m, n).sum(axis=0)
        return (1/np.pi) * omega**2 * spectralw / (omega**2+omega_n**2)

    def lor_pi(self, omega_n, center, height, width):
        integrated_function = 2 * height * center / (center**2 + (omega_n + width)**2)
        return integrated_function

    def set_freq_grid(self, w_max, beta):
        delta_w = w_max/self.N_w                # Grid size is largest value divided by number of datapoints
        delta_wn = 2*np.pi/beta                 
        wn_max = delta_wn * self.N_wn           # Max value is grid size multiplied by number of datapoints

        self.wn_list = np.arange(0.0, wn_max, delta_wn, dtype=float)    # Equally-distributed points between 0 and the max value with separation equal to the grid sizes
        self.w_list  = np.arange(0.0, w_max , delta_w, dtype=float)
        return self.wn_list, self.w_list

    def set_integration_grid(self, w_max, tail_power, N_tail):
        pos_w_list = self.w_list
        neg_w_list  = -np.flip(pos_w_list[1:]) # Reverse order of datapoints and make them negative, extending the list into the negatives
        pos_tail = np.logspace(np.log10(w_max), tail_power, N_tail)         # The tail is a set of exponentially-spaced grid lines placed after the last peak, used as
                                                                            # an integration tool.
        neg_tail = -np.flip(pos_tail)[:-1]
        
        full_w_list = [ neg_tail, neg_w_list, pos_w_list, pos_tail ]
        full_w_list = np.concatenate(full_w_list) + SMALL                   # Combine the positive and negative lists together. "SMALL" is a small offset value to prevent
                                                                            # issues involving zero.

        self.w_grid, self.wn_grid = np.meshgrid(full_w_list, self.wn_list)  
        return  self.w_grid, self.wn_grid

    # Center Distribution Functions

    def piecelin(self, v): # A randomizable monotonically increasing piecewise linear function
        angles = np.random.uniform(0, np.pi/2, size=self.N_seg) # Generates a list of random angles of length N_seg in the interval [0,pi/2).
        sines = np.sin(angles)
        cosines = np.cos(angles) # Two lists, one of the sines of the angles, one of the cosines.
        x_list = np.cumsum(cosines) # The x-coordinates of the connection points between line segments. Each is the previous x-value plus the cosine of the current angle.
        x_list *= 1.1*self.w_max/x_list[-1] # Resize the x-coordinates so all the peaks fit inside
        y_list = np.cumsum(sines) # The y-coordinates of the connection points between line segments
        x_list = np.insert(x_list,0,0)
        y_list = np.insert(y_list,0,0) # We add (0,0) to the beginning of the list of endpoints
        x_lower = np.zeros(len(v))
        y_lower = np.zeros(len(v))
        x_upper = np.zeros(len(v))
        y_upper = np.zeros(len(v)) # Initialized vectors that will store the endpoints of the line segment each input point is on
        for i in range(len(v)):
            # For each value in the vector v, this for-loop will find the x and y coordinates of the endpoints to either side of it.
            # In other words, these lists determine which line segment each point is located on.
            x_lower[i] = np.max(x_list[x_list <= v[i]])
            y_lower[i] = np.max(y_list[x_list <= v[i]])
            x_upper[i] = np.min(x_list[x_list > v[i]])
            y_upper[i] = np.min(y_list[x_list > v[i]])
        x_diffs = x_upper - x_lower
        y_diffs = y_upper - y_lower
        slopes = y_diffs/x_diffs # These three lines find the slope of the line segment each point exists on
        return slopes*v + y_lower - slopes*x_lower # This outputs the heights of the piecewise function at each point. 
                                                   # This is derived from expressing the line segment in point-slope form.

    def softp(self, x): # Each term has the form A*log(1+exp(x-c)), plus a linear term 
        c = np.random.uniform(0, 1, self.N_seg) # This will store all the values of c
        c = np.sort(c) # Put the values in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale the list of c-values so too much of the tail doesn't get included
        A = np.zeros(self.N_seg + 1) # This will store all the values of A, including the linear coefficient (hence the extra element)
        A[0] = 10 * np.random.rand() # This is the linear coefficient
        for i in range(self.N_seg):
            A_sum = np.sum(A) # Add up all the coefficients so far
            A[i+1]= np.random.uniform(-A_sum, 10) # We want each subsequent coefficient to be greater than the negative sum of all the
                                                  # previous coefficients, or the function will not be monotonically increasing
        mat = np.tile(x,(self.N_seg,1))
        mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        expmat = np.exp(mat)
        logmat = np.log(1+expmat)
        unsized = np.vstack((x,logmat)) # Add one more copy of x to the matrix to represent the linear term
        unsummed = unsized * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by a coefficient before summing
        return unsummed.sum(axis=0)

    def arctsum(self, x): # Each term has the form A*arctan(B*(x+c))
        c = np.random.uniform(-100, 0, size=self.N_seg)
        B = np.random.uniform(0, 10, size=self.N_seg)
        A = np.random.uniform(0, 30, size=self.N_seg)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
        arcmat = np.arctan(mat)
        unsummed = arcmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)

    def erfsum(self,x): # Each term has the form A*erf(B*(x-c)), plus a linear term
        c = np.random.uniform(-10, 0, size=self.N_seg)
        B = np.random.uniform(0, 100, size=self.N_seg)
        A = np.random.uniform(0, 10, size=self.N_seg+1)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
        erfmat = erf(mat)
        unsized = np.vstack((x,erfmat)) # Add one more copy of x to the matrix to represent the linear term
        unsummed = unsized * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)

    def arssum(self,x): # Each term has the form A*arsinh(B*(x+c))
        c = np.random.uniform(-10, 0, size=self.N_seg)
        B = np.random.uniform(0, 10, size=self.N_seg)
        A = np.random.uniform(0, 10, size=self.N_seg)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
        arsmat = np.arcsinh(mat)
        unsummed = arsmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)
    
    def rootsum(self,x): # Each term has the form A*sign(x+c)*(|x+c|)^(1/n)
        c = np.random.uniform(-10, 0, size=self.N_seg)
        A = np.random.uniform(0, 10, size=self.N_seg)
        n = np.random.choice([2, 3, 4, 5], self.N_seg, p=[0.5, 0.25, 0.2, 0.05])
        n = 1/n
        # Generate the order of the root for each term. We will use second, third, fourth, and fifth roots with decreasing probability.
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat + np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        signmat = np.sign(mat) # Store the signs of the elements
        powermat = np.power(np.abs(mat), np.transpose(np.tile(n,(len(x),1)))) # Take the relevant root for the absolute value of each element, row by row
        powermat = signmat * powermat # Restore the original signs after taking the root, so all negative inputs yield negative outputs
        unsummed = powermat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)

    def exparsinh(self,x): # Each term has the form A*exp(B*arsinh(x+c))
        c = np.random.uniform(0, 100, size=self.N_seg)
        B = np.random.uniform(0, 0.9999, size=self.N_seg) # As long as B < 1, the function's slope will decrease for large x
        A = np.random.uniform(0, 10, size=self.N_seg)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        arsmat = np.arcsinh(mat)
        arsmat = arsmat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
        expmat = np.exp(arsmat)
        unsummed = expmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)

    def exparctan(self,x): # Each term has the form A*exp(arctan(B(x+c)))
        c = np.random.uniform(0, 100, size=self.N_seg)
        B = np.random.uniform(0, 2, size=self.N_seg) # As long as B < 1, the function's slope will decrease for large x
        A = np.random.uniform(0, 10, size=self.N_seg)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        mat = mat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
        arcmat = np.arctan(mat)
        expmat = np.exp(arcmat)
        unsummed = expmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)

    def arssoft(self,x): # Each term has the form A*arsinh(ln(1+exp(x+c)))
        c = np.random.uniform(0, 100, size=self.N_seg)
        A = np.random.uniform(0, 10, size=self.N_seg)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        expmat = np.exp(mat)
        logmat = np.log(1 + expmat)
        arsmat = np.arcsinh(logmat)
        unsummed = arsmat * np.transpose(np.tile(A,(len(x),1))) # Multiply each term by its coefficient before summing
        return unsummed.sum(axis=0)

    def tanerf(self,x): # Each term has the form A*tan(B*erf(C*(x+c)))
        c = np.random.uniform(0, 10, size=self.N_seg)
        A = np.random.uniform(0, 10, size=self.N_seg)
        B = np.random.uniform(0, np.pi/2, size=self.N_seg) # If B were greater than pi/2, multiple cycles of tan would activate and the function would be discontinuous
        C = np.random.uniform(0, 2, size=self.N_seg)
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        mat = mat * np.transpose(np.tile(C,(len(x),1))) # Multiply all the arguments by their coefficients
        erfmat = erf(mat)
        erfmat = erfmat * np.transpose(np.tile(B,(len(x),1))) # Multiply all the arguments by their coefficients
        tanmat = np.tan(erfmat)
        unsummed = tanmat * np.transpose(np.tile(A,(len(x),1))) # Multiply all the arguments by their coefficients
        return unsummed.sum(axis=0)
    
    def logarc(self,x): # Each term has the form A*log(np.pi/2 + a + arctan(x+c))
        c = np.random.uniform(0, 10, size=self.N_seg)
        A = np.random.uniform(0, 10, size=self.N_seg)
        a = np.random.uniform(0, 10, size=self.N_seg) # If you add something less than pi/2 before taking the log, you will get a vertical asymptote, which we do not want
        c = np.sort(c) # Put the random values of c in order
        c *= np.random.uniform(0.8, 1.2)*self.w_max/c[-1] # Rescale so too much of the tail doesn't get included
        mat = np.tile(x,(self.N_seg,1))
        mat = mat - np.transpose(np.tile(c,(len(x),1))) # Each row of the matrix is a duplicate of x, each of which gets offset by a different amount
        arcmat = np.arctan(mat)
        arcmat = arcmat + np.pi/2 + np.transpose(np.tile(c,(len(x),1)))
        logmat = np.log(arcmat)
        unsummed = logmat * np.transpose(np.tile(A,(len(x),1))) # Multiply all the arguments by their coefficients
        return unsummed.sum(axis=0)

    def debug(self,x): # A simple, non-randomizable function that is used to test whether the code is working. Should not be called normally
        return x
    # More center distribution functions to be added.

    def generate_gauss_batch(self, batch_size):
        pi_of_wn_array = np.zeros([ batch_size, self.N_wn]) # Array stores the values of the integrated function
        sig_of_w_array = np.zeros([ batch_size, self.N_w ]) # Array stores the values of the sigma function
        # alternative sampling (attempt for scale-free spectra)
        sqrt_smpl_sigm = np.zeros([ batch_size, self.N_w ])
        cbrt_smpl_sigm = np.zeros([ batch_size, self.N_w ])

        for i in range(batch_size):
            if (i==0 or (i+1)%(max(1,batch_size//100))==0): print(f"sample {i+1}")

            # random spectrum characteristics
            num_drude    = np.random.randint( 0 if self.max_peaks > 0 else 1,     self.max_drude+1 )    # max_peaks is the maximum number of non-Drude peaks
            num_others   = np.random.randint( 0 if num_drude > 0 else 1,     self.max_peaks+1 )
            weight_ratio = np.random.uniform( SMALL, self.weight_ratio)
            num_peak = num_drude + num_others
            
            # random initialization (center, width, height, n, m) of peaks
            min_c = self.peak_position_range[0]
            max_c = self.peak_position_range[1]
            c  = np.random.uniform( min_c, max_c, size=num_peak )
            w  = np.random.uniform( 0.0  , 1.000, size=num_peak )   # The widths and heights are initialized as random numbers between 0 and 1
            h  = np.random.uniform( 0.0  , 1.000, size=num_peak )
            if self.bernstein:            
                m = np.random.randint(2, 20, size=num_peak)
                n = np.ceil(np.random.uniform( 0.0  , 1.000, size=num_peak ) * (m-1))
            else:
                m = np.ones(num_peak)
                n = np.ones(num_peak)

            print(m, n)

            # Drude peaks adjustments
            c[:num_drude]  = 0.0 # The first num_drude peaks are put at 0
            w[:num_drude] *= (self.drude_width_range[1] - self.drude_width_range[0]) # The ranges of the widths and heights are adjusted to their proper values
            w[:num_drude] += self.drude_width_range[0] #min
            h[:num_drude] *= weight_ratio/( h[:num_drude].sum() + SMALL ) # weight_ratio is the weight put into Drude peaks compared to other peaks
            
            # other peaks adjustments
            w[num_drude:] *= (self.peak_width_range[1] - self.peak_width_range[0])
            w[num_drude:] += self.peak_width_range[0] #min
            h[num_drude:] *= (1-weight_ratio)/( h[num_drude:].sum() + SMALL )
            
            #symmetrize and normalize
            c = np.hstack([c,-c])   # hstack is like concatenate but works in more general cases, such as for tensors instead of just vectors
            w = np.hstack([w, w])
            h = np.hstack([h, h])
            n = np.hstack([n, m-n])
            m = np.hstack([m, m])
            if self.normalize:
                h /= h.sum(axis=-1, keepdims=True) # -1 index is the last index
            h *= self.Pi0 * np.pi # Normalizing to something other than 1 (Pi0)

            # compute matsubara spectrum (training inputs)
            matsubaraGrid = self.grid_integrand(                    # matsubaraGrid is a 3-d tensor, which is generated by adding axes to the arguments as needed.
                                self.w_grid [ np.newaxis,:,: ], 
                                self.wn_grid[ np.newaxis,:,: ],     
                                c[ :, np.newaxis, np.newaxis ],
                                w[ :, np.newaxis, np.newaxis ],
                                h[ :, np.newaxis, np.newaxis ],
                                m[ :, np.newaxis, np.newaxis ],
                                n[ :, np.newaxis, np.newaxis ]
                            )
            pi_of_wn_array[i] = integrate.simps( matsubaraGrid[0], self.w_grid, axis=1) # Approximation of an integral (actually a sum over the discrete grid values)

            # sample real spectrum (training targets)
            sig_of_w_array[i] = self.peak(
                                    self.w_list[np.newaxis,:], 
                                    c[:,np.newaxis],
                                    w[:,np.newaxis],
                                    h[:,np.newaxis],
                                    m[:,np.newaxis],
                                    n[:,np.newaxis] 
                                ).sum(axis=0)
            
            # squareroot sampling real spectrum (alternative training targets)
            second_moment = (self.wn_list[-1])**2*pi_of_wn_array[i][-1]
            sqrt_w_max = self.sqrt_ratio * np.sqrt(second_moment)
            sqrt_w_list = np.linspace(0.0, sqrt_w_max , self.N_w, dtype=float)
            sqrt_smpl_sigm[i] = self.peak(
                                    sqrt_w_list[np.newaxis,:], 
                                    c[:,np.newaxis],
                                    w[:,np.newaxis],
                                    h[:,np.newaxis],
                                    m[:,np.newaxis],
                                    n[:,np.newaxis]
                                ).sum(axis=0)
            
            # cubicroot sampling real spectrum (alternative training targets)
            cbrt_w_max = self.cbrt_ratio * np.cbrt(second_moment)
            cbrt_w_list = np.linspace(0.0, cbrt_w_max , self.N_w, dtype=float)
            cbrt_smpl_sigm[i] = self.peak(
                                    cbrt_w_list[np.newaxis,:], 
                                    c[:,np.newaxis],
                                    w[:,np.newaxis],
                                    h[:,np.newaxis],
                                    m[:,np.newaxis],
                                    n[:,np.newaxis]
                                ).sum(axis=0)

        return pi_of_wn_array, sig_of_w_array, sqrt_smpl_sigm, cbrt_smpl_sigm

    def generate_lorentz_batch(self, batch_size):
        # center_function is the function that will determine the locations of the centres of the Lorentzians.
        pi_of_wn_array = np.zeros([ batch_size, self.N_wn]) # Array stores the values of the integrated function
        sig_of_w_array = np.zeros([ batch_size, self.N_w ]) # Array stores the values of the sigma function
        # alternative
        sqrt_smpl_sigm = np.zeros([ batch_size, self.N_w ])
        cbrt_smpl_sigm = np.zeros([ batch_size, self.N_w ])

        for i in range(batch_size):
            if (i==0 or (i+1)%(max(1,batch_size//100))==0): print(f"sample {i+1}")
            
            # initialization (center, width, height) of peaks
            method = random.randint(0,10)
            # method = -1 # Comment out the previous line and uncomment this one for debug mode
            if method == 0:
                center = self.piecelin(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 1:
                center = self.softp(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 2:
                center = self.arctsum(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 3:
                center = self.erfsum(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 4:
                center = self.arssum(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 5:
                center = self.rootsum(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 6:
                center = self.exparsinh(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 7:
                center = self.exparctan(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 8:
                center = self.arssoft(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 9:
                center = self.tanerf(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == 10:
                center = self.logarc(np.linspace(0, self.w_max, self.lor_peaks))
            elif method == -1:
                center = self.debug(np.linspace(0, self.w_max, self.lor_peaks)) # Used for debugging, should not normally be called
            # Use center distribution functions to generate peaks spaced as necessary, starting from a linearly-spaced set from 0 to 1
            # As more center distribution functions get completed, will need to add calls for them here.
            center -= center[0]
            #center += center[1]
            center *= self.w_max/center[-1] # Shift the vectors to make them strictly greater than zero, then rescale
            
            width = np.ones(self.lor_peaks) * self.lor_width # All peaks have the same width.
            height = np.ones(self.lor_peaks) # All peaks have the same height.

            #normalize
            if self.normalize:
                height /= height.sum(axis=-1, keepdims=True) # -1 index is the last index
            # height *= self.Pi0 * np.pi # Normalizing to something other than 1 (Pi0)

            # sample real spectrum (training targets)
            sig_of_w_array[i] = self.peak(
                                    self.w_list[np.newaxis,:], 
                                    center[:,np.newaxis], 
                                    width[:,np.newaxis], 
                                    height[:,np.newaxis] 
                                ).sum(axis=0)

            pi_of_wn_array[i] = self.lor_pi(
                                self.wn_list[ np.newaxis,: ],
                                center[ :, np.newaxis ], 
                                height[ :, np.newaxis ],
                                width[ :, np.newaxis  ]
                                ).sum(axis=0)

            # TODO: Modify the squareroot and cuberoot sampling for the Lorentzian case
            # squareroot sampling real spectrum (alternative training targets)
            second_moment = (self.wn_list[-1])**2*pi_of_wn_array[i][-1]
            sqrt_w_max = self.sqrt_ratio * np.sqrt(second_moment)
            sqrt_w_list = np.linspace(0.0, sqrt_w_max , self.N_w, dtype=float)
            sqrt_smpl_sigm[i] = self.peak(
                                    sqrt_w_list[np.newaxis,:], 
                                    center[:,np.newaxis], 
                                    width[:,np.newaxis], 
                                    height[:,np.newaxis] 
                                ).sum(axis=0)
            
            # cubicroot sampling real spectrum (alternative training targets)
            cbrt_w_max = self.cbrt_ratio * np.cbrt(second_moment)
            cbrt_w_list = np.linspace(0.0, cbrt_w_max , self.N_w, dtype=float)
            cbrt_smpl_sigm[i] = self.peak(
                                    cbrt_w_list[np.newaxis,:], 
                                    center[:,np.newaxis], 
                                    width[:,np.newaxis], 
                                    height[:,np.newaxis] 
                                ).sum(axis=0)

        return pi_of_wn_array, sig_of_w_array, sqrt_smpl_sigm, cbrt_smpl_sigm

    def compute_tail_ratio(self, pi_of_wn_array, sig_of_w_array, N=10):
        pi_tail = self.wn_list[-N:]**2*pi_of_wn_array[-N:]
        pi_diff = pi_tail[1:]-pi_tail[:-1]

        sigma_tail = np.cumsum(self.w_list[-N:]**2*sig_of_w_array[-N:])
        sigma_diff = sigma_tail[1:]-sigma_tail[:-1]

        return sigma_diff/(pi_diff+0.00001)

    def plot(self, pi_of_wn_array, sig_of_w_array, sqrt_smpl_sigm, cbrt_smpl_sigm):
        alpha = np.sqrt(self.wn_list**2*pi_of_wn_array)
        s2avg = np.sqrt(np.cumsum((self.w_list)**2*sig_of_w_array,axis=-1)*self.w_volume)
        
        print('\nnormalization')
        print('sum =', sig_of_w_array.sum(axis=-1)*self.w_volume)
        print('Pi0 =', pi_of_wn_array[:,0].real)
        print('s2avg = ',s2avg[:,-1])
        print('alpha = ', alpha[:,-1],'\n')

        if self.lorentz:
            print('Using Lorentzians')
        else:
            print('Using Gaussians')
    
        fig, ax = plt.subplots(2, 4, figsize=[12,5])
        
        ax[0,0].set_ylabel(r"$\Pi_n = \Pi(\omega_n)$")
        plt.setp(ax[0,0].get_xticklabels(), visible=False)
        ax[1,0].set_ylabel(r"$\sqrt{\omega_n^2 \Pi_n}$")
        ax[1,0].set_xlabel(r"$i\omega_n$")
        
        ax[0,1].set_ylabel(r"$\sigma(\omega)$")
        plt.setp(ax[0,1].get_xticklabels(), visible=False)
        ax[1,1].set_ylabel(r"running $\sqrt{ \langle \omega^2 \rangle_{\sigma} }$") #"$\int_0^\omega dz z^2\sigma(z^2)$"
        ax[1,1].set_xlabel(r"$\omega$")
        
        ax[0,2].set_ylabel(r"$\sigma_m$")
        plt.setp(ax[0,2].get_xticklabels(), visible=False)
        ax[1,2].set_ylabel(r"$\frac{\sqrt{\langle \omega^2 \rangle}}{M}\sum_{n=0}^{m}n^{2}\sigma_{n}$")
        ax[1,2].set_xlabel(r"$m$")

        ax[0,3].set_ylabel(r"$\sigma_m$")
        plt.setp(ax[0,3].get_xticklabels(), visible=False)
        ax[1,3].set_ylabel(r"$\frac{1}{M}\sum_{n=0}^{m}n^{2}\sigma_{n}$")
        ax[1,3].set_xlabel(r"$m$")

        for i in range(len(pi_of_wn_array)):
            ax[0,0].plot( self.wn_list, pi_of_wn_array[i] )
            ax[1,0].plot( self.wn_list, alpha[i] )
            if self.lorentz:
                ax[0,1].plot( self.w_list , sig_of_w_array[i])
            else:
                ax[0,1].plot( self.w_list , sig_of_w_array[i] )
            ax[1,1].plot( self.w_list , s2avg[i] )
            # ax[2,0].plot( self.compute_tail_ratio(pi_of_wn_array[i], sig_of_w_array[i]) )
            
            integer_w_list = np.arange(len(self.w_list))
            ax[0,2].plot( sqrt_smpl_sigm[i] )
            ax[1,2].plot( integer_w_list ,  (np.sqrt(alpha[i,-1]))*np.sqrt(np.cumsum((integer_w_list)**2*sqrt_smpl_sigm[i] ))/(self.N_w) )

            ax[0,3].plot( cbrt_smpl_sigm[i] )
            ax[1,3].plot( integer_w_list ,  np.sqrt(np.cumsum((integer_w_list)**2*cbrt_smpl_sigm[i] ))/(self.N_w) )

        fig.tight_layout()
        plt.show()

    def generate_dataset(self, N, path='./'):
        if self.lorentz:
            pi_wn_array, sig_of_w_array, sqrt_smpl_sigm, cbrt_smpl_sigm = self.generate_lorentz_batch(batch_size=N)
        else:
            pi_wn_array, sig_of_w_array, sqrt_smpl_sigm, cbrt_smpl_sigm = self.generate_gauss_batch(batch_size=N)
        np.savetxt( path + 'Pi.csv'     , pi_wn_array   , delimiter=',')
        np.savetxt( path + 'SigmaRe.csv', sig_of_w_array, delimiter=',')
        np.savetxt( path + 'SigmaRe_sqrtScale.csv', sqrt_smpl_sigm, delimiter=',')
        np.savetxt( path + 'SigmaRe_cbrtScale.csv', cbrt_smpl_sigm, delimiter=',')

if __name__ == '__main__':
    default_args = {
        # script parameters
        'test'         : False,
        'plot'         : 0,
        'generate'     : 0,
        'path'         : './',
        'normalize'    : True,
        # data generation parameters
        'in_size'      : 128,
        'out_size'     : 512,
        'w_max'        : 20.0,
        'beta'         : 10.0,#2*np.pi, # 2pi/beta = 1
        'N_tail'       : 128,
        'tail_power'   : 5,
        'Pi0'          : 1.0,
        'sqrt_ratio'   : 4,
        'cbrt_ratio'   : 6,
        # spectrum parameters (relative)
        'lorentz'      : False,
        'bernstein'    : False,
        'max_drude'    : 4,
        'max_peaks'    : 6,
        'weight_ratio' : 0.50,
        'drude_width'  : [.02, .1],
        'peak_pos'     : [.2 , .8],
        'peak_width'   : [.05, .1],
        'seed'         : int(time.time())
    }
    args = utils.parse_file_and_command(default_args, {})

    np.random.seed(args.seed)
    print('seed :',args.seed)
    sigma_path = args.path+'SigmaRe.csv'
    pi_path = args.path+'Pi.csv'

    
    if not (os.path.exists(sigma_path) or os.path.exists(pi_path)):
        print(args.generate)
        generator = DataGenerator(args)
        
        if args.generate > 0:
            os.makedirs(args.path, exist_ok=True)
            generator.generate_dataset(N=args.generate, path=args.path)
            if args.plot > 0:
                print('WARNING: examples printed are not part of the dataset')

        if args.plot > 0:
            if args.lorentz:
                pi, sigma, sigma2, sigma3 = generator.generate_lorentz_batch(batch_size=args.plot)
            else:
                pi, sigma, sigma2, sigma3 = generator.generate_gauss_batch(batch_size=args.plot)
            generator.plot(pi, sigma, sigma2, sigma3)

    else:
        if args.generate > 0:
            raise ValueError('ABORT GENERATION: there is already a dataset on this path')

        #%% test dataset with ContinuationData
        my_dataset = ContinuationData(args.path, normalize=args.normalize)

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

    if args.generate==0 and args.plot==0 and not args.test:
        print('nothing to do. try --help, --plot 10, or --generate 1000')
