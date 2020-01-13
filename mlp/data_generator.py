#!/usr/local/bin/python3
#%%
import numpy as np
from scipy import integrate
import random
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
np.random.seed(139874)

def lorentzian(omega, height=1, width=1, center=0):
    return (1/np.pi)*height * width/( (omega-center)**2 + width**2 )

def integrand(omega, z=0, height=1, width=1, center=0):
    return omega * lorentzian(omega, height, width, center)/ ( omega-z)

def gridIntegrand(wGrid, wnGrid, height=1, width=1, center=0):
    return wGrid * lorentzian(wGrid, height, width, center)/ (wGrid-1j*wnGrid)

# def complex_quad(func, a, b, fargs=None, **kwargs):
#     def real_func(x, *args):
#         return np.real(func(x, *fargs))
#     def imag_func(x, *args):
#         return np.imag(func(x, *fargs))
#     real_integral = integrate.quad(real_func, a, b, **kwargs)
#     imag_integral = integrate.quad(imag_func, a, b, **kwargs)
#     return real_integral[0] + 1j*imag_integral[0]

# def spectralIntegral(z, height=1, width=1, center=0):
#     return complex_quad(integrand, -np.inf, np.inf, fargs=(z, height, width, center) )

N_sample = 500

eta = 0.001
beta = 10
wn_max = 15*beta
N_limit = wn_max*beta/(2*np.pi) ## necessary number of frequency wn = (2n+1)pi/beta to reach wn_max
N_wn = int(2**np.ceil(np.log(N_limit)/np.log(2))) ## closest power of two from above

N_w = 2048
w_min = -10
w_max = 10

realFreqs = np.arange(w_min,w_max,(w_max-w_min)/N_w)
matsFreqs = np.arange(0,wn_max,wn_max/N_wn)

N_tail=128
realFreqsDense = np.concatenate((
                    -np.flip(np.logspace(1,5,N_tail))[:-1],
                    realFreqs,
                    np.logspace(1,5,N_tail)
                ))
wGrid, wnGrid = np.meshgrid(realFreqsDense,matsFreqs)
N_wGrid = N_w+2*N_tail-1

spectralWeightArr = np.zeros([N_sample,N_w] )
matsubaraRepreArr = np.zeros([N_sample,N_wn], dtype='complex128' )
# matsubaraRepreArr2 = np.zeros([N_sample,N_wn], dtype='complex128' )

for i in range(N_sample):
    # print('sample',i+1,'of',N_sample)

    matsubaraRepreGrid = np.zeros([N_wn,N_wGrid], dtype='complex128' )

    # running tracking of the normalization    
    accumWeight=0
    maxFracInPeak = np.random.uniform(0,0.15)

    numDrudePeaks = np.random.randint(1, 3)
    for j in range(numDrudePeaks): # drude peaks
        width = np.random.uniform(0.001,0.5)
        center = 0        
        height = np.random.uniform(0,maxFracInPeak-accumWeight) 
        accumWeight += height
        
        spectralWeightArr[i] += lorentzian(realFreqs, height, width+eta, center)
        matsubaraRepreGrid += gridIntegrand(wGrid, wnGrid, height, width+eta, center)

        ### QUADRATURE INTEGRATION (SLOWER)
        # for k, wn in enumerate(matsFreqs):
        #     matsubaraRepreArr2[i][k] += spectralIntegral(1j*wn, height, width+eta, center)
    
    numFiniteFreqPeaks = np.random.randint(1, 20)
    for j in range(numFiniteFreqPeaks): # finite freq peaks
        
        width = np.random.uniform(0,4) + .6
        center = np.random.uniform(-6,6)        
        if j+1 < numFiniteFreqPeaks:
            height = np.random.uniform(0,1-accumWeight)
            accumWeight += height
        else:
            height = 1 - accumWeight
        
        spectralWeightArr[i] += lorentzian(realFreqs, height, width+eta, center)
        matsubaraRepreGrid += gridIntegrand(wGrid, wnGrid, height, width+eta, center)
        
        ### QUADRATURE INTEGRATION (SLOWER)
        # for k, wn in enumerate(matsFreqs):
        #     matsubaraRepreArr2[i][k] += spectralIntegral(1j*wn, height, width+eta, center)

    matsubaraRepreArr[i] = integrate.simps( matsubaraRepreGrid, wGrid, axis=1)


print('done.')

# for i in range(N_sample):
#     plt.plot(realFreqs, spectralWeightArr[i])
# plt.show()

# for i in range(N_sample):
#     plt.plot(matsFreqs, np.real(matsubaraRepreArr[i]))
# plt.show()

# COMPARE THE TWO INTEGRALS
# for i in range(N_sample):
#     plt.plot(matsFreqs, np.real(matsubaraRepreArr2[i]))
# plt.show()

# for i in range(N_sample):
#     plt.plot(matsFreqs, np.real(matsubaraRepreArr[i] - matsubaraRepreArr2[i]))
# plt.show()

