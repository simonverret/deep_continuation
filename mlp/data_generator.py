#!/usr/local/bin/python3
# %%
import numpy as np
from scipy import integrate
import random
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.random.seed(139874)
SMALL = 1e-10

#%%
N_sample = 20

freq_scale = 1

w_min = 0.0
w_max = 10.0
N_w = 512
N_wn = 128
N_tail = 128


realFreqs = np.arange(w_min,w_max,(w_max-w_min)/N_w)
realFreqs += SMALL

matsFreqs = np.arange(0.0,N_wn)

tail = np.logspace(1,5,N_tail)
neg_tail = -np.flip(tail)[:-1]
neg_realFreqs = -np.flip(realFreqs[1:])
realFreqsDense = np.concatenate((neg_tail,neg_realFreqs,realFreqs,tail))
realFreqsDense += SMALL

N_dense = len(realFreqsDense)

wGrid, wnGrid = np.meshgrid(realFreqsDense,matsFreqs)

# peaks parameters
gauss      = False
min_w         = 2.0 * freq_scale
max_w         = 6.0 * freq_scale
maxNumPeaks   = 6
maxNumDrude   = 4
maxDrudeWeightFrac = 0.15
maxDrudeWidth = 0.5 * freq_scale
minDrudeWidth = 0.1 * freq_scale
maxPeakWidth  = 1.0 * freq_scale
minPeakWidth  = 0.2 * freq_scale

def peak(omega, height=1, width=1, center=0):
    if gauss:
        return (height/np.sqrt(np.pi)/width) * np.exp(-(omega-center)**2/width**2)
    else:
        return (height/np.pi) * width/( (omega-center)**2 + (width)**2 )

def fullGridIntegrand(wGrid, wnGrid, h, w, c):
    spectralw = peak(wGrid, h, w, c).sum(axis=0)
    return wGrid**2 * spectralw/ (wGrid**2+wnGrid**2)

spectralWeightArr = np.zeros([N_sample,N_w] )
matsubaraArr = np.zeros([N_sample,N_wn])

for i in range(N_sample):
    if (i==0 or (i+1)%(max(1,N_sample//100))==0): print(f"sample {i+1}")
    matsubaraGrid = np.zeros(wGrid.shape)

    # spectrum characteristics
    numDrude    = np.random.randint(        0, maxNumDrude)
    numPeak     = np.random.randint(        1, maxNumPeaks)
    DrudeWeight = np.random.uniform(      SMALL, maxDrudeWeightFrac)
    # random initialization (width, center, height) of peaks
    w  = np.random.uniform( 0.001, 1.000, size=numPeak )
    c  = np.random.uniform( min_w, max_w, size=numPeak )
    h  = np.random.uniform( 0.001, 1.000, size=numPeak )
    # Drude peaks adjustments
    c[:numDrude] *= 0
    w[:numDrude] *= maxDrudeWidth 
    w[:numDrude] += minDrudeWidth
    h[:numDrude] *= DrudeWeight/( h[:numDrude].sum() + SMALL )
    # other peaks adjustments
    w[numDrude:] *= maxPeakWidth
    w[numDrude:] += minPeakWidth
    h[numDrude:] *= (1-DrudeWeight)/( h[numDrude:].sum() + SMALL )
    #symmetrize
    w = np.hstack([w, w])
    h = np.hstack([h, h])
    c = np.hstack([c,-c])
    #normalize
    h *= 1/h.sum(axis=-1,keepdims=True)

    spectralWeightArr[i] = peak( 
                                realFreqs[np.newaxis,:] + SMALL, 
                                h[:,np.newaxis], 
                                w[:,np.newaxis], 
                                c[:,np.newaxis] 
                            ).sum(axis=0)    

    matsubaraGrid        = fullGridIntegrand( 
                                wGrid [ np.newaxis,:,: ] , 
                                wnGrid[ np.newaxis,:,: ] , 
                                h[ :, np.newaxis, np.newaxis ], 
                                w[ :, np.newaxis, np.newaxis ], 
                                c[ :, np.newaxis, np.newaxis ]
                            )
    # print(wGrid.shape)
    # print(matsubaraGrid[0].shape)
    # plt.plot( wGrid[0, 128:-128], matsubaraGrid[0,0,128:-128] )
    
    matsubaraArr[i]      = integrate.simps( matsubaraGrid[0], wGrid, axis=1)
    
    # print(wGrid[:,0].shape)
    # print(matsubaraArr[i].shape)
    # plt.plot( wnGrid[:,0], matsubaraArr[i] )

# plt.show()

for i in range(N_sample):
    plt.plot(realFreqs,spectralWeightArr[i] )
plt.show()

for i in range(N_sample):
    plt.plot( matsFreqs, matsubaraArr[i] )
plt.show()

print(spectralWeightArr.sum(axis=-1)*(2*10/512))
print(matsubaraArr[:,0].real)

# np.savetxt('SigmaRe.csv', spectralWeightArr, delimiter=',')
# np.savetxt('Pi.csv', matsubaraArr.real, delimiter=',')
