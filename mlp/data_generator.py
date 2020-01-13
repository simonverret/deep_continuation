# %%
#!/usr/local/bin/python3
import numpy as np
from scipy import integrate
import random
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
# np.random.seed(139874)
EPS = 1e-10

N_sample = 64

beta = 10
wn_max = 15*beta
N_limit = wn_max*beta/(2*np.pi) ## necessary number of frequency wn = (2n+1)pi/beta to reach wn_max
N_wn = int(2**np.ceil(np.log(N_limit)/np.log(2))) ## closest power of two from above

N_w   =  512
w_min =  0
w_max =  10

realFreqs = np.arange(w_min,w_max,(w_max-w_min)/N_w)
matsFreqs = np.arange(0,wn_max,wn_max/N_wn)

N_tail = 128
tail = np.logspace(1,5,N_tail)
neg_tail = -np.flip(tail)[:-1]
neg_realFreqs = -np.flip(realFreqs[1:])
realFreqsDense = np.concatenate((neg_tail,neg_realFreqs,realFreqs,tail))
N_dense = len(realFreqsDense)

# AVOID zero
realFreqs += EPS
realFreqsDense += EPS

wGrid, wnGrid = np.meshgrid(realFreqsDense,matsFreqs)

# peaks parameters
gaussian      = False
min_w         = 2.0
max_w         = 6.0
maxNumPeaks   = 6
maxNumDrude   = 4
maxDrudeWeightFrac = 0.15
maxDrudeWidth = 0.5
minDrudeWidth = 0.1
maxPeakWidth  = 1.0
minPeakWidth  = 0.2

def lorentzian(omega, height=1, width=1, center=0):
    if gaussian:
        return (height/np.sqrt(np.pi)/width) * np.exp(-(omega-center)**2/width**2)
    else:
        return (height/np.pi) * width/( (omega-center)**2 + (width)**2 )

def fullGridIntegrand(wGrid, wnGrid, h, w, c):
    spectralw = lorentzian(wGrid, h, w, c).sum(axis=0)
    return wGrid * spectralw/ (wGrid-1j*(wnGrid))

spectralWeightArr = np.zeros([N_sample,N_w] )
matsubaraArr = np.zeros([N_sample,N_wn], dtype='complex128' )

for i in range(N_sample):
    if (i==0 or (i+1)%(N_sample//10)==0): print(f"sample {i+1}")
    matsubaraGrid = np.zeros(wGrid.shape, dtype='complex128' )

    # spectrum characteristics
    numDrude    = np.random.randint(        0, maxNumDrude)
    numPeak     = np.random.randint(        1, maxNumPeaks)
    DrudeWeight = np.random.uniform(      EPS, maxDrudeWeightFrac)
    # random initialization (width, center, height) of peaks
    w  = np.random.uniform( 0.001, 1.000, size=numPeak )
    c  = np.random.uniform( min_w, max_w, size=numPeak )
    h  = np.random.uniform( 0.001, 1.000, size=numPeak )
    # Drude peaks adjustments
    c[:numDrude] *= 0
    w[:numDrude] *= maxDrudeWidth 
    w[:numDrude] += minDrudeWidth
    h[:numDrude] *= DrudeWeight/( h[:numDrude].sum() + EPS )
    # other peaks adjustments
    w[numDrude:] *= maxPeakWidth
    w[numDrude:] += minPeakWidth
    h[numDrude:] *= (1-DrudeWeight)/( h[numDrude:].sum() + EPS )
    #symmetrize
    w = np.hstack([w, w])
    h = np.hstack([h, h])
    c = np.hstack([c,-c])
    #normalize
    h *= 1/h.sum(axis=-1,keepdims=True)

    spectralWeightArr[i] = lorentzian( 
                                realFreqs[np.newaxis,:] + EPS, 
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

print(matsubaraArr[:,0].real)
