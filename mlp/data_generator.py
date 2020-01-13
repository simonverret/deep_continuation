#%%
#!/usr/local/bin/python3
import numpy as np
from scipy import integrate
import random
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
np.random.seed(139874)
EPS = 1e-10

N_sample = 10

eta = 0.001
beta = 10
wn_max = 15*beta
N_limit = wn_max*beta/(2*np.pi) ## necessary number of frequency wn = (2n+1)pi/beta to reach wn_max
N_wn = int(2**np.ceil(np.log(N_limit)/np.log(2))) ## closest power of two from above

N_w = 1024
w_min = -10
w_max = 10

realFreqs = np.arange(w_min,w_max,(w_max-w_min)/N_w)
matsFreqs = np.arange(0,wn_max,wn_max/N_wn)

N_tail = 128
tail = np.logspace(1,24,N_tail)
neg_tail = -np.flip(tail)[:-1]
realFreqsDense = np.concatenate((neg_tail,realFreqs,tail))
N_wGrid = N_w+2*N_tail-1

wGrid, wnGrid = np.meshgrid(realFreqsDense,matsFreqs)


spectralWeightArr  = np.zeros( [N_sample, N_w ] )
matsubaraRepreArr  = np.zeros( [N_sample, N_wn] , dtype='complex128' )
matsubaraRepreGrid = np.zeros( [N_sample, N_wn, N_wGrid], dtype='complex128' )


# peaks parameters
min_w         = 2.0
max_w         = 6.0
maxNumPeaks   = 8
maxNumDrude   = 4
maxDrudeWeightFrac = 0.50
maxDrudeWidth = 0.5
minDrudeWidth = 0.01
maxPeakWidth  = 1.0
minPeakWidth  = 0.2

# initialize
widths  = np.random.uniform( 0.001, 1.000, size=[N_sample, maxNumPeaks] )
centers = np.random.uniform( min_w, max_w, size=[N_sample, maxNumPeaks] )
heights = np.random.uniform( 0.001, 1.000, size=[N_sample, maxNumPeaks] )

# adjust
for i in range(N_sample):
    numDrude    = np.random.randint(     0, maxNumDrude)
    numPeak     = np.random.randint(     5, maxNumPeaks)
    DrudeWeight = np.random.uniform( 0.001, maxDrudeWeightFrac)

    heights[i, numPeak: ]  *= 0.0
    widths [i, numPeak: ]  *= 0.0
    centers[i, numPeak: ]  *= 0.0

    # Drude peaks
    heights[i, :numDrude ] *= DrudeWeight/( heights[i, :numDrude].sum() + EPS )
    widths [i, :numDrude ] *= maxDrudeWidth 
    widths [i, :numDrude ] += minDrudeWidth
    centers[i, :numDrude ] *= 0
    
    # other peaks
    heights[i, numDrude: ] *= (1-DrudeWeight)/( heights[i, numDrude:].sum() + EPS )
    widths [i, numDrude: ] *= maxPeakWidth
    widths [i, numDrude: ] += minPeakWidth

#symmetrize
widths  = np.hstack([widths,widths])
heights = np.hstack([heights,heights])
centers = np.hstack([centers,-centers])

#normalize
heights *= 1/heights.sum(axis=-1,keepdims=True)

def lorentzian(omega, height=1, width=1, center=0):
    return (height/np.sqrt(np.pi)/width) * np.exp(-(omega-center)**2/width**2)
    # return (height/np.pi) * width/( (omega-center)**2 + width**2 )

def integrand(omega, z=0, height=1, width=1, center=0):
    return omega * lorentzian(omega, height, width, center)/ ( omega-z)

def gridIntegrand(wGrid, wnGrid, height=1, width=1, center=0):
    return wGrid * lorentzian(wGrid, height, width, center)/ (wGrid-1j*wnGrid)

h = heights[:, :, np.newaxis, np.newaxis]
w = widths[ :, :, np.newaxis, np.newaxis]
c = centers[:, :, np.newaxis, np.newaxis]

# spectral function as a sum of lorentzian
spectralw = lorentzian( wGrid, h, w, c ).sum(axis=1)

# full plot
# strt =  N_tail-1
# stop = -N_tail
# for i in range(N_sample):
#     plt.plot( wGrid[0, strt:stop], spectralw[ i, 0, strt:stop ])
# plt.show()

# half
# strt = (N_tail-1+N_w//2)
# stop = -N_tail
# for i in range(N_sample):
#     plt.plot( wGrid[0, strt:stop], spectralw[ i, 0, strt:stop ])
# plt.show()

# normalization = integrate.simps( spectralw[:,0,:], wGrid[0,:], axis=-1)
# print(normalization)


def fullGridIntegrand(wGrid, wnGrid, height, width, center):
    h = heights[:, :, np.newaxis, np.newaxis]
    w = widths [:, :, np.newaxis, np.newaxis]
    c = centers[:, :, np.newaxis, np.newaxis]
    spectralw = lorentzian(wGrid, h, w, c).sum(axis=1)
    return wGrid * spectralw/ (wGrid-1j*wnGrid)

matsubaraGridIntegrand = fullGridIntegrand( wGrid, wnGrid+1e-50, heights, widths, centers )
matsubaraArray = integrate.simps( matsubaraGridIntegrand, wGrid[np.newaxis,:,:], axis=-1)

# for i in range(N_sample):
#     plt.plot(wnGrid[:,0],matsubaraArray[i,:])
# plt.show()

print(matsubaraArray[:,0])



