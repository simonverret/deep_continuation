#!/usr/local/bin/python3
# %%
import numpy as np
from scipy import integrate
import random
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.random.seed(139874)
SMALL = 1e-10

class DataGenerator():
    def __init__(self,  in_size=128, out_size=512, w_max=10, N_tail=128, power=5):
        self.N_wn = in_size
        self.N_w  = out_size
        
        self.wn_list, self.w_list = self.set_freq_grid( w_max )
        self.w_grid, self.wn_grid = self.set_integration_grid( N_tail, power )
        
        self.scale = 1
        # self.w_list *= self.scale

        # default peaks characteristics
        self.gauss = False
        self.drude_width_range   = np.array([0.1, 0.5]) * self.scale
        self.peak_position_range = np.array([2.0, 6.0]) * self.scale
        self.peak_width_range    = np.array([0.2, 1.0]) * self.scale
        self.max_drude = 4
        self.max_peaks = 6
        self.max_drude_ratio = 0.20

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

    def generate_batch(self, N_sample, plot=False):

        sig_of_w_array = np.zeros([ N_sample, self.N_w ])
        pi_of_wn_array = np.zeros([ N_sample, self.N_wn])

        for i in range(N_sample):
            if (i==0 or (i+1)%(max(1,N_sample//100))==0): print(f"sample {i+1}")

            # random spectrum characteristics
            num_drude    = np.random.randint( 0, self.max_drude)
            num_peak     = np.random.randint( 1, self.max_peaks)
            weight_ratio = np.random.uniform( SMALL, self.max_drude_ratio)
            
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
        
        # plot
        if plot: 
            for i in range(N_sample):
                plt.plot( generator.wn_list, pi_of_wn_array[i] )
            plt.show()

            for i in range(N_sample):
                plt.plot( generator.w_list, sig_of_w_array[i] )
            plt.show()
            
            print(sig_of_w_array.sum(axis=-1)*(2*10/512))
            print(pi_of_wn_array[:,0].real)

        return pi_of_wn_array, sig_of_w_array


    def generate_dataset(self, N_sample, path='./'):
        pi_wn_array, sig_of_w_array = self.generate_batch(N_sample)
        np.savetxt( path + 'SigmaRe.csv', sig_of_w_array, delimiter=',')
        np.savetxt( path + 'Pi.csv'     , pi_wn_array   , delimiter=',')


generator = DataGenerator()
# generator.generate_batch(10, plot=True)
generator.generate_dataset(50000)


# np.savetxt('SigmaRe.csv', spectralWeightArr, delimiter=',')
# np.savetxt('Pi.csv', matsubaraArr.real, delimiter=',')
