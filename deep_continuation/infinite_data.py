#%%
import numpy as np
import matplotlib.pyplot as plt

from deep_continuation.function_generator import (
    default_parameters,
    pi_integral,
    simple_plot,
    SigmaPiGenerator,
)

# data generator
default_parameters.update({'rescale': 4.0})
generator = SigmaGenerator.create(**default_parameters)

def make_even(base_func):
    new_func =  lambda x: 0.5*(base_func(x)+base_func(-x))
    return new_func

def rescale(base_func, wmax, factor):
    s = (BIG**2)*pi_integral(BIG, base_func, grid_end=wmax)
    new_wmax = np.sqrt(s) * factor
    new_func = lambda x: (new_wmax/wmax)*base_func((new_wmax/wmax)*x) 
    return new_func, new_wmax


# def main():
Nw = 2048
w = np.linspace(0, wmax, Nw)
Nwn = 128

for f in range(10):

    sigma_base = sigma_generator.generate()
    sigma_even = make_even(sigma_base)
    sigma_func, new_wmax = rescale(sigma_even,
        wmax = default_parameters['wmax'],
        factor = 3.0
    )
    pi_func = lambda x: pi_integral(x, sigma_func, grid_end=new_wmax)
    
    beta = np.random.uniform(10,30)
    wn = np.arange(0, Nwn)*2*np.pi/beta
    # w = np.linspace(0, wmax, Nw)
    w = np.random.uniform(0, new_wmax, Nw)
    pi = pi_func(wn)

    targ = sigma_func(w)
    
    # simple_plot(pi, wn, targ, w)
    
    # pred = model(w, pi)




# if __name__ == "__main__":
#     main()