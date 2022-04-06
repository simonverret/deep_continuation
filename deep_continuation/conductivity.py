#!/usr/bin/env python
from pathlib import Path

import numpy as np
from scipy import integrate

HERE = Path(__file__).parent
SMALL = 1e-10
INF = 1e10


def get_sigma_and_pi(
    distrib,
    Nwn=128,
    Nw=512,
    beta=30,
    wmax=20,
    rescale=False,
    spurious=False,   
):
    def sigma_func(x): 
        return 0.5 * (distrib(x) + distrib(-x))    

    if rescale:
        sec_moment = second_moment(sigma_func, grid_end=wmax)
        s = np.sqrt(sec_moment) * rescale/wmax * np.sqrt(1/np.pi)
        def new_sigma_func(w):
            return s * sigma_func(s * w)

        # # get sigma using the new wmax with the old sigma        
        # new_wmax = s*wmax
        # w = np.linspace(0, new_wmax, Nw)
        # sigma = sigma_func(w)
        # sigma *= (2*new_wmax)/(Nw*np.pi)  # sum(sigma) == 1 to use with softmax

        # or the old wmax with the new sigma
        w = np.linspace(0, wmax, Nw)
        sigma = new_sigma_func(w)
        sigma *= (2*wmax)/(Nw*np.pi)  # sum(sigma) == 1 to use with softmax
    else:
        w = np.linspace(0, wmax, Nw)
        sigma = sigma_func(w)
        sigma *= (2*wmax)/(Nw*np.pi)
        s = 1

    if rescale and not spurious:
        # # get the new pi using the old sigma with the new temperature:
        # new_beta = beta/s
        # pi_func = lambda x: pi_integral(x, sigma_func, grid_end=new_wmax)  # using the new_wmax (natural treshold of the old function) make the integral more precise
        # w_n = np.arange(0, Nwn) * 2*np.pi/new_beta
        # Pi = pi_func(w_n)
        
        # or using the new sigma with old temperature (and old wmax)
        pi_func = lambda x: pi_integral(x, new_sigma_func, grid_end=wmax)
        w_n = np.arange(0, Nwn) * 2*np.pi/beta
        Pi = pi_func(w_n)
    else:
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=wmax)
        w_n = np.arange(0, Nwn) * 2*np.pi/beta
        Pi = pi_func(w_n)
    
    return sigma, Pi, s
    

def pi_integral(wn, sigma_func, **kwargs):
    if isinstance(wn, np.ndarray):
        wn = wn[:, np.newaxis]  # to broadcast integral to all wn

    def integrand(w): 
        return (1/np.pi) * w**2/(w**2 + wn**2) * sigma_func(w)
    
    return integrate_with_tails(integrand, **kwargs)


def second_moment(sigma_func, grid=4096, tail=1024, grid_end=10, tail_power=7):
    
    def integrand(w): 
        return w**2 * sigma_func(w)
    
    return integrate_with_tails(integrand, grid, tail, grid_end, tail_power)


def integrate_with_tails(integrand, grid=4096, tail=1024, grid_end=10, tail_power=7):
    """Broadcastable integration on dense grid with long tails

    Integrate using `scipy.integrate.simps` using a three piece grid: one linearly
    spaced grid centered at zero, and two logarithmically spaced grid at each ends.

    Args:
        integrand (function): Function to be integrated
        grid (int, optional): Number of points in central grid. Defaults to 4096.
        tail (int, optional): Number of points in each tail. Defaults to 1024.
        grid_end (int, optional): Span of central grid (`-grid_end` to `grid_end`). Defaults to 10.
        tail_power (int, optional): Tail . Defaults to 7.

    Returns:
        ndarray: Result from an integration on `axis=-1`
    """
    grid_sampling = np.linspace(-grid_end, grid_end, grid)
    tail_sampling = np.logspace(np.log10(grid_end), tail_power, tail)[1:]
    full_sampling = np.concatenate(
        [-np.flip(tail_sampling), grid_sampling, tail_sampling]
    )
    return integrate.simps(integrand(full_sampling), full_sampling, axis=-1)
