#!/usr/bin/env python
import numpy as np
from scipy import integrate

SMALL = 1e-10
INF = 1e10


def get_rescaled_sigma(sigma_func, rescale, wmax=20):
    sec_moment = second_moment(sigma_func, grid_end=wmax)
    s = np.sqrt(sec_moment) / rescale
    
    def new_sigma_func(w):
        return s * sigma_func(s * w)
    
    return new_sigma_func, s


def get_sigma(sigma_func, Nw, wmax):
    w = np.linspace(0, wmax, Nw)
    return (2*wmax)/(Nw*np.pi) * sigma_func(w)  # normalized to 1


def get_pi(sigma_func, Nwn=128, beta=30, wmax=20):
    wn = np.arange(0, Nwn) * 2*np.pi/beta
    return pi_integral(wn, sigma_func, grid_end=wmax)


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
        new_sigma_func, s = get_rescaled_sigma(sigma_func, rescale, wmax)
        sigma = get_sigma(new_sigma_func, Nw, wmax)
    else:
        s=1
        sigma = get_sigma(sigma_func, Nw, wmax)
        
    if rescale and not spurious:
        Pi = get_pi(new_sigma_func, Nwn, beta, wmax)
    else:
        Pi = get_pi(sigma_func, Nwn, beta, wmax)

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
