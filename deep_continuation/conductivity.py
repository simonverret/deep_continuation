#!/usr/bin/env python
import numpy as np
from scipy import integrate

SMALL = 1e-10
INF = 1e10


def get_rescaled_sigma(sigma_func, old_std, new_std):
    
    s = old_std / new_std
    def new_sigma_func(w):
        return s * sigma_func(s * w)
    
    return new_sigma_func


def sample_on_grid(sigma_func, Nw, wmax):
    w = np.linspace(0, wmax, Nw)
    return (2*wmax)/(Nw*np.pi) * sigma_func(w)  # normalized to 1


def compute_matsubara_response(sigma_func, Nwn, beta, tail_start):
    wn = np.arange(0, Nwn) * 2*np.pi/beta
    return pi_integral(wn, sigma_func, tail_start=tail_start)


def pi_integral(wn, sigma_func, **kwargs):
    if isinstance(wn, np.ndarray):
        wn = wn[:, np.newaxis]  # to broadcast integral to all wn

    def integrand(w): 
        return (1/np.pi) * w**2/(w**2 + wn**2) * sigma_func(w)
    
    return integrate_with_tails(integrand, **kwargs)


def second_moment(sigma_func, grid=4096, tail=1024, tail_start=10, tail_power=7):
    
    def integrand(w): 
        return w**2 * sigma_func(w)
    
    return integrate_with_tails(integrand, grid, tail, tail_start, tail_power)


def integrate_with_tails(integrand, grid=4096, tail=1024, tail_start=10, tail_power=7):
    """Broadcastable integration on dense grid with long tails

    Integrate using `scipy.integrate.simps` using a three piece grid: one linearly
    spaced grid centered at zero, and two logarithmically spaced grid at each ends.

    Args:
        integrand (function): Function to be integrated
        grid (int, optional): Number of points in central grid. Defaults to 4096.
        tail (int, optional): Number of points in each tail. Defaults to 1024.
        tail_start (int, optional): Span of central grid (`-tail_start` to `tail_start`). Defaults to 10.
        tail_power (int, optional): Tail . Defaults to 7.

    Returns:
        ndarray: Result from an integration on `axis=-1`
    """
    grid_sampling = np.linspace(-tail_start, tail_start, grid)
    tail_sampling = np.logspace(np.log10(tail_start), tail_power, tail)[1:]
    full_sampling = np.concatenate(
        [-np.flip(tail_sampling), grid_sampling, tail_sampling]
    )
    return integrate.simps(integrand(full_sampling), full_sampling, axis=-1)
