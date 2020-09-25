from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from deep_continuation.data_generator import *
from deep_continuation.monotonous_functions import *

np.set_printoptions(precision=4)
HERE = Path(__file__).parent
SMALL = 1e-10


def bernstein(x, m, n):
    return binom(m, n) * (x**n) * ((1-x)**(m-n)) * (x >= 0) * (x <= 1)


def centered_bernstein(x, m, n):
    c = (1+n)/(2+m)   # mathematica
    return (m+1)*bernstein(x+c, m, n)


def standardized_bernstein(x, m, n):
    w = np.sqrt(-((1+n)**2/(2+m)**2)+((1+n)*(2+n))/((2+m)*(3+m)))  # mathematica
    return centered_bernstein(x*w, m, n)*w


def free_bernstein(x, c, w, h, m, n):
    return h*standardized_bernstein((x-c)/w, m, n)/w


def random_cwh(num, cr=[0, 1], wr=[.05, .5], hr=[0, 1], norm=1.0, even=True):
    c = np.random.uniform(cr[0], cr[1], size=num)
    w = np.random.uniform(0.0, 1.0, size=num)*(wr[1]-wr[0])+wr[0]
    h = np.random.uniform(hr[0], hr[1], size=num)
    if even:
        c = np.hstack([c, -c])
        w = np.hstack([w, w])
        h = np.hstack([h, h])
    if norm is not None:
        h *= norm/(h.sum()+SMALL)
    return c, w, h


def random_ab(num, ra=[0.5, 20], rb=[0.5, 20], even=True):
    a = np.random.uniform(ra[0], ra[1], size=num)
    b = np.random.uniform(rb[0], rb[1], size=num)
    if even:
        aa, bb = a, b
        a = np.hstack([aa, bb])
        b = np.hstack([bb, aa])
    return a, b


def random_mn(num, rm=[1, 20], even=True):
    m = np.random.randint(rm[0], rm[1], size=num)
    n = np.ceil(np.random.uniform(0.0, 1.000, size=num)*(m-1))
    if even:
        n = np.hstack([n, m-n])
        m = np.hstack([m, m])
    return m, n


def test_plot_bernstein(c, w, h, m, n, **kwargs):
    nrm1 = integrate.quad(lambda x: free_bernstein(x, c, w, h, m, n), -np.inf, np.inf)[0]
    avg1 = integrate.quad(lambda x: x*free_bernstein(x, c, w, h, m, n), -np.inf, np.inf)[0]
    std1 = np.sqrt(integrate.quad(lambda x: (x-avg1)**2*free_bernstein(x, c, w, h, m, n), -np.inf, np.inf)[0])
    
    nrm2 = normalization(lambda x: free_bernstein(x, c, w, h, m, n), **kwargs)
    avg2 = first_moment(lambda x: free_bernstein(x, c, w, h, m, n), **kwargs)
    std2 = np.sqrt(second_moment(lambda x: free_bernstein(x, c, w, h, m, n), **kwargs))
    
    print(f"quad nrm = {nrm1}, intwtails nrm = {nrm2}")
    print(f"quad avg = {avg1}, intwtails avg = {avg2}")
    print(f"quad std = {std1}, intwtails sdt = {std2}")
    
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, bernstein(x, m, n))
    plt.plot(x, centered_bernstein(x, m, n))
    plt.plot(x, standardized_bernstein(x, m, n))
    plt.plot(x, free_bernstein(x, c, w, h, m, n))
    plt.show()


def test_plot_beta(c, w, h, a, b, **kwargs):
    nrm1 = integrate.quad(lambda x: free_beta(x, c, w, h, a, b), -np.inf, np.inf)[0]
    avg1 = integrate.quad(lambda x: x*free_beta(x, c, w, h, a, b), -np.inf, np.inf)[0]
    std1 = np.sqrt(integrate.quad(lambda x: (x-avg1)**2*free_beta(x, c, w, h, a, b), -np.inf, np.inf)[0])
    
    nrm2 = normalization(lambda x: free_beta(x, c, w, h, a, b), **kwargs)
    avg2 = first_moment(lambda x: free_beta(x, c, w, h, a, b), **kwargs)
    std2 = np.sqrt(second_moment(lambda x: free_beta(x, c, w, h, a, b), **kwargs))
    
    print(f"quad nrm = {nrm1}, intwtails nrm = {nrm2}")
    print(f"quad avg = {avg1}, intwtails avg = {avg2}")
    print(f"quad std = {std1}, intwtails sdt = {std2}")
    
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, beta_dist(x, a, b))
    plt.plot(x, centered_beta(x, a, b))
    plt.plot(x, standardized_beta(x, a, b))
    plt.plot(x, free_beta(x, c, w, h, a, b))
    plt.show()


def test_plot_compare(c, w, h, a, b, xmax=3):
    x = np.linspace(-xmax, xmax, 1000)
    plt.plot(x, gaussian(x, c, w, h))
    plt.plot(x, lorentzian(x, c, w, h))
    plt.plot(x, free_beta(x, c, w, h, a,b))
    plt.plot(x, free_bernstein(x, c, w, h, int(a+b-2),int(a-1)))
    plt.show()


def test_plot_spectra(xmax=1, drudes=4, others=12):
    plt.figure(num=None, figsize=(8, 6))
    x = np.linspace(-xmax, xmax, 1000)
    
    c1, w1, h1 = random_cwh(drudes, cr=[0.,0.], wr=[0.01,0.05])
    c2, w2, h2 = random_cwh(others, cr=[0.2,0.8], wr=[0.01,0.05])
    ratio = np.random.choice([0, np.random.uniform(0.2,0.8)])
    c = np.hstack([c1, c2])
    w = np.hstack([w1, w2])
    h = np.hstack([h1*ratio, h2*(1-ratio)])
    plt.plot(x, sum_on_args(gaussian, x, c, w, h), linewidth=2)

    m1, n1 = random_mn(drudes)
    m2, n2 = random_mn(others)
    m = np.hstack([m1, m2])
    n = np.hstack([n1, n2])
    plt.plot(x, sum_on_args(free_bernstein, x, c, w, h, m, n), linewidth=1)

    a1, b1 = random_ab(drudes)
    a2, b2 = random_ab(others)
    a = np.hstack([a1, a2])
    b = np.hstack([b1, b2])
    plt.plot(x, sum_on_args(free_beta, x, c, w, h, a, b), linewidth=1)
    plt.show()


def test_plot_arsenault():



if __name__ == "__main__":
    test_plot_bernstein(0, 1, 1, 5, 4)
    test_plot_compare(0, 1, 1, 0.9, 2)
    test_plot_spectra()