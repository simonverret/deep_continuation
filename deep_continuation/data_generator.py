#%% Data generator
import os
import time
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.special import binom, gamma
import matplotlib.pyplot as plt

from deep_continuation import utils
from deep_continuation import monotonous_functions as monofunc

np.set_printoptions(precision=3)
HERE = Path(__file__).parent
SMALL = 1e-10


def integrate_with_tails(integrand, grid_points=2048, tail_points=1024, grid_end=10, tail_power=7):
    grid_sampling = np.linspace(-grid_end, grid_end, grid_points)
    tail_sampling = np.logspace(
        np.log10(grid_end), tail_power, tail_points)[1:]
    full_sampling = np.concatenate([
        -np.flip(tail_sampling),
        grid_sampling,
        tail_sampling
    ])
    return integrate.simps(integrand(full_sampling), full_sampling, axis=-1)


def pi_integral(wn, spectral_function):
    if isinstance(wn, np.ndarray):
        wn = wn[:, np.newaxis]
    def integrand(x): return (1/np.pi) * x**2 / (x**2+wn**2) * spectral_function(x)
    return integrate_with_tails(integrand)


def second_moment(spectral_function):
    def integrand(x): return (1/np.pi) * x**2 * spectral_function(x)
    return integrate_with_tails(integrand)

def gaussian(x, c, w, h):
    return (h/(np.sqrt(2*np.pi)*w))*np.exp(-((x-c)/w)**2/2)


def lorentzian(x, c, w, h):
    return (h/np.pi)*w/((x-c)**2+w**2)


def even_lorentzian(x, c=0, w=1, h=1):
    return (1/np.pi)*4*c*w*h/(((x-c)**2+w**2)*((x+c)**2+w**2))


def analytic_pi(x, c=0, w=0, h=0):
    return 2*h*c/(c**2+(x+w)**2)


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


def test_plot_bernstein(c, w, h, m, n):
    avg = integrate.quad(lambda x: x*free_bernstein(x, c, w, h, m, n), -np.inf, np.inf)[0]
    std = np.sqrt(integrate.quad(lambda x: (x-avg)**2*free_bernstein(x, c, w, h, m, n), -np.inf, np.inf)[0])
    print("avg =", avg)
    print("std =", std)
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, bernstein(x, m, n))
    plt.plot(x, centered_bernstein(x, m, n))
    plt.plot(x, standardized_bernstein(x, m, n))
    plt.plot(x, free_bernstein(x, c, w, h, m, n))
    plt.show()


def beta_dist(x, a, b):
    return (gamma(a+b)/(SMALL+gamma(a)*gamma(b))) * np.nan_to_num((x**(a-1))*((1-x)**(b-1)) * (x>0) * (x<1), copy=False)


def centered_beta(x, a, b):
    c = a/(a+b)
    return beta_dist(x+c, a, b)


def standardized_beta(x, a, b):
    w = np.sqrt(a*b/((a+b+1)*(a+b)**2))
    return centered_beta(x*w, a, b)*w


def free_beta(x, c, w, h, a, b):
    return h*standardized_beta((x-c)/w, a, b)/w


def test_plot_beta_dist(c, w, h, a, b):
    nrm = integrate_with_tails(lambda x: free_beta(x, c, w, h, a, b))
    avg = integrate_with_tails(lambda x: x*free_beta(x, c, w, h, a, b))
    std = integrate_with_tails(lambda x: (x-avg)**2*free_beta(x, c, w, h, a, b))
    print("nrm =", nrm)
    print("avg =", avg)
    print("std =", std)
    x = np.linspace(-3, 3, 1000)
    plt.plot(x, beta_dist(x, a, b))
    plt.plot(x, centered_beta(x, a, b))
    plt.plot(x, standardized_beta(x, a, b))
    plt.plot(x, free_beta(x, c, w, h, a, b))
    plt.show()

# test_plot_beta_dist(0.6, 0.81, 1, 0.5, 20)

#%%

def test_plot_compare(c, w, h, a,b, xmax=3):
    x = np.linspace(-xmax, xmax, 1000)
    plt.plot(x, gaussian(x, c, w, h))
    plt.plot(x, lorentzian(x, c, w, h))
    plt.plot(x, free_beta(x, c, w, h, a,b))
    plt.plot(x, free_bernstein(x, c, w, h, int(a+b-2),int(a-1)))
    plt.show()

def sum_on_args(f, x, *args):
    if isinstance(x, np.ndarray):
        x = x[np.newaxis, :]
        args = [a for a in args] # copy args to allow reassign 
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                while len(args[i].shape) < len(x.shape):
                    args[i] = args[i][:, np.newaxis]
    return f(x,*args).sum(axis=0)


def random_cwh(num, cr=[0,1], wr=[.05,.5], hr=[0,1], norm=1.0, even=True):
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


def random_mn(num, rm=[1,20], even=True):
    m = np.random.randint(rm[0], rm[1], size=num)
    n = np.ceil(np.random.uniform(0.0, 1.000, size=num)*(m-1))
    if even:
        n = np.hstack([n, m-n])
        m = np.hstack([m, m])
    return m, n


def random_ab(num, ra=[1,20], rb=[1,20], even=True):
    a = np.random.randint(ra[0], ra[1], size=num)
    b = np.random.randint(rb[0], rb[1], size=num)
    if even:
        aa, bb = a, b
        a = np.hstack([aa, bb])
        b = np.hstack([bb, aa])
    return a, b


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


def unscaled_plot(Pi, sigma, filename=None):
    N = len(Pi[0])
    M = len(sigma[0])
    n2Pi = np.sqrt(np.arange(N)**2*Pi)
    cumul_sum2 = np.sqrt(np.cumsum(np.linspace(0, 1, M)**2*sigma, axis=-1))

    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi_n$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{n^2 \Pi_n}$")
    ax[1, 0].set_xlabel(r"$n$")
    ax[0, 1].set_ylabel(r"$\sigma_m$")
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)
    ax[1, 1].set_ylabel(r"$\sqrt{ \sum_{r}^{n} n^2 \sigma_n }$")
    ax[1, 1].set_xlabel(r"$m$")
    for i in range(len(Pi)):
        ax[0, 0].plot(Pi[i], '.')
        ax[1, 0].plot(n2Pi[i], '.')
        ax[0, 1].plot(sigma[i])
        ax[1, 1].plot(cumul_sum2[i])
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def scale_plot(Pi, sigma, beta, w_max, filename=None):
    N = len(Pi[0])
    M = len(sigma[0])
    wn = (2*np.pi/beta[:, np.newaxis]) * np.arange(N)
    w = w_max[:, np.newaxis] * np.linspace(0, 1, M)
    n2Pi = wn**2*Pi
    cumul_sum2 = np.cumsum(w**2*sigma, axis=-1)

    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi(i\omega_n)$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{\omega_n^2 \Pi(i\omega_n)}$")
    ax[1, 0].set_xlabel(r"$\omega_n$")
    ax[0, 1].set_ylabel(r"$\sigma(\omega)$")
    ax[0, 1].set_ylim(0,0.5)
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)
    ax[1, 1].set_ylabel(
        r"$\sqrt{ \int\frac{d\omega}{\pi} \omega^2 \sigma(\omega) }$")
    ax[1, 1].set_xlabel(r"$\omega$")
    for i in range(len(Pi)):
        ax[0, 0].plot(wn[i], Pi[i], '.')
        ax[1, 0].plot(wn[i], n2Pi[i], '.')
        ax[0, 1].plot(w[i], sigma[i])
        ax[1, 1].plot(w[i], cumul_sum2[i])
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def infer_scale_plot(Pi, sigma, filename=None):
    N = len(Pi[0])
    Pi0 = Pi[:, 0]
    PiN = Pi[:, -1]
    M = len(sigma[0])
    sum1 = np.sum(sigma, axis=-1)
    sum2 = np.sum(np.arange(M), axis=-1)
    beta = 2*N*np.sqrt(PiN*sum1**3/(Pi0**3*sum2))
    w_max = np.pi*Pi0*M/sum1
    print(f"infered scales:\n  beta  = {beta}\n  w_max = {w_max}")
    scale_plot(Pi, sigma, beta, w_max)


class DataGenerator():
    def __init__(self, in_size, out_size, beta, w_max, rescale, **args):
        self.num_wn = in_size
        self.num_w = out_size
        self.beta = beta
        self.w_max = w_max
        self.w = np.linspace(0, self.w_max, self.num_w)
        self.wn = (2*np.pi/self.beta) * np.arange(0, self.num_wn)
        self.rescale = rescale

    def random_functions(self):
        raise NotImplementedError

    def generate_batch(self, size):
        Pi = np.zeros((size, self.num_wn))
        sigma = np.zeros((size, self.num_w))
        for i in range(size):
            if (i == 0 or (i+1)%(max(1,size//100)) == 0): 
                print(f"sample {i+1}")
            sigma_func, pi_func = self.random_functions()
            if self.rescale > SMALL:
                inf = 1e6
                s = np.sqrt(inf**2*pi_integral(inf, sigma_func))
                new_w_max = self.rescale*s
                resampl_w = np.linspace(0, new_w_max, self.num_w)
                sigma[i] = s*sigma_func(resampl_w)
            else:
                sigma[i] = sigma_func(self.w)
            Pi[i] = pi_func(self.wn)
        return Pi, sigma

    def generate_files(self, size, sigma_path, pi_path):
        if (os.path.exists(sigma_path) or os.path.exists(pi_path)):
            raise ValueError('there is already a dataset on this path')
        Pi, sigma = self.generate_batch(size)
        np.savetxt(pi_path, Pi, delimiter=',')
        np.savetxt(sigma_path, sigma, delimiter=',')

    def plot(self, size, name=None, basic=True, scale=False, infer=False):
        Pi, sigma = self.generate_batch(size)
        print(Pi[:,0])
        if basic: 
            unscaled_plot(Pi, sigma, name+"_basic.pdf" if name else None)
        if scale: 
            scale_plot(Pi, sigma, self.beta, self.w_max, name+"_scale.pdf" if name else None)
        if infer: 
            infer_scale_plot(Pi, sigma, name+"_infer.pdf" if name else None)
        

class PeakMixIntegrator(DataGenerator):
    def __init__(self, 
        kind="Gaussian", 
        kernel="Matsubara",
        Pi0=1,
        max_drude=4, 
        max_peaks=6, 
        weight_ratio=0.50, 
        drude_width=[.02, .1], 
        peak_pos=[.2, .8], 
        peak_width=[.05, .1], 
        **args
    ):
        super().__init__(**args)
        self.norm = Pi0*np.pi
        self.max_drude = max_drude
        self.max_peaks = max_peaks
        self.ratio = weight_ratio
        self.drude_wr = drude_width
        self.other_cr = peak_pos
        self.other_wr = peak_width
        self.kind = kind
        self.kernel = kernel

    def random_functions(self):
        drudes = np.random.randint(0 if self.max_peaks > 0 else 1, self.max_drude+1)
        others = np.random.randint(0 if drudes > 0 else 1, self.max_peaks+1)
        if drudes and others:
            drude_weight = np.random.choice([0, np.random.uniform(0, self.ratio)])
            rest = 1-drude_weight
        else:
            drude_weight = 1
            rest = 1
        c1, w1, h1 = random_cwh(drudes, [0,0], self.drude_wr, norm=drude_weight)
        a1, b1 = random_ab(drudes)
        c2, w2, h2 = random_cwh(others, self.other_cr, self.other_wr, norm=rest)
        a2, b2 = random_ab(others)
        c = np.hstack([c1, c2])*self.w_max
        w = np.hstack([w1, w2])*self.w_max
        h = np.hstack([h1, h2])*self.norm
        a = np.hstack([a1, a2])
        b = np.hstack([b1, b2])

        if self.kind in ["G", "Gaussian", "gaussian"]:
            sigma_func = lambda x: sum_on_args(gaussian, x, c, w, h)
        elif self.kind in ["B", "Beta", "beta"]:
            sigma_func = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)
        else: 
            raise ValueError(f"kind {self.kind} not recognized")
        
        if self.kernel in ["M", "Matsubara", "matsubara"]:
            pi_func = lambda x: pi_integral(x, sigma_func)
        else: 
            raise ValueError(f"kernel {self.kernel} not recognized")
        return sigma_func, pi_func


class LorentzComb(DataGenerator):
    def __init__(self, Pi0=1, num_peaks=2048, peak_widths=0.02, N_seg=8, **args):
        super().__init__(**args)
        self.norm = Pi0
        self.num_peaks = num_peaks
        self.N_seg = N_seg
        self.peak_widths = peak_widths

    def random_functions(self):
        k = np.linspace(0, 1, self.num_peaks)
        # c = monofunc.piecewise_gap(k, n=8, soft=0.05, xlims=[0,1], ylims=[0,0.8*self.w_max])
        c = monofunc.random_climb(k, xlims=[0,1], ylims=[0,0.8*self.w_max])
        w = np.ones(self.num_peaks)*self.peak_widths        
        h = abs(c) + 0.05
        h *= self.norm/(2*h*c/(c**2+w**2)).sum()
        sigma_func = lambda x: sum_on_args(even_lorentzian, x, c, w, h)
        pi_func = lambda x: sum_on_args(analytic_pi, x, c, w, h)
        return sigma_func, pi_func


def main():
    default_args = {
        'seed': int(time.time()),
        'plot': 0,
        'generate': 0,
        'path': str(HERE),
        'in_size': 128,
        'out_size': 512,
        'w_max': 20.0,
        'beta': 10.0,  # 2*np.pi, # 2pi/beta = 1
        'Pi0': 1.0,
        'rescale': 0.0,
        # peak_mix
        'use_bernstein': False,
        'max_drude': 4,
        'max_peaks': 6,
        'weight_ratio': 0.50,
        'drude_width': [.01, .05],
        'peak_pos': [.2, .8],
        'peak_width': [.01, .1],
        'kind': "Gaussian",
        # lorentz
        'lorentz': False,
        'num_peaks': 10000,
        'peak_widths': 0.05,
        'N_seg': 2,
        'center_method': -1,
        'remove_nonphysical': False,
        'scaled_plot': False
    }
    args = utils.parse_file_and_command(default_args, {})
    print(f"seed : {args.seed}")
    np.random.seed(args.seed)

    if args.lorentz:
        generator = LorentzComb(**vars(args))
    else:
        generator = PeakMixIntegrator(**vars(args))

    if args.plot > 0: 
        generator.plot(args.plot)
    elif args.generate > 0:
        os.makedirs(args.path, exist_ok=True)
        generator.generate_files(
            args.generate,
            pi_path = args.path+'/Pi.csv',
            sigma_path = args.path+'/SigmaRe.csv',
        )
        
    if args.generate==0 and args.plot==0:
        print("nothing to do, use --plot 10 or --generate 10000")

if __name__ == "__main__":
    main()
