#!/usr/bin/env python
#
#   deep_continuation
#
#   Simon Verret
#   Quinton Weyrich
#   Reza Nourafkan
#   Andre-Marie Tremblay
#


import os
import time
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.special import binom, gamma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from deep_continuation import utils
from deep_continuation import monotonous_functions as monofunc

np.set_printoptions(precision=4)
HERE = Path(__file__).parent
SMALL = 1e-10
INF = 1e10
COLORS = list(mcolors.TABLEAU_COLORS)

def sum_on_args(f, x, *args):
    if isinstance(x, np.ndarray):
        x = x[np.newaxis, :]
        args = [a for a in args]  # copy args to allow reassign
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                while len(args[i].shape) < len(x.shape):
                    args[i] = args[i][:, np.newaxis]
    return f(x, *args).sum(axis=0)


def integrate_with_tails(integrand, grid_points=4096, tail_points=1024, grid_end=10, tail_power=7):
    grid_sampling = np.linspace(-grid_end, grid_end, grid_points)
    tail_sampling = np.logspace(
        np.log10(grid_end), tail_power, tail_points)[1:]
    full_sampling = np.concatenate([
        -np.flip(tail_sampling),
        grid_sampling,
        tail_sampling
    ])
    return integrate.simps(integrand(full_sampling), full_sampling, axis=-1)


def pi_integral(wn, spectral_function, **kwargs):
    if isinstance(wn, np.ndarray):
        wn = wn[:, np.newaxis]

    def integrand(x): return (1/np.pi) * x**2 / \
        (x**2+wn**2) * spectral_function(x)
    return integrate_with_tails(integrand, **kwargs)


def normalization(f, **kwargs):
    def integrand(x): return f(x)
    return integrate_with_tails(integrand, **kwargs)


def first_moment(f, **kwargs):
    def integrand(x): return x*f(x)
    return integrate_with_tails(integrand, **kwargs)


def second_moment(f, **kwargs):
    def integrand(x): return ((x - first_moment(f))**2)*f(x)
    return integrate_with_tails(integrand, **kwargs)


def gaussian(x, c, w, h):
    return (h/(np.sqrt(2*np.pi)*w))*np.exp(-((x-c)/w)**2/2)


def lorentzian(x, c, w, h):
    return (h/np.pi)*w/((x-c)**2+w**2)


def even_lorentzian(x, c=0, w=1, h=1):
    return (1/np.pi)*4*c*w*h/(((x-c)**2+w**2)*((x+c)**2+w**2))


def analytic_pi(x, c=0, w=0, h=0):
    return 2*h*c/(c**2+(x+w)**2)


def beta_dist(x, a, b):
    return (gamma(a+b)/(SMALL+gamma(a)*gamma(b))) * np.nan_to_num((x**(a-1))*((1-x)**(b-1)) * (x > 0) * (x < 1), copy=False)


def centered_beta(x, a, b):
    c = a/(a+b)
    return beta_dist(x+c, a, b)


def standardized_beta(x, a, b):
    w = np.sqrt(a*b/((a+b+1)*(a+b)**2))
    return centered_beta(x*w, a, b)*w


def free_beta(x, c, w, h, a, b):
    return h*standardized_beta((x-c)/w, a, b)/w


def infer_scales(Pi, sigma):
    N = len(Pi[0,0])
    M = len(sigma[0])
    norm = Pi[0, :, 0]  # Pi0 is independent from temperature
    PiN = Pi[:, :, -1]
    sum1 = 2*np.sum(sigma, axis=-1) - np.take(sigma, 0, axis=-1)
    dm = np.pi*norm/sum1
    m = np.arange(M)
    sum2 = 2*np.sum(m**2*sigma, axis=-1)
    
    wmaxs = M*dm
    betas = 2*N*np.sqrt((np.pi**3)*PiN/(dm**3*sum2))
    print(betas.shape)
    print(wmaxs.shape)
    return wmaxs, betas


def unscaled_plot(Pi, sigma, filename=None):
    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi_n$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{n^2 \Pi_n}$")
    ax[1, 0].set_xlabel(r"$n$")
    ax[0, 1].set_ylabel(r"$\sigma_m$")
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)
    ax[1, 1].set_ylabel(r"$\sqrt{ \sum_{r}^{n} n^2 \sigma_n }$")
    ax[1, 1].set_xlabel(r"$m$")
    
    N = len(Pi[0,0])
    n2Pi = np.sqrt(np.arange(N)**2*Pi)
    for b in range(len(Pi)):
        for i in range(len(Pi[b])):
            ax[0, 0].plot(Pi[b,i], '.', c=COLORS[i%10])
            ax[1, 0].plot(n2Pi[b,i], '.', c=COLORS[i%10])
    M = len(sigma[0])
    cumul_sum2 = np.sqrt(np.cumsum(np.linspace(0, 1, M)**2*sigma, axis=-1))
    for i in range(len(sigma)):
        ax[0, 1].plot(sigma[i], c=COLORS[i%10])
        ax[1, 1].plot(cumul_sum2[i], c=COLORS[i%10])
        
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def scale_plot(Pi, sigma, betas, wmaxs, filename=None):
    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi(i\omega_n)$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{\omega_n^2 \Pi(i\omega_n)}$")
    ax[1, 0].set_xlabel(r"$\omega_n$")
    ax[0, 1].set_ylabel(r"$\sigma(\omega)$")
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)
    ax[1, 1].set_ylabel(r"$\sqrt{\int\frac{d\omega}{\pi}\omega^2\sigma(\omega)}$")
    ax[1, 1].set_xlabel(r"$\omega$")
    
    N = len(Pi[0,0])
    wn = (2*np.pi/betas[:, :, np.newaxis]) * np.arange(N)
    n2Pi = np.sqrt(wn**2*Pi)
    for b in range(len(Pi)):
        for i in range(len(Pi[b])):
            ax[0, 0].plot(wn[b,i], Pi[b,i], '.', c=COLORS[i%10], markersize=2*b+3)
            ax[1, 0].plot(wn[b,i], n2Pi[b,i], '.', c=COLORS[i%10], markersize=2*b+3)
    M = len(sigma[0])
    w = wmaxs[:, np.newaxis] * np.linspace(0, 1, M)
    cumul_sum2 = np.sqrt(np.cumsum(np.linspace(0, 1, M)**2*sigma, axis=-1))
    for i in range(len(sigma)):
        ax[0, 1].plot(w[i], sigma[i], c=COLORS[i%10])
        ax[1, 1].plot(w[i], cumul_sum2[i], c=COLORS[i%10])

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def infer_scale_plot(Pi, sigma, filename=None):
    wmaxs, betas = infer_scales(Pi, sigma)
    print(f" infered scales:\n  betas =\n{betas}\n  wmaxs =\n{wmaxs}")
    scale_plot(Pi, sigma, betas, wmaxs, filename)


class DataGenerator():
    def __init__(self, Nwn=128, Nw=512, beta=[10], wmax=20, rescale=0.0, **kwargs):
        self.Nwn = Nwn
        self.Nn = Nw
        self.beta = beta
        self.wmax = wmax
        self.rescale = rescale

    def generate_functions(self):
        raise NotImplementedError

    def generate_batch(self, size):
        Pi = np.zeros((len(self.beta), size, self.Nwn))
        betas = np.zeros((len(self.beta), size))
        sigma = np.zeros((size, self.Nn))
        sigma_r = np.zeros((size, self.Nn))
        wmaxs = np.zeros(size)

        for i in range(size):
            if (i == 0 or (i+1)%(max(1, size//100)) == 0): print(f"{i+1}/{size}")
            sigma_func, pi_func = self.generate_functions()

            
            if self.rescale > SMALL:
                s = INF**2*pi_integral(INF, sigma_func, grid_end=self.wmax)
                wmax = np.cbrt(s) * self.rescale
                omega = np.linspace(0, wmax, self.Nn)
                sigma_r[i] = sigma_func(omega)
                wmaxs[i] = wmax
            else:
                wmaxs[i] = self.wmax
            
            for b, beta in enumerate(self.beta):
                omega_n = np.arange(0, self.Nwn)*2*np.pi/beta        
                Pi[b,i] = pi_func(omega_n)
                betas[b,i] = beta

            omega = np.linspace(0, self.wmax, self.Nn)
            sigma[i] = sigma_func(omega)

        return Pi, sigma, betas, wmaxs, sigma_r

    def generate_files(self, size, sigma_path, pi_path, wmaxs_path=None):
        if (os.path.exists(sigma_path) or os.path.exists(pi_path)):
            raise ValueError('there is already a dataset on this path')
        Pi, sigma, betas, wmaxs, sigma_r = self.generate_batch(size)
        np.savetxt(pi_path, Pi[0], delimiter=',')
        np.savetxt(sigma_path, sigma, delimiter=',')

        for b, beta in enumerate(self.beta[1:]):
            new_path = pi_path.replace(".csv", f"_beta_{beta}.csv")
            np.savetxt(new_path, Pi[b], delimiter=',')
        if self.rescale:
            new_path = sigma_path.replace(".csv", f"_scaled_{self.rescale}.csv")
            np.savetxt(new_path, sigma_r, delimiter=',')
        if wmaxs_path:
            np.savetxt(wmaxs_path, wmaxs, delimiter=',')

    def plot(self, size, name=None, basic=True, scale=False, infer=False):
        Pi, sigma, betas, wmaxs, sigma_r = self.generate_batch(size)
        print(f" true scales:\n  betas =\n{betas}\n  wmaxs = {wmaxs}")
        print(f" normalization check\n{Pi[:,:,0]}")
        if self.rescale:
            sigma = sigma_r
        if basic:
            unscaled_plot(Pi, sigma, name+"_basic.pdf" if name else None)
        if scale:
            scale_plot(Pi, sigma, betas, wmaxs, name +"_scale.pdf" if name else None)
        if infer:
            infer_scale_plot(Pi, sigma, name+"_infer.pdf" if name else None)


class GaussianMix(DataGenerator):
    def __init__(self, 
                 nmbrs=[[0,4],[0,6]],
                 cntrs=[[0.00, 0.00], [4.00, 16.0]],
                 wdths=[[0.04, 0.40], [0.04, 0.40]],
                 wgths=[[0.00, 1.00], [0.00, 1.00]],
                 norm=1, even=True, anormal=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.nmbrs = nmbrs
        self.cntrs = cntrs
        self.wdths = wdths
        self.wgths = wgths
        self.norm = norm
        self.even = even
        self.anormal = anormal
        self.tmp_num_per_group = None

    def new_random_num_per_group(self):
        num_per_group = [np.random.randint(n[0], n[1]+1) for n in self.nmbrs]
        if all(num_per_group) == 0:
            lucky_group = np.random.randint(0,len(num_per_group)-1)
            num_per_group[lucky_group] = 1
        self.tmp_num_per_group = num_per_group

        return num_per_group

    def random_cwh(self):
        cl, wl, hl = [], [], []
        for i, n in enumerate(self.tmp_num_per_group):
            cl.append(np.random.uniform(self.cntrs[i][0], self.cntrs[i][1], n))
            wl.append(np.random.uniform(self.wdths[i][0], self.wdths[i][1], n))
            hl.append(np.random.uniform(self.wgths[i][0], self.wgths[i][1], n))
        c = np.hstack(cl)
        w = np.hstack(wl)
        h = np.hstack(hl)

        if self.even:
            c = np.hstack([-c, c])
            w = np.hstack([w, w])
            h = np.hstack([h, h])

        if self.anormal:
            h *= w  # In some papers the gaussians are not normalized
        if self.norm:
            h *= np.pi*self.norm/(h.sum()+SMALL)

        return c, w, h

    def generate_functions(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        sigma_func = lambda x: sum_on_args(gaussian, x, c, w, h)
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=self.wmax)

        return sigma_func, pi_func


class LorentzMix(GaussianMix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_functions(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        sigma_func = lambda x: sum_on_args(lorentzian, x, c, w, h)
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=self.wmax)
        return sigma_func, pi_func


class BetaMix(GaussianMix):
    def __init__(self, 
                 arngs=[[2.00, 5.00], [0.50, 5.00]],
                 brths=[[2.00, 5.00], [0.50, 5.00]],
                 **kwargs):
        super().__init__(**kwargs)
        self.arngs = arngs
        self.brths = brths

    def random_ab(self):
        al, bl = [], []
        for i, n in enumerate(self.tmp_num_per_group):
            al.append(np.random.uniform(self.arngs[i][0], self.arngs[i][1], n))
            bl.append(np.random.uniform(self.brths[i][0], self.brths[i][1], n))
        a = np.hstack(al)
        b = np.hstack(bl)
        
        if self.even:
            aa, bb = a, b
            a = np.hstack([aa, bb])
            b = np.hstack([bb, aa])

        return a, b

    def generate_functions(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        a, b = self.random_ab()
        sigma_func = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=self.wmax)
        return sigma_func, pi_func
    

class LorentzComb(DataGenerator):
    def __init__(self, norm=1, num_peaks=1000, width=0.05, **kwargs):
        super().__init__(**kwargs)
        self.norm = norm
        self.num_peaks = num_peaks
        self.width = width

    def generate_functions(self):
        k = np.linspace(0, 1, self.num_peaks)
        # c = monofunc.piecewise_gap(k, n=8, soft=0.05, xlims=[0,1], ylims=[0,0.8*self.wmax])
        c = monofunc.random_climb(k, xlims=[0, 1], ylims=[0, 0.8*self.wmax])
        w = np.ones(self.num_peaks)*self.width
        h = abs(c) + 0.05
        h *= self.norm/(2*h*c/(c**2+w**2)).sum()
        sigma_func = lambda x: sum_on_args(even_lorentzian, x, c, w, h)
        pi_func = lambda x: sum_on_args(analytic_pi, x, c, w, h)
        return sigma_func, pi_func


class ContinuationData(torch.utils.data.Dataset):
    def __init__(self, path, noise=0.0, beta=[], rescaled=False):
        self.x_data = np.loadtxt(open(path+"Pi.csv", "rb"), delimiter=",")
        self.y_data = np.loadtxt(open(path+"SigmaRe.csv", "rb"), delimiter=",")
        self.noise = noise
        
        self.beta = beta
        if self.beta:
            self.xT_data = [self.x_data]
            for t in beta[1:]:
                self.xT_data.append(
                    np.loadtxt(open(path+f"Pi_beta_{t}.csv", "rb"), delimiter=",")
                )
            self.xT_data = np.stack(self.xT_data)
            self.nT = len(self.xT_data)

        self.rescaled = rescaled
        if self.rescaled:
            scaled_path = path+"SigmaRe_scaled_4.0.csv"
            self.scaled_y_data = np.loadtxt(open(scaled_path, "rb"), delimiter=",")
            self.wmaxs = np.loadtxt(open(path+"wmaxs.csv", "rb"), delimiter=",")
        
        self.N = self.y_data.shape[-1]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        if self.beta:
            x = self.xT_data[np.random.randint(self.nT), index]
            x += np.random.normal(0,1, size=x.shape)*self.noise
        else:
            x = self.x_data[index] 
            x += np.random.normal(0,1, size=x.shape)*self.noise

        if self.rescaled:
            y = self.scaled_y_data[index]
        else: 
            y = self.y_data[index]
        return x, y


def main():
    default_args = {
        'seed': int(time.time()),
        'plot': 0,
        'generate': 0,
        'path': str(HERE),
        'Nwn': 128,
        'Nw': 512,
        'wmax': 20.0,
        'beta': [4, 7, 10, 13, 16, 19, 22, 25, 28],  # 2*np.pi, # 2pi/beta = 1
        'norm': 1.0,
        'rescale': 0.0,
        # peaks
        "variant": "Gaussian",
        "anormal": False,
        "wmax": 20.0,
        "nmbrs": [[0, 4],[0, 6]],
        "cntrs": [[0.00, 0.00], [4.00, 16.0]],
        "wdths": [[0.40, 4.00], [0.40, 4.00]],
        "wghts": [[0.00, 1.00], [0.00, 1.00]],
        "arngs": [[2.00, 10.00], [0.70, 10.00]],
        "brths": [[2.00, 10.00], [0.70, 10.00]],
        "even": True,
        # lorentz
        'num_peaks': 10000,
        'width': 0.05,
        # plot
        'plot_name': "",
        'basic_plot': True,
        'scaled_plot': False,
        'infer_scale': False,
    }
    args = utils.parse_file_and_command(default_args, {})
    print(f"seed : {args.seed}")
    np.random.seed(args.seed)

    if args.variant in ["G", "Gaussian", "gaussian"]:
        generator = GaussianMix(**vars(args))
    elif args.variant in ["B", "Beta", "beta"]:
        generator = BetaMix(**vars(args))
    elif args.variant in ["L", "Lorentzian", "lorentzian"]:
        generator = LorentzMix(**vars(args))
    elif args.variant in ["LC", "Lorentz_comb", "lorentz_comb"]:
        generator = LorentzComb(**vars(args))
    else:
        raise ValueError(f"variant {args.variant} not recognized")

    if args.plot > 0:
        generator.plot(
            args.plot,
            name=args.plot_name if args.plot_name else None,
            basic=args.basic_plot,
            scale=args.scaled_plot,
            infer=args.infer_scale
        )

    elif args.generate > 0:
        os.makedirs(args.path, exist_ok=True)
        generator.generate_files(
            args.generate,
            pi_path=args.path+'/Pi.csv',
            sigma_path=args.path+'/SigmaRe.csv',
            wmaxs_path=args.path+'/wmaxs.csv' if args.rescale > SMALL else None
        )

    if args.generate == 0 and args.plot == 0:
        print("nothing to do, use --plot 10 or --generate 10000")
    
if __name__ == "__main__":
    main()
