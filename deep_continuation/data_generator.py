#!/usr/bin/env python
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from deep_continuation import utils
from deep_continuation.function_generator import (
    default_parameters,
    rescaling,
    SigmaPiGenerator,
)

np.set_printoptions(precision=4)
HERE = Path(__file__).parent
SMALL = 1e-10
COLORS = list(mcolors.TABLEAU_COLORS)


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


def scale_plot(Pi, sigma, betas, wmaxs, filename=None, default_wmax=20.0):
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
        ax[0, 1].plot(w[i], (default_wmax/wmaxs[i])*sigma[i], c=COLORS[i%10])
        ax[1, 1].plot(w[i], cumul_sum2[i], c=COLORS[i%10])

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def infer_scale_plot(Pi, sigma, filename=None, default_wmax=20.0):
    wmaxs, betas = infer_scales(Pi, sigma)
    print(f" infered scales:\n  betas =\n{betas}\n  wmaxs =\n{wmaxs}")
    scale_plot(Pi, sigma, betas, wmaxs, filename, default_wmax)


class DataGenerator():
    def __init__(self, Nwn=128, Nw=512, beta=[10.0], wmax=20, rescale=0.0, **kwargs):
        self.Nwn = Nwn
        self.Nw = Nw
        self.beta = beta
        self.wmax = wmax
        self.rescale = rescale
        # note that rescale is not passed to the factory below, so the base SigmaPiIntegrator is used
        self.generator = SigmaPiGenerator.factory(wmax=wmax, **kwargs)

    def generate_batch(self, size):
        Pi = np.zeros((len(self.beta), size, self.Nwn))
        betas = np.zeros((len(self.beta), size))
        sigma = np.zeros((size, self.Nw))
        sigma_r = np.zeros((size, self.Nw))
        wmaxs = np.zeros(size)
        
        for i in range(size):
            if (i == 0 or (i+1)%(max(1, size//100)) == 0): print(f"{i+1}/{size}")

            sigma_func, pi_func = self.generator.generate()
            omega = np.linspace(0, self.wmax, self.Nw)
            sigma[i] = sigma_func(omega)
            sigma[i] *= (2*self.wmax)/(self.Nw*np.pi)  # so that sum=1  (x2 because half spectrum, (wmax/pi)/Nw is the integral discretizedcod
            wmaxs[i] = self.wmax

            for b, beta in enumerate(self.beta):
                omega_n = np.arange(0, self.Nwn)*2*np.pi/beta        
                Pi[b,i] = pi_func(omega_n)
                betas[b,i] = beta
            
            if self.rescale > SMALL:
                sigma_r_func, new_wmax = rescaling(sigma_func, self.wmax, self.rescale)
                sigma_r[i] = sigma_r_func(omega)
                sigma_r[i] *= (2*self.wmax)/(self.Nw*np.pi)  # so that sum=1
                wmaxs[i] = new_wmax
            
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
        print(f" Pi normalization check\n{Pi[:,:,0]}")
        print(f" sigma normalization check\n{sigma.sum(-1)}")

        if self.rescale:
            sigma = sigma_r
        if basic:
            unscaled_plot(Pi, sigma, name+"_basic.pdf" if name else None)
        if scale:
            scale_plot(Pi, sigma, betas, wmaxs, 
                name +"_scale.pdf" if name else None, 
                default_wmax=self.wmax
            )
        if infer:
            infer_scale_plot(Pi, sigma,
                name+"_infer.pdf" if name else None,
                default_wmax=self.wmax
            )


def main():
    default_parameters.update({
        'plot': 0,
        'generate': 0,
        'path': str(HERE),
        'Nwn': 128,
        'Nw': 512,
        'wmax': 20.0,
        'beta': [20.0],#[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 50.0],  # 2*np.pi, # 2pi/beta = 1
        'norm': 1.0,
        'rescale': 0.0,
        # plot
        'plot_name': "",
        'basic_plot': True,
        'scaled_plot': False,
        'infer_scale': False,
    })
    args = utils.parse_file_and_command(default_parameters, {})
    print(f"seed : {args.seed}")
    np.random.seed(args.seed)

    generator = DataGenerator(**vars(args))

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
