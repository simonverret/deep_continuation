import os
import time
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.special import binom, gamma
import matplotlib.pyplot as plt

from deep_continuation import utils
from deep_continuation import monotonous_functions as monofunc

np.set_printoptions(precision=4)
HERE = Path(__file__).parent
SMALL = 1e-10


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


def unnormed_gaussian(x, c, w, h):
    return np.exp(-((x-c)/w)**2/2)


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


def scale_plot(Pi, sigma, beta, wmax, filename=None):
    N = len(Pi[0])
    M = len(sigma[0])
    wn = (2*np.pi/beta[:, np.newaxis]) * np.arange(N)
    w = wmax[:, np.newaxis] * np.linspace(0, 1, M)
    n2Pi = wn**2*Pi
    cumul_sum2 = np.cumsum(w**2*sigma, axis=-1)

    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi(i\omega_n)$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{\omega_n^2 \Pi(i\omega_n)}$")
    ax[1, 0].set_xlabel(r"$\omega_n$")
    ax[0, 1].set_ylabel(r"$\sigma(\omega)$")
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
    M = len(sigma[0])
    norm = Pi[:, 0]
    PiN = Pi[:, -1]
    sum1 = 2*np.sum(sigma, axis=-1) - np.take(sigma, 0, axis=-1)
    dm = np.pi*norm/sum1
    wmaxs = M*dm
    m = np.arange(M)
    sum2 = 2*np.sum(m**2*sigma, axis=-1)
    betas = 2*N*np.sqrt((np.pi**3)*PiN/(dm**3*sum2))
    print(
        f" infered pieces:\n  PiN  = {PiN}\n  norm  = {norm}\n  sum1 = {sum1}\n  sum2 = {sum2}")
    print(f" infered scales:\n  betas = {betas}\n  wmaxs = {wmaxs}")
    scale_plot(Pi, sigma, betas, wmaxs, filename)


class DataGenerator():
    def __init__(self, Nwn=128, Nw=512, beta=10, wmax=20, rescale=0, **kwargs):
        self.num_wn = Nwn
        self.num_w = Nw
        self.beta = beta
        self.wmax = wmax
        self.w = np.linspace(0, self.wmax, self.num_w)
        self.wn = (2*np.pi/self.beta) * np.arange(0, self.num_wn)
        self.rescale = rescale

    def generate_functions(self):
        raise NotImplementedError

    def generate_batch(self, size):
        Pi = np.zeros((size, self.num_wn))
        sigma = np.zeros((size, self.num_w))
        betas = np.zeros(size)
        wmaxs = np.zeros(size)

        for i in range(size):
            if (i == 0 or (i+1) % (max(1, size//100)) == 0):
                print(f"sample {i+1}")
            sigma_func, pi_func = self.generate_functions()

            Pi[i] = pi_func(self.wn)

            if self.rescale > SMALL:
                inf = 1e6
                s = np.cbrt(
                    inf**2*pi_integral(inf, sigma_func, grid_end=self.wmax))
                # N = self.num_wn-1
                # s = np.cbrt(Pi[i,N]*N*2)
                new_w_max = self.rescale*s
                resampl_w = np.linspace(0, new_w_max, self.num_w)
                sigma[i] = sigma_func(resampl_w)
                betas[i] = self.beta
                wmaxs[i] = new_w_max
            else:
                sigma[i] = sigma_func(self.w)
                betas[i] = self.beta
                wmaxs[i] = self.wmax

        return Pi, sigma, betas, wmaxs

    def generate_files(self, size, sigma_path, pi_path, scale_path=None):
        if (os.path.exists(sigma_path) or os.path.exists(pi_path)):
            raise ValueError('there is already a dataset on this path')
        Pi, sigma, betas, wmaxs = self.generate_batch(size)
        np.savetxt(pi_path, Pi, delimiter=',')
        np.savetxt(sigma_path, sigma, delimiter=',')
        if scale_path:
            scales = np.vstack(betas, wmaxs)
            np.savetxt(scale_path, scales, delimiter=',', header="beta, wmax")

    def plot(self, size, name=None, basic=True, scale=False, infer=False):
        Pi, sigma, betas, wmaxs = self.generate_batch(size)
        print(f" true scales:\n  betas = {betas}\n  wmaxs = {wmaxs}")
        print(f" normalization check {Pi[:,0]}")
        if basic:
            unscaled_plot(Pi, sigma, name+"_basic.pdf" if name else None)
        if scale:
            scale_plot(Pi, sigma, betas, wmaxs, name +
                       "_scale.pdf" if name else None)
        if infer:
            infer_scale_plot(Pi, sigma, name+"_infer.pdf" if name else None)


class GaussianNestedMix(DataGenerator):
    def __init__(self, 
                 nmbrs=[[0,4],[0,6]],
                 cntrs=[[0.00, 0.00], [4.00, 16.0]],
                 wdths=[[0.04, 0.40], [0.04, 0.40]],
                 wgths=[[0.00, 1.00], [0.00, 1.00]],
                 norm=1, even=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.nmbrs = nmbrs
        self.cntrs = cntrs
        self.wdths = wdths
        self.wgths = wgths
        self.norm = norm
        self.even = even
        
        self.tmp_num_per_group = None

    def new_random_num_per_group(self):
        num_per_group = [np.random.randint(n[0], n[1]) for n in self.nmbrs]
        if all(num_per_group) == 0:
            lucky_group = np.random.randint(0,len(num_per_group)-1)
            num_per_group[lucky_group] = 1
        self.tmp_num_per_group = num_per_group
        return num_per_group

    def random_cwh(self):
        for grp, num in enumerate(self.tmp_num_per_group):
            c = [np.random.uniform(self.cntrs[grp][0], self.cntrs[grp][1], num)]
            w = [np.random.uniform(self.wdths[grp][0], self.wdths[grp][1], num)]
            h = [np.random.uniform(self.wgths[grp][0], self.wgths[grp][1], num)]
            c = np.hstack(c)
            w = np.hstack(w)
            h = np.hstack(h)
        
        if self.even:
            c = np.hstack([-c, c])
            w = np.hstack([w, w])
            h = np.hstack([h, h])

        if self.norm is not None:
            h *= self.norm/(h.sum()+SMALL)

        return c, w, h

    def generate_functions(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        sigma_func = lambda x: sum_on_args(gaussian, x, c, w, h)
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=self.wmax)
        return sigma_func, pi_func


class BetaNestedMix(GaussianNestedMix):
    def __init__(self, 
                 arngs=[[0.00, 0.00], [4.00, 16.0]],
                 brths=[[0.04, 0.40], [0.04, 0.40]],
                 **kwargs):
        super().__init__(**kwargs)
        self.arngs = arngs
        self.brths = brths

    def random_ab(self):
        for grp, num in enumerate(self.tmp_num_per_group):
            a = [np.random.uniform(self.cntrs[grp][0], self.cntrs[grp][1], num)]
            b = [np.random.uniform(self.wdths[grp][0], self.wdths[grp][1], num)]
            a = np.hstack(c)
            b = np.hstack(w)
        
        if self.even:
            aa, bb = a, b
            a = np.hstack([aa, bb])
            b = np.hstack([bb, aa])

        return a, b

    def generate_functions(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        a, b = random_ab()
        sigma_func = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=self.wmax)
        return sigma_func, pi_func
    

class PeakMix(DataGenerator):
    def __init__(self,
                 Nwn, Nw, beta, wmax, rescale,
                 peak_type, norm, max_peaks, peaks_position_range, 
                 peaks_width_range, peaks_weight_range, beta_ab_range,
                 **kwargs
                ):
        super().__init__(Nwn, Nw, beta, wmax, rescale)
        self.peak_type = peak_type
        self.norm = norm
        self.max_peaks = max_peaks
        self.peaks_position_range = peaks_position_range
        self.peaks_width_range = peaks_width_range
        self.peaks_weight_range = peaks_weight_range
        self.beta_ab_range = beta_ab_range


    def generate_cwhab(self):
        num = np.random.randint(1, self.max_peaks+1)
        c, w, h = random_cwh(
            num, self.peaks_position_range, self.peaks_width_range, self.peaks_weight_range, self.norm
        )
        c = c*self.wmax
        w = w*self.wmax
        h = h*self.norm*np.pi
        a, b = random_ab(num, ra=self.beta_ab_range, rb=self.beta_ab_range)
        return c, w, h, a, b

    def generate_functions(self):
        c, w, h, a, b = self.generate_cwhab()
        if self.peak_type in ["G", "Gaussian", "gaussian"]:
            def sigma_func(x): return sum_on_args(gaussian, x, c, w, h)
        if self.peak_type in ["UG", "unnormed_Gaussian", "unnormed_gaussian"]:
            def sigma_func(x): return sum_on_args(unnormed_gaussian, x, c, w, h)
        elif self.peak_type in ["B", "Beta", "beta"]:
            def sigma_func(x): return sum_on_args(free_beta, x, c, w, h, a, b)
        elif self.peak_type in ["L", "Lorentzian", "lorentzian"]:
            def sigma_func(x): return sum_on_args(lorentzian, x, c, w, h)
        else:
            raise ValueError(f"peak_type {self.peak_type} not recognized")

        def pi_func(x): return pi_integral(x, sigma_func, grid_end=self.wmax)
        return sigma_func, pi_func


class DrudePeakMix(PeakMix):
    def __init__(self,
                 Nwn, Nw, beta, wmax, rescale,
                 peak_type, norm, max_peaks, peaks_position_range,
                 peaks_width_range, peaks_weight_range, beta_ab_range,
                 min_drude, max_drude, drude_ratio, drude_width_range, drude_weight_range,
                 **kwargs
                 ):
        super().__init__(Nwn, Nw, beta, wmax, rescale,
                         peak_type, norm, max_peaks, peaks_position_range, 
                         peaks_width_range, peaks_weight_range, beta_ab_range)
        self.min_drude = min_drude
        self.max_drude = max_drude
        self.max_peaks = max_peaks
        self.drude_ratio = drude_ratio
        self.drude_width_range = drude_width_range
        self.drude_weight_range = drude_weight_range


    def generate_cwhab(self):
        if self.max_peaks > 0:
            min_drude = self.min_drude
        else:
            min_drude = np.max(1, self.min_drude)
        drudes = np.random.randint(min_drude, self.max_drude+1)
        others = np.random.randint(0 if drudes > 0 else 1, self.max_peaks+1)

        if drudes and others:
            drude_weight = np.random.choice(
                [0, np.random.uniform(0, self.drude_ratio)]
            )
            rest = 1-drude_weight
        else:
            drude_weight = 1
            rest = 1

        c1, w1, h1 = random_cwh(
            drudes, [0, 0], self.drude_width_range, self.drude_weight_range, drude_weight
        )
        c2, w2, h2 = random_cwh(
            others, self.peaks_position_range, self.peaks_width_range, self.peaks_weight_range, rest
        )
        c = np.hstack([c1, c2])*self.wmax
        w = np.hstack([w1, w2])*self.wmax
        h = np.hstack([h1, h2])*self.norm*np.pi

        a1, b1 = random_ab(drudes, ra=self.beta_ab_range, rb=self.beta_ab_range)
        a2, b2 = random_ab(others, ra=self.beta_ab_range, rb=self.beta_ab_range)
        a = np.hstack([a1, a2])
        b = np.hstack([b1, b2])

        return c, w, h, a, b


class LorentzComb(DataGenerator):
    def __init__(self,
                 Nwn, Nw, beta, wmax, rescale,
                 norm, num_peaks, width, 
                 **kwargs
                 ):
        super().__init__(Nwn, Nw, beta, wmax, rescale)
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
        def sigma_func(x): return sum_on_args(even_lorentzian, x, c, w, h)
        def pi_func(x): return sum_on_args(analytic_pi, x, c, w, h)
        return sigma_func, pi_func


def main():
    default_args = {
        'seed': int(time.time()),
        'plot': 0,
        'generate': 0,
        'path': str(HERE),
        'Nwn': 128,
        'Nw': 512,
        'wmax': 20.0,
        'beta': 10.0,  # 2*np.pi, # 2pi/beta = 1
        'norm': 1.0,
        'rescale': 0.0,
        # peaks
        'peak_type': "Gaussian",
        'max_peaks': [[0,4], [0,10]],
        'peaks_position_range': [.2, .8],
        'peaks_width_range': [.01, .1],
        'peaks_weight_range': [0.0,1],
        'beta_ab_range': [0.5,10],
        # drude
        'min_drude': 0,
        'max_drude': 4,
        'drude_ratio': 0.50,
        'drude_width_range': [.01, .05],
        'drude_weight_range': [0.0, 1],
        # lorentz
        'lorentz': False,
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

    if args.lorentz:
        print("YESS")
        generator = LorentzComb(
            args.Nwn, args.Nw, args.beta, args.wmax, args.rescale,
            args.norm, args.num_peaks, args.width,
        )
    elif args.max_drude:
        generator = DrudePeakMix(
            args.Nwn, args.Nw, args.beta, args.wmax, args.rescale,
            args.peak_type, args.norm, args.max_peaks, args.peaks_position_range,
            args.peaks_width_range, args.peaks_weight_range, args.beta_ab_range,
            args.min_drude, args.max_drude, args.drude_ratio, args.drude_width_range, args.drude_weight_range
        )
    else:
        generator = PeakMix(
            args.Nwn, args.Nw, args.beta, args.wmax, args.rescale,
            args.peak_type, args.norm, args.max_peaks, args.peaks_position_range,
            args.peaks_width_range, args.peaks_weight_range, args.beta_ab_range
        )

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
        )

    if args.generate == 0 and args.plot == 0:
        print("nothing to do, use --plot 10 or --generate 10000")
    
    print(args.max_peaks)

if __name__ == "__main__":
    main()
