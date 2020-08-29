#%%

import os
import time
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.special import binom, gamma, factorial, hyp2f1
import matplotlib.pyplot as plt

from deep_continuation import utils
from deep_continuation import monotonous_functions as monofunc

np.set_printoptions(precision=3)
HERE = Path(__file__).parent
SMALL = 1e-10


def gaussian(x, c, w, h):
    return (h/(np.sqrt(np.pi)*w))*np.exp(-((x-c)/w)**2)


def lorentzian(x, c, w, h):
    return (h/np.pi)*w/((x-c)**2+w**2)


def bernstein(x, m, n):
    return binom(m, n) * (x**n) * ((1-x)**(m-n)) * (x >= 0) * (x <= 1)


def free_bernstein(x, m, n, c=0, w=1, h=1):
    sq = np.sqrt(m+1)
    xx = (x-c)/(w*sq) + n/m
    return (h*sq/w)*bernstein(xx, m, n)


def bernstein_center(m, n):
    # from mathematica
    return (1+n)/(2+m)


def bernstein_width(m, n):
    # from mathematica
    return np.sqrt(-((1+n)**2/(2+m)**2)+((1+n)*(2+n))/((2+m)*(3+m)))


def centered_bernstein(x, m, n):
    c = bernstein_center(m,n)
    return (m+1)*bernstein(x+c, m, n)


def standardized_bernstein(x, m, n):
    w = bernstein_width(m,n)
    return centered_bernstein(x*w, m, n)*w


def fbernstein_norm(m, n, N=10000):
    x = np.linspace(-10, 10, N)
    return integrate.simps(standardized_bernstein(x, m, n), x)


def fbernstein_avg(m, n, N=10000):
    x = np.linspace(-10, 10, N)
    return integrate.simps(x*standardized_bernstein(x, m, n), x)


def fbernstein_std(m, n, N=10000):
    x = np.linspace(-10, 10, N)
    return integrate.simps(x**2*standardized_bernstein(x, m, n), x)


def peak(w, center=0, width=1, height=1, type_m=0, type_n=0):
    out = 0
    out += (type_m == 0) * lorentzian(w, center, width, height)
    out += (type_m == 1) * gaussian(w, center, width, height)
    out += (type_m >= 2) * free_bernstein(w, type_m, type_n, center, width, height)
    return out


def even_lorentz(x, c=0, w=1, h=1):
    return (1/np.pi)*4*c*w*h/(((x-c)**2+w**2)*((x+c)**2+w**2))


def integrated_even_lorentz(x, c=0, w=0, h=0):
    return 2*h*c/(c**2+(x+w)**2)


def sum_on_args(f, x, *args):
    if isinstance(x, np.ndarray):
        x = x[np.newaxis, :]
        args = [a for a in args] # copy args to allow reassign 
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                while len(args[i].shape) < len(x.shape):
                    args[i] = args[i][:, np.newaxis]
    return f(x,*args).sum(axis=0)


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

    def integrand(x): return (1/np.pi) * x**2 / \
        (x**2+wn**2) * spectral_function(x)
    return integrate_with_tails(integrand)


def second_moment(spectral_function):
    def integrand(x): return (1/np.pi) * x**2 * spectral_function(x)
    return integrate_with_tails(integrand)


class LorentzGenerator():
    def __init__(self, out_size, in_size, beta, w_max, Pi0, seed, rescale,
                 num_peaks, N_seg, peak_widths, center_method, remove_nonphysical, **args):
        self.num_w = out_size
        self.num_wn = in_size
        self.beta = beta
        self.w_max = w_max

        self.w = np.linspace(0, self.w_max, self.num_w)
        self.wn = (2*np.pi/self.beta) * np.arange(0, self.num_wn)

        self.Pi0 = Pi0
        self.seed = seed
        self.rescale = rescale

        self.num_peaks = num_peaks
        self.N_seg = N_seg
        self.peak_widths = peak_widths
        self.center_method = center_method
        self.remove_nonphysical = remove_nonphysical

    def distributed_peaks_parameters(self):
        if self.center_method == -1:
            if self.remove_nonphysical == True:
                method = np.random.randint(1, 8)
            else:
                method = np.random.randint(1, 11)
        else:
            method = self.center_method

        k = np.linspace(0, self.w_max, self.num_peaks)

        if method == 0:
            center = monofunc.piecelin(k, self.N_seg)
        elif method == 1:
            center = monofunc.softp(k, self.N_seg)
        elif method == 2:
            center = monofunc.arctsum(k, self.N_seg)
        elif method == 3:
            center = monofunc.erfsum(k, self.N_seg)
        elif method == 4:
            center = monofunc.arssum(k, self.N_seg)
        elif method == 5:
            center = monofunc.rootsum(k, self.N_seg)
        elif method == 6:
            center = monofunc.exparsinh(k, self.N_seg)
        elif method == 7:
            center = monofunc.exparctan(k, self.N_seg)
        elif method == 8:
            center = monofunc.arssoft(k, self.N_seg)
        elif method == 9:
            center = monofunc.tanerf(k, self.N_seg)
        elif method == 10:
            center = monofunc.logarc(k, self.N_seg)

        center -= center[0]
        # random gap
        center += np.random.choice([0,abs(np.random.normal(scale=self.w_max/8, size=1))])
        center *= self.w_max/center[-1]

        width = np.ones(self.num_peaks)*self.peak_widths
        # countering 1/w
        height = abs(center) + 0.05

        normalizer_vector = 2*height*center/(center**2+width**2)
        height /= normalizer_vector.sum(axis=-1, keepdims=True)
        height *= self.Pi0
        return center, width, height

    def generate_batch(self, size):
        Pi = np.zeros([size, self.num_wn])
        sigma = np.zeros([size, self.num_w])
        for i in range(size):
            if (i == 0 or (i+1) % (max(1, size//100)) == 0):
                print(f"sample {i+1}")

            c, w, h = self.distributed_peaks_parameters()
            def sigma_func(x): return sum_on_args(even_lorentz, x, c, w, h)
            Pi[i] = sum_on_args(integrated_even_lorentz, self.wn, c, w, h)

            if self.rescale > SMALL:
                inf = 1e6
                s = np.sqrt(inf**2*sum_on_args(integrated_even_lorentz, inf, c, w, h))
                new_w_max = self.rescale*s
                resampl_w = np.linspace(0, new_w_max, self.num_w)
                sigma[i] = s*sigma_func(resampl_w)
            else:
                sigma[i] = sigma_func(self.w)

        return Pi, sigma


class GaussBernsteinGenerator():
    def __init__(self, out_size, in_size, beta, w_max, Pi0, use_bernstein,
                 max_drude, max_peaks, weight_ratio, peak_pos, peak_width, drude_width,
                 seed, rescale, **kwargs):
        self.num_w = out_size
        self.num_wn = in_size
        self.beta = beta
        self.w_max = w_max

        self.w = np.linspace(0, self.w_max, self.num_w)
        self.wn = (2*np.pi/self.beta) * np.arange(0, self.num_wn)

        self.Pi0 = Pi0
        self.bernstein = use_bernstein
        self.max_drude = max_drude
        self.max_peaks = max_peaks
        self.ratio = weight_ratio
        self.min_c, self.max_c = peak_pos
        self.min_w, self.max_w = peak_width
        self.min_dw, self.max_dw = drude_width

        self.seed = seed
        self.rescale = rescale

    def random_peak_args(self):
        num_drude = np.random.randint(
            0 if self.max_peaks > 0 else 1,
            self.max_drude+1
        )
        num_others = np.random.randint(
            0 if num_drude > 0 else 1,
            self.max_peaks+1
        )
        num = num_drude + num_others
        ratio = np.random.uniform(SMALL, self.ratio)

        # centers
        c = np.random.uniform(self.min_c, self.max_c, size=num)
        c[:num_drude] = 0.0
        c = np.hstack([c, -c])*self.w_max

        # width
        w = np.random.uniform(0.0, 1.0, size=num)
        w[:num_drude] *= self.max_dw-self.min_dw
        w[:num_drude] += self.min_dw
        w[num_drude:] *= self.max_w-self.min_w
        w[num_drude:] += self.min_w
        w = np.hstack([w, w])*self.w_max

        # heighs
        h = np.random.uniform(0.0, 1.0, size=num)
        h[:num_drude] *= ratio/(h[:num_drude].sum() + SMALL)
        h[num_drude:] *= (1-ratio)/(h[num_drude:].sum() + SMALL)
        h = np.hstack([h, h])
        h /= h.sum(axis=-1, keepdims=True)
        h *= self.Pi0 * np.pi

        # berstein selectors
        if self.bernstein:
            m = np.random.randint(2, 20, size=num)
            n = np.ceil(np.random.uniform(0.0, 1.000, size=num)*(m-1))
        else:
            m = np.ones(num)
            n = np.ones(num)
        n = np.hstack([n, m-n])
        m = np.hstack([m, m])

        return c, w, h, m, n

    def generate_batch(self, size):
        Pi = np.zeros((size, self.num_wn))
        sigma = np.zeros((size, self.num_w))
        for i in range(size):

            # np.random.seed(self.seed)
            # self.beta = self.beta/2
            # self.wn = (2*np.pi/self.beta) * np.arange(0,self.num_wn)

            c, w, h, m, n = self.random_peak_args()
            def sigma_func(x): return sum_on_args(peak, x, c, w, h, m, n)
            Pi[i] = pi_integral(self.wn, sigma_func)

            if self.rescale > SMALL:
                inf = 1e6
                s = np.sqrt(inf**2*pi_integral(inf, sigma_func))
                new_w_max = self.rescale*s
                resampl_w = np.linspace(0, new_w_max, self.num_w)
                sigma[i] = s*sigma_func(resampl_w)
            else:
                sigma[i] = sigma_func(self.w)

        return Pi, sigma


def unscaled_plot(Pi, sigma):
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
    plt.show()


def infer_scale_plot(Pi, sigma, ):
    N = len(Pi[0])
    Pi0 = Pi[:, 0]
    PiN = Pi[:, -1]
    M = len(sigma[0])
    sum1 = np.sum(sigma, axis=-1)
    sum2 = np.sum(np.arange(M), axis=-1)

    beta = 2*N*np.sqrt(PiN*sum1**3/(Pi0**3*sum2))
    wn = (2*np.pi/beta[:, np.newaxis]) * np.arange(N)
    w_max = np.pi*Pi0*M/sum1
    w = w_max[:, np.newaxis] * np.linspace(0, 1, M)

    n2Pi = wn**2*Pi
    cumul_sum2 = np.cumsum(w**2*sigma, axis=-1)

    print(f'''
    beta  = {beta}
    w_max = {w_max}
    ''')

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
    plt.show()


def main():
    default_args = {
        'plot': 0,
        'generate': 0,
        'path': str(HERE),
        'in_size': 128,
        'out_size': 512,
        'w_max': 20.0,
        'beta': 10.0,  # 2*np.pi, # 2pi/beta = 1
        'Pi0': 1.0,
        'use_bernstein': False,
        'max_drude': 4,
        'max_peaks': 6,
        'weight_ratio': 0.50,
        'drude_width': [.02, .1],
        'peak_pos': [.2, .8],
        'peak_width': [.05, .1],
        'seed': int(time.time()),
        # lorentz
        'rescale': 0.0,
        'lorentz': False,
        'num_peaks': int(2048),
        'peak_widths': 0.02,
        'N_seg': 8,
        'center_method': -1,
        'remove_nonphysical': False,
        'scaled_plot': False
    }
    args = utils.parse_file_and_command(default_args, {})
    np.random.seed(args.seed)
    print(f"seed : {args.seed}")

    if args.lorentz:
        generator = LorentzGenerator(**vars(args))
    else:
        generator = GaussBernsteinGenerator(**vars(args))

    if args.plot > 0:
        print(f"ploting {args.plot}")
        Pi, sigma = generator.generate_batch(size=args.plot)
        if args.scaled_plot:
            infer_scale_plot(Pi, sigma)
        else:
            unscaled_plot(Pi, sigma)
    
    if args.generate > 0:
        sigma_path = args.path+'SigmaRe.csv'
        pi_path = args.path+'Pi.csv'
        if (os.path.exists(sigma_path) or os.path.exists(pi_path)):
            raise ValueError('ABORT GENERATION: there is already a dataset on this path')
        
        print(f"generating {args.generate}")
        os.makedirs(args.path, exist_ok=True)
        Pi, sigma = generator.generate_batch(size=args.generate)
        np.savetxt(f"{args.path}/Pi.csv"     , Pi   , delimiter=',')
        np.savetxt(f"{args.path}/SigmaRe.csv", sigma, delimiter=',')
        
        if args.plot > 0:
            print('WARNING: examples printed are not part of the dataset')

    if args.generate==0 and args.plot==0:
        print("nothing to do, use --plot 10 or --generate 10000")


if __name__ == "__main__":
    main()
