import numpy as np
from scipy import integrate
from scipy.special import gamma
import matplotlib.pyplot as plt

from deep_continuation import utils
from deep_continuation import monotonous_functions as monofunc


SMALL = 1e-10
INF = 1e10

default_parameters = {
    # peaks
    "variant": "B",
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
}

def main():
    args = utils.parse_file_and_command(default_parameters, {})
    generator = SigmaPiGenerator.factory(**vars(args))

    np.random.seed(111)
    sigma_func, pi_func = generator.generate()
    
    wmax_list = [20]
    M = 512
    beta_list = [20,30, 40, 50, 60]
    N = 128

    wn = np.array([np.arange(0, N)*2*np.pi/beta for beta in beta_list])
    Pi = np.array([pi_func(np.arange(0, N)*2*np.pi/beta) for beta in beta_list])

    fig, ax = plt.subplots(1, 3, figsize=[10, 5])
    ax[0].set_ylabel(r"$\Pi(i\omega_n)$")
    ax[0].set_xlabel(r"$\omega_n$")
    ax[0].plot(wn.T, Pi.T, '.')

    ax[1].set_xlabel(r"$\omega_n$")
    ax[1].plot(Pi.T, '.')

    w = np.array([np.linspace(-wmax, wmax, 2*M+1) for wmax in wmax_list])
    sigma = np.array([(wmax/20)*sigma_func(np.linspace(-wmax, wmax, 2*M+1)) for wmax in wmax_list])
    
    ax[2].set_ylabel(r"$\sigma(\omega)$")
    ax[2].set_xlabel(r"$\omega$")
    ax[2].plot(sigma.T)
    
    fig.tight_layout()
    plt.show()

    # s = INF**2*pi_integral(INF, sigma_func, grid_end=self.wmax)
    # new_wmax = np.sqrt(s) * 4.0
    # sigma_r[i] = (new_wmax/self.wmax) * sigma_func(omega)


def sum_on_args(f, x, *args):
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


class SigmaGenerator():
    def __init__(self, wmax=20, **kwargs):
        self.wmax = wmax

    def generate(self):
        '''outputs one function'''
        raise NotImplementedError

    def factory(variant, **kwargs):
        if variant in ["G", "Gaussian", "gaussian"]:
            return GaussianMix(**kwargs)
        elif variant in ["B", "Beta", "beta"]:
            return BetaMix(**kwargs)
        elif variant in ["L", "Lorentzian", "lorentzian"]:
            return LorentzMix(**kwargs)
        else:
            raise ValueError(f"SigmaGenerator variant {variant} not recognized")
    factory = staticmethod(factory)
    
    
class GaussianMix(SigmaGenerator):
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

        if self.anormal:
            h *= w  # In some papers the gaussians are not normalized
        if self.norm:
            h *= np.pi*self.norm/(h.sum()+SMALL)

        return c, w, h

    def generate(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        sigma_func = lambda x: sum_on_args(gaussian, x, c, w, h)
        return sigma_func


class LorentzMix(GaussianMix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        sigma_func = lambda x: sum_on_args(lorentzian, x, c, w, h)
        return sigma_func


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
        return a, b

    def generate(self):
        self.new_random_num_per_group()
        c, w, h = self.random_cwh()
        a, b = self.random_ab()
        sigma_func = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)
        return sigma_func

    
class SigmaPiGenerator():
    def __init__(self, wmax=20, **kwargs):
        self.wmax = wmax

    def generate(self):
        '''outputs two functions'''
        raise NotImplementedError

    def factory(variant, **kwargs):
        if variant in ["LC", "Lorentz_comb", "lorentz_comb"]:
            return LorentzComb(**kwargs)
        else:
            sigma_generator = SigmaGenerator.factory(variant, **kwargs)
            return IntegralGenerator(sigma_generator, **kwargs)
    factory = staticmethod(factory)


class IntegralGenerator(SigmaPiGenerator):
    def __init__(self, sigma_generator, **kwargs):
        super().__init__(**kwargs)
        self.sigma_generator = sigma_generator

    def generate(self):
        half_sigma_func = self.sigma_generator.generate()
        sigma_func = lambda x: 0.5*(half_sigma_func(x)+half_sigma_func(-x))
        pi_func = lambda x: pi_integral(x, sigma_func, grid_end=self.wmax)
        return sigma_func, pi_func


class LorentzComb(SigmaPiGenerator):
    def __init__(self, norm=1, num_peaks=1000, width=0.05, **kwargs):
        super().__init__(**kwargs)
        self.norm = norm
        self.num_peaks = num_peaks
        self.width = width

    def generate(self):
        k = np.linspace(0, 1, self.num_peaks)
        # c = monofunc.piecewise_gap(k, n=8, soft=0.05, xlims=[0,1], ylims=[0,0.8*self.wmax])
        c = monofunc.random_climb(k, xlims=[0, 1], ylims=[0, 0.8*self.wmax])
        w = np.ones(self.num_peaks)*self.width
        h = abs(c) + 0.05
        h *= self.norm/(2*h*c/(c**2+w**2)).sum()
        sigma_func = lambda x: sum_on_args(even_lorentzian, x, c, w, h)
        pi_func = lambda x: sum_on_args(analytic_pi, x, c, w, h)
        return sigma_func, pi_func


if __name__ == "__main__":
    main()