import time

import numpy as np
from scipy import integrate
from scipy.special import gamma
import matplotlib.pyplot as plt

from deep_continuation import utils
from deep_continuation import monotonous_functions as monofunc


SMALL = 1e-10
BIG = 1e10

default_parameters = {
    'seed': int(time.time()),
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
    # rescale
    'rescale': 4.0,
}


def simple_plot(pi, wn, sigma, w):
    fig, ax = plt.subplots(1, 3, figsize=[10, 5])
    
    ax[0].set_ylabel(r"$\Pi(i\omega_n)$")
    ax[0].set_xlabel(r"$\omega_n$")
    ax[0].plot(wn.T, pi.T, '.')

    ax[1].set_xlabel(r"$n$")
    ax[1].plot( pi.T, '.')
    
    ax[2].set_ylabel(r"$\sigma(\omega)$")
    ax[2].set_xlabel(r"$\omega$")
    ax[2].plot(w.T, sigma.T, ms=2, lw=1)
    
    # fig.tight_layout()
    plt.show()


def main():
    args = utils.parse_file_and_command(default_parameters, {})
    generator = SigmaPiGenerator.factory(**vars(args))

    np.random.seed(args.seed)
    sigma_func, pi_func = generator.generate()
    
    wmax_list = [args.wmax]
    M = 512
    beta_list = [200,400,600]
    N = 128

    wn = np.array([np.arange(0, N)*2*np.pi/beta for beta in beta_list])
    Pi = np.array([pi_func(np.arange(0, N)*2*np.pi/beta) for beta in beta_list])
    
    w = np.array([np.linspace(-wmax, wmax, 2*M+1) for wmax in wmax_list])
    sigma = np.array([(wmax/20)*sigma_func(np.linspace(-wmax, wmax, 2*M+1)) for wmax in wmax_list])
    
    simple_plot(Pi, wn, sigma, w)


def sum_on_args(f, x, *args):
    """Broadcasts a 1D function to all arguments and return the sum.

    computes: `f(x, a0[0], a1[0], ...) + f(x, a0[1], a1[1], ...) + ...`

    Args:
        f (function): Function to broadcast.
        x (array): Array on which to evaluate
        *args (arrays): Regular arguments of the function as arrays

    Returns:
        array: Sum of functions at each `x`
    """    
    if isinstance(x, np.ndarray):
        x = x[np.newaxis, :]
        args = [a for a in args]  # copy args to allow reassign
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                while len(args[i].shape) < len(x.shape):
                    args[i] = args[i][:, np.newaxis]
    return f(x, *args).sum(axis=0)


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
    tail_sampling = np.logspace(
        np.log10(grid_end), tail_power, tail)[1:]
    full_sampling = np.concatenate([
        -np.flip(tail_sampling),
        grid_sampling,
        tail_sampling
    ])
    return integrate.simps(integrand(full_sampling), full_sampling, axis=-1)


def pi_integral(wn, spectral_function, **kwargs):
    """Broadcastable integral for the Current-current response function.

    Integrate the spectral function :math:`\sigma(\omega)`
    .. math::
        \\Pi(i\\omega_n) = \\int_{-\infty}^{\\infty}
        \\frac{\\omega^2}{\\omega^2+\\omega_n^2}\sigma()_{i}
    using :func:`~integrate_with_tails`

    Args:
        wn (array): Matsubara frequencies at which to compute the response
        spectral_function (function): Callable spectral function

    Keyword Args:
        see :func:`~deep_continuation.function_generator.integrate_with_tails` 
    
    Returns:
        array: Result from an integration on `axis=-1`
    """
    if isinstance(wn, np.ndarray):
        wn = wn[:, np.newaxis]

    integrand = lambda x: (1/np.pi) * x**2 / (x**2+wn**2) * spectral_function(x)
    return integrate_with_tails(integrand, **kwargs)


def normalization(f, **kwargs):
    """Integrate function using :func:`~integrate_with_tails`

    Args:
        f (function): Function to be integrated.

    Returns:
        float: Normalization value
    """    
    def integrand(x): return f(x)
    return integrate_with_tails(integrand, **kwargs)


def first_moment(f, **kwargs):
    """Computes the first central moment (average) using :func:`~integrate_with_tails`

    Args:
        f (function): Input function for which the moment is computed
    
    Returns:
        float: First central moment (average)
    """
    def integrand(x): return x*f(x)
    return integrate_with_tails(integrand, **kwargs)


def second_moment(f, **kwargs):
    """Computes the sencond central moment (variance) using :func:`~integrate_with_tails`

    Args:
        f (function): Input function for which the moment is computed

    Returns:
        float: Second central moment (variance)
    """
    def integrand(x): return ((x - first_moment(f))**2)*f(x)
    return integrate_with_tails(integrand, **kwargs)


def gaussian(x, c, w, h):
    """Gaussian distributions.

    Args:
        x (array): Values at which the gaussian is evaluated
        c (float): Center of the distribution (average)
        w (float): Width of the distribution (variance)
        h (float): Height/weight of the distribtuion (area under the curve)

    Returns:
        array: Values of the gaussian at values in `x`
    """
    return (h/(np.sqrt(2*np.pi)*w))*np.exp(-((x-c)/w)**2/2)


def lorentzian(x, c, w, h):
    """Lorentz distributions.

    Args:
        x (array): Values at which the lorentzian is evaluated
        c (float): Center of the distribution
        w (float): Width of the distribution (at half height)
        h (float): Height/weight of the distribtuion (area under the curve)

    Returns:
        array: Values of the lorentzian at values in `x`
    """
    return (h/np.pi)*w/((x-c)**2+w**2)


def even_lorentzian(x, c, w, h):
    """Even pair of identical Lorentz distributions.
    
    Args:
        x (array): Values at which the lorentzian is evaluated
        c (float): Center of the distribution (+ or -)
        w (float): Width of the distribution (variance)
        h (float): Height/weight of the distribtuion (area under the curve)

    Returns:
        array: Values of the lorentzian pair at values in `x`
    """
    return (1/np.pi)*4*c*w*h/(((x-c)**2+w**2)*((x+c)**2+w**2))


def analytic_pi(x, c, w, h):
    """Analytic response function for an even pair of Lorentz distributions.

    Correspond to
    .. math::
        \\Pi(x) = \\int_{-\infty}^{\\infty}
        \\frac{\\omega^2}{\\omega^2+x^2}\sigma()_{i}
    where :math:`\\sigma(\\omega)` is :func:`~even_lorentzian`.

    Args:
        x (array): matsubara at which the response function is evaluated
        c (float): Center of the distribution (+ or -)
        w (float): Width of the distribution (variance)
        h (float): Height/weight of the distribtuion (area under the curve)

    Returns:
        array: Values of the integral at imaginary `x`
    """
    return 2*h*c/(c**2+(x+w)**2)


def beta_dist(x, a, b):
    """Beta distribution.

    Args:
        x (array): Values at which to evaluate the distribution
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """    
    return (gamma(a+b)/(SMALL+gamma(a)*gamma(b)))\
        * np.nan_to_num((x**(a-1))*((1-x)**(b-1))\
        * (x > 0) * (x < 1), copy=False)


def centered_beta(x, a, b):
    """Beta distribution centered at x=0.

    Args:
        x (array): Values at which to evaluate the distribution
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """
    c = a/(a+b)
    return beta_dist(x+c, a, b)


def standardized_beta(x, a, b):
    """Beta distribution centered at x=0 with variance 1.

    Args:
        x (array): Values at which to evaluate the distribution
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """
    w = np.sqrt(a*b/((a+b+1)*(a+b)**2))
    return centered_beta(x*w, a, b)*w


def free_beta(x, c, w, h, a, b):
    """Beta distribution with user-defined center, width and height.

    Args:
        x (array): Values at which to evaluate the distribution
        c (float): Center of the distribution (average)
        w (float): Width of the distribution (variance)
        h (float): Height/weight of the distribtuion (area under
            the curve)
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """
    return h*standardized_beta((x-c)/w, a, b)/w


class SigmaGenerator():
    """Base class for conductivity functions generators, with static factory method."""
    
    def generate(self):
        """Each call outputs a new random function as specified in subclasses."""
        raise NotImplementedError("To be overridden in subclasses")

    def factory(variant, **kwargs):
        """Static factory method: Creates the subclass specified by `variant`.
        
        The available generators include:
            - Gaussian mixture generator (G)
            - Beta mixture generator (B)
            - Lorentzian mixture generator (L)

        Args:
            variant (string): Specifies which subclass to instanciate
            nmbrs (list of tuples, optional): List of groups of peaks
                where tuples indicate a range for the number of peaks
                in each group. Defaults to [[0,4],[0,6]] (two groups
                of peaks, one up to 4 peaks the other up to 6).
            cntrs (list of tuples, optional): List of groups of peaks
                where tuples indicate a range for the centers of the
                peaks in each group. Defaults to [[0.00, 0.00], 
                [4.00, 16.0]] (two groups of peaks, the firsts
                centered at 0 the other centered between 4 and 16).
            wdths (list of tuples, optional): List of groups of peaks
                where tuples indicate a range for the widths of the
                peaks in each group. Defaults to [[0.04, 0.40],
                [0.04, 0.40]].
            wgths (list of tuples, optional): List of groups of peaks
                where tuples indicate a range for the heights/widths
                of the peaks in each group. Defaults to 
                [[0.00, 1.00], [0.00, 1.00]].
            arngs (list of tuples, optional): List of groups of peaks
                where tuples indicate a range for the `a` parameters
                for Beta peaks. Defaults to 
                [[2.00, 5.00], [0.50, 5.00]].
            brths (list of tuples, optional): List of groups of peaks
                where tuples indicate a range for the `b` parameters
                for Beta peaks. Defaults to 
                [[2.00, 5.00], [0.50, 5.00]].
            norm (int, optional): Total weight. Defaults to 1.
            anormal (bool, optional): All peaks are equally weighted.
                Defaults to False.

        Raises:
            ValueError: if `variant` is not recognized

        Returns:
            SigmaGenerator: One subclass of SigmaGenerator
        """        
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
    """Gaussian mixture generator, doc at :func:`~SigmaGenerator.factory`"""
    
    def __init__(self, 
        nmbrs=[[0,4],[0,6]],
        cntrs=[[0.00, 0.00], [4.00, 16.0]],
        wdths=[[0.04, 0.40], [0.04, 0.40]],
        wgths=[[0.00, 1.00], [0.00, 1.00]],
        norm=1, anormal=False,
        **kwargs
    ):
        self.nmbrs = nmbrs
        self.cntrs = cntrs
        self.wdths = wdths
        self.wgths = wgths
        self.norm = norm
        self.anormal = anormal

    def _random_num_per_group(self):
        num_per_group = [np.random.randint(n[0], n[1]+1) for n in self.nmbrs]
        if all(num_per_group) == 0:
            lucky_group = np.random.randint(0,len(num_per_group)-1)
            num_per_group[lucky_group] = 1
        return num_per_group

    def _random_cwh(self, num_per_groups):
        cl, wl, hl = [], [], []
        for i, n in enumerate(num_per_groups):
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
        c, w, h = self._random_cwh(self._random_num_per_group())
        sigma = lambda x: sum_on_args(gaussian, x, c, w, h)
        return sigma


class LorentzMix(GaussianMix):
    """Lorentzian mixture generator, doc at :func:`~SigmaGenerator.factory`"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self):
        c, w, h = self._random_cwh(self._random_num_per_group())
        sigma = lambda x: sum_on_args(lorentzian, x, c, w, h)
        return sigma


class BetaMix(GaussianMix):
    """Beta mixture generator, doc at :func:`~SigmaGenerator.factory`"""

    def __init__(self, 
        arngs=[[2.00, 5.00], [0.50, 5.00]],
        brths=[[2.00, 5.00], [0.50, 5.00]],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.arngs = arngs
        self.brths = brths

    def _random_ab(self, num_per_groups):
        al, bl = [], []
        for i, n in enumerate(num_per_groups):
            al.append(np.random.uniform(self.arngs[i][0], self.arngs[i][1], n))
            bl.append(np.random.uniform(self.brths[i][0], self.brths[i][1], n))
        a = np.hstack(al)
        b = np.hstack(bl)
        return a, b

    def generate(self):
        num_per_groups = self._random_num_per_group()
        c, w, h = self._random_cwh(num_per_groups)
        a, b = self._random_ab(num_per_groups)
        sigma = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)
        return sigma


class SigmaPiGenerator():
    def generate(self):
        """outputs two functions"""
        raise NotImplementedError

    def factory(variant, rescale=False, **kwargs):
        if variant in ["LC", "Lorentz_comb", "lorentz_comb"]:
            return LorentzComb(**kwargs)

        sigma_generator = SigmaGenerator.factory(variant, **kwargs)
        if rescale:
            return Fix2ndMomentGenerator(sigma_generator, **kwargs)

        return IntegralGenerator(sigma_generator, **kwargs)
    factory = staticmethod(factory)


class IntegralGenerator(SigmaPiGenerator):
    def __init__(self, sigma_generator, wmax=20, **kwargs):
        self.sigma_generator = sigma_generator
        self.wmax = wmax

    def generate(self):
        sigma_base = self.sigma_generator.generate()
        sigma_even = lambda x: 0.5*(sigma_base(x)+sigma_base(-x))
        pi = lambda x: pi_integral(x, sigma_even, grid_end=self.wmax)
        return sigma_even, pi


class Fix2ndMomentGenerator(IntegralGenerator):
    def __init__(self, sigma_generator, factor=4.0, **kwargs):
        super().__init__(sigma_generator, **kwargs)
        self.factor = factor

    def generate(self):
        sigma_base = self.sigma_generator.generate()
        sigma_even = lambda x: 0.5*(sigma_base(x)+sigma_base(-x))
        
        # rescaling
        sec_moment = (BIG**2)*pi_integral(BIG, sigma_even, grid_end=self.wmax)
        new_wmax = np.sqrt(sec_moment) * self.factor
        s = (new_wmax/self.wmax)
        sigma_rescaled = lambda x: s*sigma_even(s*x) 
        
        pi = lambda x: pi_integral(x, sigma_rescaled, grid_end=new_wmax)
        return sigma_rescaled, pi


class LorentzComb(SigmaPiGenerator):
    def __init__(self, norm=1, num_peaks=1000, width=0.05, wmax=20, **kwargs):
        self.norm = norm
        self.num_peaks = num_peaks
        self.width = width
        self.wmax = wmax

    def generate(self):
        k = np.linspace(0, 1, self.num_peaks)
        # c = monofunc.piecewise_gap(k, n=8, soft=0.05, xlims=[0,1], ylims=[0,0.8*self.wmax])
        c = monofunc.random_climb(k, xlims=[0, 1], ylims=[0, 0.8*self.wmax])
        w = np.ones(self.num_peaks)*self.width
        h = abs(c) + 0.05
        h *= self.norm/(2*h*c/(c**2+w**2)).sum()
        sigma = lambda x: sum_on_args(even_lorentzian, x, c, w, h)
        pi = lambda x: sum_on_args(analytic_pi, x, c, w, h)
        return sigma, pi


if __name__ == "__main__":
    main()
