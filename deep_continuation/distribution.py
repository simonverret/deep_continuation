import json
import warnings

import numpy as np
from scipy.special import gamma

SMALL = 1e-10


def get_generator_from_file(file_path, seed=None):
    with open(file_path) as f:
        file_parameters = json.load(f)
    
    variant = file_parameters.get('variant')
    anormal = file_parameters.get('anormal')
    nmbrs = file_parameters.get('nmbrs')
    cntrs = file_parameters.get('cntrs')
    wdths = file_parameters.get('wdths')
    wghts = file_parameters.get('wghts')
    arngs = file_parameters.get('arngs')
    brngs = file_parameters.get('brngs')
    if seed is None:
        seed = file_parameters.get('seed')

    generator = get_generator(
        variant, nmbrs, cntrs, wdths, wghts, arngs, brngs, anormal, seed,
    )
    return generator


def get_generator(
    variant="Beta",
    nmbrs=[[0, 4], [0, 6]],
    cntrs=[[0.00, 0.00], [4.00, 16.0]],
    wdths=[[0.40, 4.00], [0.40, 4.00]],
    wghts=[[0.00, 1.00], [0.00, 1.00]],
    arngs=[[2.00, 10.00], [0.70, 10.00]],
    brngs=[[2.00, 10.00], [0.70, 10.00]],
    anormal=False,
    seed=None,
):
    """Creates the distribution generator object according to arguments.

    The available generators include:
        - Gaussian mixture generator (G)
        - Beta mixture generator (B)
        - Lorentzian mixture generator (L)

    Args:
        variant (string): Specifies which subclass to instanciate (G, B or L)
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
        wghts (list of tuples, optional): List of groups of peaks
            where tuples indicate a range for the heights/widths
            of the peaks in each group. Defaults to
            [[0.00, 1.00], [0.00, 1.00]].
        arngs (list of tuples, optional): List of groups of peaks
            where tuples indicate a range for the `a` parameters
            for Beta peaks. Defaults to
            [[2.00, 5.00], [0.50, 5.00]].
        brngs (list of tuples, optional): List of groups of peaks
            where tuples indicate a range for the `b` parameters
            for Beta peaks. Defaults to
            [[2.00, 5.00], [0.50, 5.00]].
        norm (int, optional): Total weight. Defaults to 1.
        anormal (bool, optional): All peaks are equally weighted.
            Defaults to False.

    Raises:
        ValueError: if `variant` is not recognized

    Returns:
        distribution generator: One subclass of distribution generator
    """
    if variant in ["G", "Gaussian", "gaussian"]:
        return GaussianMix(
            nmbrs=nmbrs,
            cntrs=cntrs,
            wdths=wdths,
            wghts=wghts,
            anormal=anormal,
            seed=seed,
        )
    elif variant in ["L", "Lorentzian", "lorentzian"]:
        return LorentzMix(
            nmbrs=nmbrs,
            cntrs=cntrs,
            wdths=wdths,
            wghts=wghts,
            anormal=anormal,
            seed=seed,
        )
    elif variant in ["B", "Beta", "beta"]:
        return BetaMix(
            nmbrs=nmbrs,
            cntrs=cntrs,
            wdths=wdths,
            wghts=wghts,
            arngs=arngs,
            brngs=brngs,
            anormal=anormal,
            seed=seed,
        )
    else:
        raise ValueError(f"distribution generator variant {variant} not recognized")


class GaussianMix():
    """Gaussian mixture generator, doc at :func:`~random_distribution_generator`"""

    def __init__(
        self,
        nmbrs=[[0, 4], [0, 6]],
        cntrs=[[0.00, 0.00], [4.00, 16.0]],
        wdths=[[0.04, 0.40], [0.04, 0.40]],
        wghts=[[0.00, 1.00], [0.00, 1.00]],
        norm=1,
        anormal=False,
        seed=None,
    ):
        self.nmbrs = nmbrs
        self.cntrs = cntrs
        self.wdths = wdths
        self.wghts = wghts
        self.norm = norm
        self.anormal = anormal
        
        # legacy compatible random number generator
        self.random = np.random.RandomState(seed)  
        # # newer version
        # self.random = np.random.default_rng(seed)  # newer (must find and replace `randint` for `integers`)
        
    def _random_num_per_group(self):
        num_per_group = [self.random.randint(n[0], n[1] + 1) for n in self.nmbrs]
        if all(num_per_group) == 0:
            lucky_group = self.random.randint(0, len(num_per_group) - 1)
            num_per_group[lucky_group] = 1
        return num_per_group

    def _random_cwh(self, num_per_groups):
        cl, wl, hl = [], [], []
        for i, n in enumerate(num_per_groups):
            cl.append(self.random.uniform(self.cntrs[i][0], self.cntrs[i][1], n))
            wl.append(self.random.uniform(self.wdths[i][0], self.wdths[i][1], n))
            hl.append(self.random.uniform(self.wghts[i][0], self.wghts[i][1], n))
        c = np.hstack(cl)
        w = np.hstack(wl)
        h = np.hstack(hl)

        if self.anormal:
            h *= w  # In some papers the gaussians are not normalized
        if self.norm:
            h *= np.pi * self.norm / (h.sum() + SMALL)

        return c, w, h

    def generate(self):
        c, w, h = self._random_cwh(self._random_num_per_group())
        sigma = lambda x: sum_on_args(gaussian, x, c, w, h)
        return sigma


class LorentzMix(GaussianMix):
    """Lorentzian mixture generator, doc at :func:`~random_distribution_generator`"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self):
        c, w, h = self._random_cwh(self._random_num_per_group())
        sigma = lambda x: sum_on_args(lorentzian, x, c, w, h)
        return sigma


class BetaMix(GaussianMix):
    """Beta mixture generator, doc at :func:`~random_distribution_generator`"""

    def __init__(
        self,
        arngs=[[2.00, 5.00], [0.50, 5.00]],
        brngs=[[2.00, 5.00], [0.50, 5.00]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.arngs = arngs
        self.brngs = brngs

    def _random_ab(self, num_per_groups):
        al, bl = [], []
        for i, n in enumerate(num_per_groups):
            al.append(self.random.uniform(self.arngs[i][0], self.arngs[i][1], n))
            bl.append(self.random.uniform(self.brngs[i][0], self.brngs[i][1], n))
        a = np.hstack(al)
        b = np.hstack(bl)
        return a, b

    def generate(self):
        num_per_groups = self._random_num_per_group()
        c, w, h = self._random_cwh(num_per_groups)
        a, b = self._random_ab(num_per_groups)
        sigma = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)
        return sigma


def sum_on_args(f, x, *args):
    """Broadcast a 1D function to all arguments and return the sum.

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
    return (h / (np.sqrt(2 * np.pi) * w)) * np.exp(-(((x - c) / w) ** 2) / 2)


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
    return (h / np.pi) * w / ((x - c) ** 2 + w**2)


def free_beta(x, c, w, h, a, b):
    """Beta distribution with user-defined center, width and height.

    Args:
        x (array): Values at which to evaluate the distribution
        c (float): Center of the distribution (average)
        w (float): Width of the distribution (variance)
        h (float): Height/weight of the distribtuion (area under the curve)
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """
    return h * standardized_beta((x - c) / w, a, b) / w


def standardized_beta(x, a, b):
    """Beta distribution centered at x=0 with variance 1.

    Args:
        x (array): Values at which to evaluate the distribution
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """
    w = np.sqrt(a * b / ((a + b + 1) * (a + b) ** 2))
    return centered_beta(x * w, a, b) * w


def centered_beta(x, a, b):
    """Beta distribution centered at x=0.

    Args:
        x (array): Values at which to evaluate the distribution
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """
    c = a / (a + b)
    return beta_dist(x + c, a, b)


def beta_dist(x, a, b):
    """Beta distribution.

    Args:
        x (array): Values at which to evaluate the distribution
        a (float): First Beta function parameter
        b (float): Second Beta function parameter

    Returns:
        array: Values of the function at the values of `x`
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (gamma(a + b) / (SMALL + gamma(a) * gamma(b))) * np.nan_to_num(
            (x ** (a - 1)) * ((1 - x) ** (b - 1)) * (x > 0) * (x < 1), copy=False
        )
