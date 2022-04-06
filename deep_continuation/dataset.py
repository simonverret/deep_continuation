import os
from pathlib import Path
import multiprocessing
HERE = Path(__file__).parent

from tqdm import tqdm
from fire import Fire
import numpy as np
np.set_printoptions(precision=4)

from deep_continuation.distributions import get_distribution_generator
from deep_continuation.conductivity import get_sigma_and_pi
from deep_continuation.plotting import *


def build_dataset(
    # generation parameters
    plot=0,
    save=0,
    seed=55555,
    name="B1",
    path=f"{HERE}/data/",
    plot_name=f"{HERE}/plots/last",
    basic=False,
    scale=False,
    infer=False,
    # dataset parameters
    Nwn=128, 
    Nw=512, 
    beta=30, 
    wmax=20, 
    rescale=False,
    spurious=False,
    # spectra parameters
    variant="Beta",
    nmbrs=[[0, 4], [0, 6]],
    cntrs=[[0.00, 0.00], [4.00, 16.0]],
    wdths=[[0.40, 4.00], [0.40, 4.00]],
    wghts=[[0.00, 1.00], [0.00, 1.00]],
    arngs=[[2.00, 10.00], [0.70, 10.00]],
    brngs=[[2.00, 10.00], [0.70, 10.00]],
    anormal=False,

):
    size = max(save, plot)
    sigma = np.empty((size, Nw))
    Pi = np.empty((size, Nwn))
    s = np.empty(size)

    distrib_generator = get_distribution_generator(
        variant, nmbrs, cntrs, wdths, wghts, arngs, brngs, anormal,
    )
    
    np.random.seed(seed)
    for i in (tqdm(range(size)) if save else range(size)):
        distrib = distrib_generator.generate()
        sigma[i], Pi[i], s[i] = get_sigma_and_pi(
            distrib, Nwn, Nw, beta, wmax, rescale, spurious,    
        )

    if save > 0:
        setstr = f"{name}_{size}_seed{seed}"
        pi_path = path + f"Pi_{setstr}_{Nwn}_beta{beta}.txt"
        sigma_path = path + f"sigma_{setstr}_{Nw}_wmax{wmax}_rescaled{rescale}.txt"
        scale_path = path + f"scale_{setstr}_{Nwn}_beta{beta}.txt"
        
        if os.path.exists(sigma_path) or os.path.exists(pi_path):
            raise ValueError("there is already a dataset on this path")
        np.savetxt(pi_path, Pi, delimiter=",")
        np.savetxt(sigma_path, sigma, delimiter=",")
        np.savetxt(scale_path, s, delimiter=",")

    elif plot > 0:
        print(f"scales\n  s     = {s}\n  betas = {beta/s}\n  wmaxs = {s*wmax}")
        print(f"normalizations\n  Pi    : {Pi[:,0]}\n  sigma : {sigma.sum(-1)}")

        if basic:
            plot_basic(Pi, sigma, 
                f"{plot_name}_basic.pdf" if plot_name else None
            )
        if scale:
            plot_scaled(Pi, sigma,
                # ### use beta/s
                # beta / scale,
                # wmax * np.ones_like(scale),
                ### or s*wmax
                beta * np.ones_like(scale),
                wmax * scale,
                f"{plot_name}_scale.pdf" if plot_name else None,
                default_wmax=wmax,
            )
        if infer:
            plot_infer_scale(Pi, sigma,
                f"{plot_name}_infer.pdf" if plot_name else None,
                default_wmax=wmax
            )

    else:    
        print("nothing to do, use --plot 10 or --save 10000")


if __name__ == "__main__":
    Fire(build_dataset)  # turns th function in a command line interface
