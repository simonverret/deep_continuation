import os
import json
from pathlib import Path
HERE = Path(__file__).parent

from tqdm import tqdm
from fire import Fire
import numpy as np
np.set_printoptions(precision=4)

from deep_continuation.distributions import get_distribution_generator
from deep_continuation.conductivity import get_sigma_and_pi
from deep_continuation.plotting import *


def main(
    # interactive parameters
    plot=0,
    save=0,
    basic=False,
    scale=False,
    infer=False,
    # conductivity parameters
    Nwn=128, 
    Nw=512, 
    beta=30, 
    wmax=20, 
    rescale=False,
    spurious=False,
    # distribution parameters
    file="default",
    seed=55555,
    path=f"{HERE}/data/",
    name="B1",
    overwrite=False,
):
    file_path = f"{HERE}/data/{file}.json"
    if os.path.exists(file_path):
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


    distrib_generator = get_distribution_generator(
        variant, nmbrs, cntrs, wdths, wghts, arngs, brngs, anormal,
    )
    
    size = max(save, plot)
    sigma = np.empty((size, Nw))
    Pi = np.empty((size, Nwn))
    s = np.empty(size)

    np.random.seed(seed)
    for i in (tqdm(range(size)) if save else range(size)):
        distrib = distrib_generator.generate()
        sigma[i], Pi[i], s[i] = get_sigma_and_pi(
            distrib, Nwn, Nw, beta, wmax, rescale, spurious
        )

    if save > 0:
        pi_path, sigma_path, scale_path = get_file_paths(
            path, name, size, seed, Nwn, beta, Nw, wmax, rescale, spurious,
        )
        pi_exists = os.path.exists(pi_path)
        sigma_exists = os.path.exists(sigma_path)
        scale_exists = os.path.exists(scale_path)
        
        if not overwrite and pi_exists:
            print(f"WARNING: Skipping existing {pi_path}")
        else:
            np.savetxt(pi_path, Pi, delimiter=",")
        
        if not overwrite and sigma_exists:
            print(f"WARNING: Skipping existing {sigma_path}")
        else:
            np.savetxt(sigma_path, sigma, delimiter=",")
        
        if not overwrite and scale_exists:
            print(f"WARNING: Skipping existing {scale_path}")
        else:
            np.savetxt(scale_path, s, delimiter=",")

    elif plot > 0:
        print(f"normalizations\n  Pi    : {Pi[:,0]}\n  sigma : {sigma.sum(-1)}")
        print(f"scales\n  s     = {s}\n  betas = {s*beta}\n  wmaxs = {s*wmax}")

        plot_name = f"{HERE}/plots/{name}"
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
                betas=beta * np.ones_like(s),
                wmaxs=wmax * s,
                filename=f"{plot_name}_scale.pdf" if plot_name else None,
                default_wmax=wmax,
            )
        if infer:
            plot_infer_scale(Pi, sigma,
                f"{plot_name}_infer.pdf" if plot_name else None,
                default_wmax=wmax
            )

    else:    
        print("nothing to do, use --plot 10 or --save 10000")


def get_file_paths(
    path, name, size, seed, Nwn, beta, Nw, wmax, rescale, spurious,
):
    set_str = f"{name}_{size}_seed{seed}"
    spurious_str = '_spurious' if spurious and rescale else ''
    rescale_str = f'_rescaled{rescale}' if rescale else ''
    
    pi_path = path + f"Pi_{set_str}_{Nwn}_beta{beta}{spurious_str}.txt"
    sigma_path = path + f"sigma_{set_str}_{Nw}_wmax{wmax}{rescale_str}.txt"
    scale_path = path + f"scale_{set_str}_{Nwn}_beta{beta}{rescale_str}.txt"
    
    return pi_path, sigma_path, scale_path


if __name__ == "__main__":
    Fire(main)  # turns th function in a command line interface
