import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATAPATH = os.path.join(HERE, "data")
PLOTPATH = os.path.join(HERE, "plots")

from tqdm import tqdm
from fire import Fire
import numpy as np
np.set_printoptions(precision=4)

from deep_continuation.distribution import beta_dist, get_generator_from_file
from deep_continuation.conductivity import sample_on_grid, get_rescaled_sigma, compute_matsubara_response, second_moment
from deep_continuation.plotting import plot_basic, plot_infer_scale, plot_scaled


def main(
    # script parameters
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
    # temperature sampling
    ntemp=1,
    width_temp=0,
    dist_temp='uniform',
    # distribution parameters
    name="B1",
    file="default",
    seed=55555,
    path=DATAPATH,
    overwrite=False,
):
    # obtain the distribution generator
    distrib_file_path = os.path.join(DATAPATH, f"{file}.json")
    if not os.path.exists(distrib_file_path):
        print(f"WARNING: {file}.json not found. Reverting to default.json")
        distrib_file_path = os.path.join(DATAPATH, "default.json")
    distrib_generator = get_generator_from_file(distrib_file_path, seed)
    
    # intialize empty containers for results
    ndistrib = max(save, plot)
    size = ndistrib * ntemp
    sigma = np.empty((size, Nw))
    Pi = np.empty((size, Nwn))
    s = np.empty(size)
    
    # get filenames
    pi_path, sigma_path, scale_path = get_file_paths(
        path, name, size, seed, Nwn, beta, Nw, wmax, rescale, spurious,
        ntemp, width_temp, dist_temp,
    )

    # setting skipping flags if file exists
    skip_pi =  not overwrite and os.path.exists(pi_path)
    skip_sigma = not overwrite and os.path.exists(sigma_path)
    skip_all =  skip_pi and skip_sigma

    # loading existing data
    if skip_sigma:
        print(f"WARNING: Skipping existing {sigma_path}")
        sigma = np.load(sigma_path)
        s = np.load(scale_path)
    if skip_pi:
        print(f"WARNING: Skipping existing {pi_path}")
        Pi = np.load(pi_path)

    # generate data
    i = 0
    for _ in tqdm(range(ndistrib)):
        distrib = distrib_generator.generate()
        sigma_func = lambda w: 0.5 * (distrib(w) + distrib(-w))   

        if rescale and not skip_all:
            old_std = np.sqrt(second_moment(sigma_func, tail_start=wmax))        
            rescaled_sigma_func = get_rescaled_sigma(sigma_func, old_std, new_std=rescale)    
        
        if ntemp>1 or width_temp>0:
            random_betas = np.random.uniform(
                beta-width_temp/2, beta+width_temp/2, ntemp
            )
            print(f"sampled\n  betas = {random_betas}")
        else:
            random_betas = [beta]

        for b in random_betas:

            if not (rescale or skip_sigma):
                s[i]=1
                sigma[i] = sample_on_grid(sigma_func, Nw, wmax)

            if rescale and not skip_sigma:
                s[i] = old_std / rescale
                sigma[i] = sample_on_grid(rescaled_sigma_func, Nw, wmax)
            
            if not (rescale or skip_pi):
                Pi[i] = compute_matsubara_response(sigma_func, Nwn, b, tail_start=wmax)
        
            if rescale and not (skip_pi or spurious):
                Pi[i] = compute_matsubara_response(rescaled_sigma_func, Nwn, b, tail_start=wmax)
            
            i += 1
    
    # saving the data
    if save > 0:
        if not skip_sigma:
            np.save(sigma_path, sigma[:save*ntemp])        
            np.save(scale_path, s[:save*ntemp])

        if not skip_pi:
            np.save(pi_path, Pi)

    # plotting the data
    if plot > 0:
        sigma = sigma[:plot*ntemp]
        s = s[:plot*ntemp]
        Pi = Pi[:plot*ntemp]

        print(f"normalizations\n  Pi    : {Pi[:,0]}\n  sigma : {sigma.sum(-1)}")
        print(f"scales\n  s     = {s}\n  betas = {s*beta}\n  wmaxs = {s*wmax}")

        plot_name = os.path.join(PLOTPATH,name)

        none_specified = (not any([basic, scale, infer]))
        if basic or none_specified:
            plot_basic(Pi, sigma, 
                f"{plot_name}_basic.pdf" if plot_name else None
            )
        if scale:
            plot_scaled(Pi, sigma,
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

    if save==0 and plot==0:    
        print("nothing to do, use --plot 10 or --save 10000")


def get_file_paths(
    path, name, size, seed, Nwn, beta, Nw, wmax, rescale, spurious,
    ntemp, width_temp, dist_temp,
):
    set_str = f"{name}_{size}_seed{seed}"
    
    rescale_str = f'_rescaled{rescale}' if rescale else ''

    beta_str = f'_ntemp{ntemp}_{dist_temp}{width_temp}_beta{beta}' if ntemp>1 else f'_beta{beta}'

    sigma_path = os.path.join(path, f"sigma_{set_str}_{Nw}_wmax{wmax}{rescale_str}.npy")
    scale_path = os.path.join(path, f"scale_{set_str}_{Nwn}{beta_str}{rescale_str}.npy")
    
    pi_rescale_str = rescale_str if rescale and not spurious else ''
    pi_path = os.path.join(path, f"Pi_{set_str}_{Nwn}{beta_str}{pi_rescale_str}.npy")
    
    return pi_path, sigma_path, scale_path


if __name__ == "__main__":
    Fire(main)  # turns the function in a command line interface (CLI)
