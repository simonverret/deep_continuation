import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATAPATH = os.path.join(HERE, "data")

from tqdm import tqdm
from fire import Fire
import numpy as np
np.set_printoptions(precision=4)

from deep_continuation.distribution import get_generator_from_file
from deep_continuation.conductivity import sample_on_grid, get_rescaled_sigma, compute_matsubara_response, second_moment
from deep_continuation.plotting import plot_basic


def get_file_paths(
    path=os.path.join(DATAPATH, "default"),
    size=1,
    seed=55555,
    num_std=1,
    num_beta=1,
    Nwn=128,
    beta=30,
    Nw=512,
    wmax=20,
    fixstd=False,
): 
    if fixstd:
        id = f"{size}x{num_beta}x{num_std}_seed{seed}"
        std_str = f"_std{list_to_str(fixstd)}"
    else:
        id = f"{size}x{num_beta}x{num_std}_seed{seed}"
        std_str = ""
    beta_str = f"_beta{list_to_str(beta)}"

    beta_path = os.path.join(path, f"beta_{id}{beta_str}.npy")
    pi_path = os.path.join(path, f"Pi_{id}_Nwn{Nwn}{beta_str}{std_str}.npy")
    sigma_path = os.path.join(path, f"sigma_{id}_Nw{Nw}_wmax{wmax}{std_str}.npy")
    std_path = os.path.join(path, f"std_{id}.npy")
    fixstd_path = os.path.join(path, f"fixstd_{id}{std_str}.npy")

    return beta_path, pi_path, sigma_path, std_path, fixstd_path


def main(
    path=os.path.join(DATAPATH, "default"),
    size=1,
    seed=55555,
    num_std=1,
    num_beta=1,
    Nwn=128,
    beta=30,
    Nw=512,
    wmax=20,
    fixstd=False,
    plot=False,
    save_plot=None,
):
    # getting filenames
    beta_path, pi_path, sigma_path, std_path, fixstd_path = get_file_paths(
        path, size, seed, num_std, num_beta, Nwn, beta, Nw, wmax, fixstd,
    )

    # initialize empty containers 
    true_size = size * num_std * num_beta
    std_arr = np.empty(true_size)
    fixstd_arr = np.empty(true_size)
    sigma_arr = np.empty((true_size, Nw))
    pi_arr = np.empty((true_size, Nwn))
    beta_arr = np.empty(true_size)

    # setting flags to skip unecessary computation
    load_std = fixstd and os.path.exists(std_path)
    skip_std = not fixstd or load_std
    skip_sigma = os.path.exists(sigma_path)
    skip_pi = os.path.exists(pi_path)

    # load existing data
    if load_std:
        print(f"WARNING: Skipping existing {std_path}")
        std_arr = np.load(std_path)
    if skip_sigma:
        print(f"WARNING: Skipping existing {sigma_path}")
        sigma_arr = np.load(sigma_path)
    if skip_pi:
        print(f"WARNING: Skipping existing {pi_path}")
        pi_arr = np.load(pi_path)
        beta_arr = np.load(beta_path)
    
    # making single std compatible with our multi-std implementation
    if type(fixstd) is int or type(fixstd) is float:
        if num_std > 1:
            print(f"WARNING: ignoring num_std={num_std} because fixstd={fixstd}")
            num_std = 1

    # making single beta compatible with our multi-beta implementation
    if type(beta) is int or type(beta) is float:
        if num_beta > 1:
            print(f"WARNING: ignoring num_beta={num_beta} because beta={beta}")
            num_beta = 1

    # Generate missing data
    distrib_file_path = os.path.join(path, "param.json")
    distrib_generator = get_generator_from_file(distrib_file_path, seed)
    
    if not (skip_pi and skip_sigma and skip_std):
        progress_bar = tqdm(total=true_size)
        i = 0
        while i < true_size:
            
            # generate p
            distrib = distrib_generator.generate()
            sigma_func = lambda w: 0.5 * (distrib(w) + distrib(-w))   

            if type(fixstd) is list or type(fixstd) is tuple:
                fixstd_list = np.random.uniform(fixstd[0], fixstd[1], num_std)
            else:
                fixstd_list = [fixstd]
            for fixstd_value in fixstd_list:

                # compute std
                if not skip_std:
                    this_std = np.sqrt(second_moment(sigma_func, tail_start=wmax))        
                    std_arr[i] = this_std
                if fixstd:
                    fixstd_arr[i] = fixstd_value
                    rescaled_sigma_func = get_rescaled_sigma(sigma_func, std_arr[i], new_std=fixstd_value)    
                
                if type(beta) is list or type(beta) is tuple:
                    beta_list = np.random.uniform(beta[0], beta[1], num_beta)
                else:
                    beta_list = [beta]    
                for beta_value in beta_list:
                    fixstd_arr[i] = fixstd_value

                    # compute sigma
                    if not skip_sigma:
                        if fixstd:
                            sigma_arr[i] = sample_on_grid(rescaled_sigma_func, Nw, wmax)
                        else:
                            sigma_arr[i] = sample_on_grid(sigma_func, Nw, wmax)
                    if not skip_pi:
                        beta_arr[i] = beta_value
                        if fixstd:
                            pi_arr[i] = compute_matsubara_response(rescaled_sigma_func, Nwn, beta_value, tail_start=wmax)
                        else:
                            pi_arr[i] = compute_matsubara_response(sigma_func, Nwn, beta_value, tail_start=wmax)
                    i += 1
                    progress_bar.update(1)
        progress_bar.close()

    # saving the data
    if plot or save_plot:
        # usecase with flag --save_plot with no name name provided
        if save_plot == True: 
            save_plot = 'saved_plot.pdf' 
        plot_basic(pi_arr, sigma_arr, save_plot)
    else:
        if not skip_sigma:
            np.save(sigma_path, sigma_arr)    

        if not skip_std:    
            np.save(std_path, std_arr)
            np.save(fixstd_path, fixstd_arr)

        if not skip_pi:
            np.save(pi_path, pi_arr)
            np.save(beta_path, beta_arr)


def list_to_str(x):
    try:
        return 'to'.join(map(str,x))
    except TypeError:
        return x



if __name__ == "__main__":
    Fire(main)  # turns the function in a command line interface (CLI)
