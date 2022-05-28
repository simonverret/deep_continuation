import numpy as np
from pathlib import Path
HERE = Path(__file__).parent

from deep_continuation import dataset

# test p0 fixed T

# p fixed T

def test_main(mocker):#mock_ax_plot, mock_pause):
    mocker.patch('deep_continuation.dataset.plot_basic')
    dataset.main(
        plot=4,
        basic=True,
        seed=1,
        rescale=8.86,
    )    

    call_args = dataset.plot_basic.call_args[0]  # [0] is args, [1] is kwargs
    Pi =  call_args[0]
    expected_Pi_path = HERE/'Pi_B1_4_seed1_128_beta30.txt'
    expected_Pi = np.loadtxt(open(expected_Pi_path, "rb"), delimiter=',')
    assert np.allclose(Pi, expected_Pi)
    
    sigma =  call_args[1]
    expected_sigma_path = HERE/'sigma_B1_4_seed1_512_wmax20_rescaled8.86.txt'
    expected_sigma = np.loadtxt(open(expected_sigma_path, "rb"), delimiter=',')
    assert np.allclose(sigma, expected_sigma)
