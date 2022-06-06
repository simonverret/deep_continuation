from urllib import request
import pytest
import numpy as np
from pathlib import Path
HERE = Path(__file__).parent
EXPECT = HERE/"expected"

from deep_continuation import dataset


# fixtures are pytest way to setup variables. The name of the fixture function 
# is the name of the variable. Parametrized fixtures (using params, request, and 
# request.param as below) automatically generates all combinations of variables
# called by a test, so that it tests all possible input state.

@pytest.fixture(params=[55555, 555, 1])
def seed(request):
    return request.param
    

@pytest.fixture(params=[False, 8.86])
def rescale(request):
    return request.param


@pytest.fixture()
def expected_pi(seed, rescale, spurious=False):
    rescale_str = f'_rescaled{rescale}' if rescale else ''
    rescale_str = rescale_str if rescale and not spurious else ''
    path = EXPECT/f'Pi_B1_4_seed{seed}_128_beta30{rescale_str}.txt'
    Pi = np.loadtxt(open(path, "rb"), delimiter=',')
    return Pi


@pytest.fixture()
def expected_sigma(seed, rescale):
    rescale_str = f'_rescaled{rescale}' if rescale else ''
    path = EXPECT/f'sigma_B1_4_seed{seed}_512_wmax20{rescale_str}.txt'
    sigma = np.loadtxt(open(path, "rb"), delimiter=',')
    return sigma


def test_plot(mocker, seed, rescale, expected_pi, expected_sigma):
    mocker.patch('deep_continuation.dataset.plot_basic')
    dataset.main(
        plot=4,
        seed=seed,
        rescale=rescale,
    )    

    Pi = dataset.plot_basic.call_args[0][0]  # call_args[0] is args, [1] is kwargs    
    sigma =  dataset.plot_basic.call_args[0][1]
    assert np.allclose(Pi, expected_pi)
    assert np.allclose(sigma, expected_sigma)
