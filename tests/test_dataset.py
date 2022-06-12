import pytest
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
EXPCPATH = os.path.join(HERE,"expected")

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
def expected_pi_and_name(seed, rescale, spurious=False):
    rescale_str = f'_rescaled{rescale}' if rescale else ''
    rescale_str = rescale_str if rescale and not spurious else ''
    name = f'Pi_B1_4_seed{seed}_128_beta30{rescale_str}.npy'
    path = os.path.join(EXPCPATH, name)
    Pi = np.load(path)
    return Pi, name


@pytest.fixture()
def expected_sigma_and_name(seed, rescale):
    rescale_str = f'_rescaled{rescale}' if rescale else ''
    name = f'sigma_B1_4_seed{seed}_512_wmax20{rescale_str}.npy'
    path = os.path.join(EXPCPATH, name)
    sigma = np.load(path)
    return sigma, name


def test_save(mocker, seed, rescale, expected_pi_and_name, expected_sigma_and_name):
    mocker.patch('deep_continuation.dataset.np.save')
    dataset.main(
        save=4,
        seed=seed,
        rescale=rescale,
        path=HERE
    )    
    expected_pi, pi_name = expected_pi_and_name
    expected_sigma, sigma_name = expected_sigma_and_name
    expected_pi_path = os.path.join(HERE, pi_name)
    expected_sigma_path = os.path.join(HERE, sigma_name)

    for call_args in dataset.np.save.call_args_list:
        path_received = call_args[0][0]
        tensor_received = call_args[0][1]

        if path_received == expected_pi_path:
            assert np.allclose(tensor_received, expected_pi)
        elif path_received == expected_sigma_path:
            assert np.allclose(tensor_received, expected_sigma)
        elif "scale" in path_received:
            pass
        else:
            assert False


def test_skip(mocker, seed, rescale, expected_pi_and_name, expected_sigma_and_name):
    mocker.patch('deep_continuation.dataset.np.save')
    dataset.main(
        save=4,
        seed=seed,
        rescale=rescale,
        path=EXPCPATH
    )    
    assert dataset.np.save.call_count == 0


def test_plot(mocker, seed, rescale, expected_pi_and_name, expected_sigma_and_name):
    mocker.patch('deep_continuation.dataset.plot_basic')
    dataset.main(
        plot=4,
        seed=seed,
        rescale=rescale,
    )    

    expected_pi, _ = expected_pi_and_name
    expected_sigma, _ = expected_sigma_and_name
    Pi = dataset.plot_basic.call_args[0][0]  # call_args[0] is args, [1] is kwargs    
    sigma =  dataset.plot_basic.call_args[0][1]
    assert np.allclose(Pi, expected_pi)
    assert np.allclose(sigma, expected_sigma)
