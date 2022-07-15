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
def fixstd(request):
    return request.param


@pytest.fixture()
def expected_pi_and_path(seed, fixstd):
    _, pi_path, _, _, _ = dataset.get_file_paths(
        path=EXPCPATH,
        size=4,
        seed=seed,
        fixstd=fixstd,
    )
    Pi = np.load(pi_path)
    return Pi, pi_path


@pytest.fixture()
def expected_sigma_and_path(seed, fixstd):
    _, _, sigma_path, _, _ = dataset.get_file_paths(
        path=EXPCPATH,
        size=4,
        seed=seed,
        fixstd=fixstd,
    )
    sigma = np.load(sigma_path)
    return sigma, sigma_path


def test_save(mocker, seed, fixstd, expected_pi_and_path, expected_sigma_and_path):
    mocker.patch('deep_continuation.dataset.np.save')
    dataset.main(
        size=4,
        seed=seed,
        fixstd=fixstd,
        path=EXPCPATH
    )    
    expected_pi, expected_pi_path = expected_pi_and_path
    expected_sigma, expected_sigma_path = expected_sigma_and_path

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


def test_skip(mocker, seed, fixstd):
    mocker.patch('deep_continuation.dataset.np.save')
    dataset.main(
        size=4,
        seed=seed,
        fixstd=fixstd,
        path=EXPCPATH
    )    
    assert dataset.np.save.call_count == 0


def test_plot(mocker, seed, fixstd, expected_pi_and_path, expected_sigma_and_path):
    mocker.patch('deep_continuation.dataset.plot_basic')
    dataset.main(
        size=4,
        seed=seed,
        fixstd=fixstd,
        plot=True
    )    

    expected_pi, _ = expected_pi_and_path
    expected_sigma, _ = expected_sigma_and_path
    Pi = dataset.plot_basic.call_args[0][0]  # call_args[0] is args, [1] is kwargs    
    sigma =  dataset.plot_basic.call_args[0][1]
    assert np.allclose(Pi, expected_pi)
    assert np.allclose(sigma, expected_sigma)
