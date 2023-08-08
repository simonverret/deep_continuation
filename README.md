# Deep Continuation
Analytic continuation of response functions with temperature-agnostic neural networks.

## Install
Install with

    cd deep_continuation
    pip install -e .

## Tests
Install tests requirements

    pip install pytest pytest-mock

Run the tests

    pytest

## Usage
You can train an example neural network in the `examples/tutorial.ipynb`. Install and open the Jupyter notebook editor

    pip install notebook
    jupyter notebook

The notebook lets you generate the datasets, by you can also do it from prompt.
Generate the validation set (10 000 spectra) with an explicit seed.

    python deep_continuation/dataset.py --size 10000 --seed 1

Progress is shown and data is saved under `deep_continuation/data/unbiased/`. 
Generate the training dataset (100 000 spectra) with the default seed (0):

    python deep_continuation/dataset.py --size 100000

Here are further usage examples for `dataset.py`

- Generate 4 temporary conductivities (using Beta functions) and plot them:

        python deep_continuation/dataset.py --size 4 --plot

- Change the seed to get different conductivities:

        python deep_continuation/dataset.py --size 4 --plot --seed 1

- Generate them with a fixed second moment:

        python deep_continuation/dataset.py --size 4 --plot --seed 1 --fixstd 8.86

- Generate them with a fixed second moment and random temperature:

        python deep_continuation/dataset.py --size 4 --plot --seed 1 --fixstd 8.86 --beta [0,60]
