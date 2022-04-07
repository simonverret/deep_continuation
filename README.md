# Deep Continuation
Analytic continuation of response functions with temperature-agnostic neural networks.

## Install
Install with

    cd deep_continuation
    pip install -e .

## Usage
Generate the training dataset (100 000 spectra) with:

    python deep_continuation/dataset.py --save 100000

Progress is displayed using [tqdm](https://github.com/tqdm/tqdm), and data is saved under `deep_continuation/data/`. Validation set (10 000 spectra) requires a different seed.

    python deep_continuation/dataset.py --save 10000 --seed 555

You can train an example neural network and see results in the `tutorial.ipynb` Jupyter notebook.

## Modules
#### `distributions.py`
Generators of randomly shaped distributions.

#### `conductivities.py`
Functions to compute the Matsubara frequency conductivities from the real frequency ones.

#### `plotting.py`
Plotting functions, along with temperature and frequency scale extraction.

#### `dataset.py`
Command line script (using [python-fire](https://github.com/google/python-fire) to produce training input and output vectors for the neural network, with utilities to plot and save them. Here are other usage examples:

- Generate 4 temporary conductivities (using Beta functions) and plot them:

        python deep_continuation/dataset.py --plot 4 --basic

- Change the seed to get different conductivities:

        python deep_continuation/dataset.py --plot 4 --basic --seed 1

- Rescale them to see the temperature agnostic case:

        python deep_continuation/dataset.py --plot 4 --basic --seed 1 --rescale 4

- Generate 1000 training conductivities and 200 validation conductivities and save them instead of plotting them:

        python deep_continuation/dataset.py --save 1000 --rescale 4
        python deep_continuation/dataset.py --save 200 --rescale 4
