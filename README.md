# Deep Continuation
Analytic continuation of response functions with temperature-agnostic neural networks.

## Install
Install with

    cd deep_continuation
    pip install -e .

## Generate data
Generate the training data with:

    python deep_continuation/data.py

It can take several minutes, the data is saved under `deep_continuation 

### `function_generator.py`
This module provides generators that produce pairs of functions (actual python functions) that can be used at any frequencies and temperatures (Matsubara frequencies).

### `data_generator.py`
This module uses the above function generator to produce the discrete samples of these functions (vectors) at specific freqencies and temperatures. It also provides the tools to plot and save these vectors. Here are multiple examples:

- Generate 4 temporary conductivities (using Beta functions) and plot them:

        python data_generator.py data/B1_train.json --plot 4

- Change the seed to get different conductivities:

        python data_generator.py data/B1_train.json --plot 4 --seed 1

- Rescale them to see the temperature agnostic case:

        python data_generator.py data/B1_train.json --plot 4 --seed 1 --rescale 4

- Generate 1000 training conductivities and 200 validation conductivities and save them instead of plotting them:

        python data_generator.py data/B1_train.json --generate 1000 --rescale 4
        python data_generator.py data/B1_valid.json --generate 200 --rescale 4

- 1000 conductivities is not enough for proper training of neural networks. You have to manually remove existing dataset to create a new one with the same name. The following gives a proper dataset, but is very long to run:

    python data_generator.py data/B1_train.json --generate 100000 --rescale 4
    python data_generator.py data/B1_valid.json --generate 10000 --rescale 4

Here are the calls used to generate data for the experiments on temperature agnostic training. It is very long to generate, it produces 100000 conductivies and the matsubara responses functions at 8 different temperatures. You will also find these command lines in submission scripts of the `bin` folder.

    python data_generator.py data/B1_valid.json --generate 10000 --beta 2 10 15 20 25 30 35 50 --rescale 4
    python data_generator.py data/B1_train.json --generate 100000 --beta 2 10 15 20 25 30 35 50 --rescale 4


### Advanced usage (Custom Configuration)

There are two ways to modify the data generation process. The first way is to change the content of the `data/B1_valid.json` file. For instance, we can use another file: 

    python data_generator.py data/G1_train.json --plot 4

Another way is to use command line arguments like `--variant G` here:

    python data_generator.py data/B1_train.json --plot 4 --variant G

To get a list of all configurable parameters, use `--help`:

    python data_generator.py --help

Or look for `default_parameters` and `default_parameters_help` dictionaries in the code. Here is an example, from the `function_generator.py` file, at the core of the data generating process:

    {
        'seed': "Random seed used to generate the data",
        'variant': "Gaussian (G), Beta (B) or Lorentz (L)",
        'anormal': "(bool) When true, individual peaks are not normalized as in",
        'wmax': "Maximum frequencies of the discrete samples",
        'nmbrs': "(list of list) List of ranges for number of peaks (for each peak group)",
        'cntrs': "(list of list) List of ranges for positions of peaks",
        'wdths': "(list of list) List of ranges for widths of peaks",
        'wghts': "(list of list) List of ranges for weights (heights) of peaks",
        'arngs': "(list of list) List of ranges of the a parameters of Beta peaks",
        'brths': "(list of list) List of ranges of the b parameters of Beta peaks",
        'even': "(bool) Make a copy of each peaks at negative positions",
        'num_peaks': "Number of Lorentz peaks used in the Lorentz comb",
        'width': "Width of Lorentz peaks of the Lorentz comb",
        'rescale': "Value for fixing the variance of all spectra",
        'spurious': "(bool) Compute Matsubara responses BEFORE rescaling, introducing spurious correlation",
    }

The `data/B1_train.json` used up to now defines these parameters:

    {
        "seed": 55555,
        "generate": 100000,
        "path": "data/B1/train",
        "variant": "Beta",
        "wmax": 20.0,
        "beta": [10.0],
        "nmbrs": [[0, 4],[0, 6]],
        "cntrs": [[0.00, 0.00], [4.00, 16.0]],
        "wdths": [[0.20, 4.00], [0.40, 4.00]],
        "wghts": [[0.00, 1.00], [0.00, 1.00]],
        "arngs": [[2.00, 10.0], [0.50, 10.0]],
        "brngs": [[2.00, 10.0], [0.50, 10.0]]
    }

## Train a neural networks

Simple script that trains the best neural network
