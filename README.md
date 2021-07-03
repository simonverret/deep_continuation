# Deep Continuation
Python module by S. Verret, R. Nourafkan, Q. Weyrich, & A.-M. S. Tremblay for Analytic continuation of response functions with temperature-agnostic neural networks.

## Install
Download with

    git clone https://github.com/simonverret/deep_continuation.git

Install with

    cd deep_continuation/
    pip install -e .


## Usage
All the code is in `deep_continuation/deep_continuation/`, so move there:

    cd deep_continuation

### Generate data

Generate 4 temporary conductivities (using Beta functions) and plot them:

    python data_generator.py data/B1_train.json --plot 4

Change the seed to get different conductivities:

    python data_generator.py data/B1_train.json --plot 4 --seed 1

Rescale them to see the temperature agnostic case:

    python data_generator.py data/B1_train.json --plot 4 --seed 1 --rescale 4

Generate 1000 training conductivities and 200 validation conductivities and save them instead of plotting them:

    python data_generator.py data/B1_train.json --generate 1000 --rescale 4
    python data_generator.py data/B1_valid.json --generate 200 --rescale 4

Note that 1000 conductivities is not enough for proper training of neural networks. You have to manually remove existing dataset to create a new one with the same name. The following gives a proper dataset, but is very long to run:

    python data_generator.py data/B1_train.json --generate 100000 --rescale 4
    python data_generator.py data/B1_valid.json --generate 10000 --rescale 4

As an example, here are the actual calls I used to generate data for experiments on the temperature agnostic training. I is very long to generate, it produces 100000 conductivies and the matsubara responses functions at 8 different temperatures. You will also find these command lines in submission scripts of the `bin` folder.

    python data_generator.py data/B1_valid.json --generate 10000 --beta 2 10 15 20 25 30 35 50 --rescale 4
    python data_generator.py data/B1_train.json --generate 100000 --beta 2 10 15 20 25 30 35 50 --rescale 4


**More Details**: Data generation is highly configurable. There are two ways to modify the generation parameters. The first way is to change the content of the `data/B1_valid.json` file, or use another file. For example:

    python data_generator.py data/G1_train.json --plot 4

Another way is to use command line arguments:

    python data_generator.py data/B1_train.json --variant G



### Train neural networks
There are two neural nets code. The first is `simpler_mlp.py` it only requires the G1 data to work. The other is in the `train.py` file but it requires G1 to G4 to work.

To train the neural network

    python simpler_mlp.py
    
The goal is to compare datasets.

To remove errors at the beginning, you can comment all the `wandb` lines

## SLURM CLUSTERS
Examples of submission scripts are provided in `bin/` to run the program on compute canada clusters


