# DEEP CONTINUATION
Simon Verret,
Reza Nourafkan,
& Andre-Marie Tremablay

Research project on analytical continuation of the conductivity with neural networks.

## USAGE
To install (with dependencies)

    cd deep_continuation/
    pip install -e .

To generate a datasets of 100 samples (100 is for example purposes, ~50000 is better to actually train the neural network):

    cd deep_continuation/
    python data.py data/G1_train.json --generate 10000
    python data.py data/G1_valid.json --generate 1000

There are two neural nets code. The first is `simpler_mlp.py` it only requires the G1 data to work. The other is in the `train.py` file but it requires G1 to G4 to work.

To train the neural network

    python simpler_mlp.py
    
The goal is to compare datasets.

To remove errors at the beginning, you can comment all the `wandb` lines

## SLURM CLUSTERS
Two example submission scripts are provided in `bin/` to run the program on compute canada clusters
