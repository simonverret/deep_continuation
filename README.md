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
    python data.py data/G1_train.json --generate 100

To train the neural network

    python train.py
    
