# DEEP CONTINUATION
Simon Verret,
Reza Nourafkan,
& Andre-Marie Tremablay

Research project on analytical continuation of the conductivity with neural networks.

## USAGE
Train the default neural network with:

    cd mlp
    python deep_continuation.py

The default dataset location is `../sdata/`, relative to the `mlp` directory. The training and validation data are splitted automatically. Files containing the data must be added and named `../data/Pi.csv` (input) and `../data/SigmaRe.csv` (target output).

You can pass arguments to the python script. For example:

    python deep_continuation.py --path ./Database/

will fetch the dataset from `./Database/` instead of the default `../sdata/`. However, the filenames still have to be `Pi.csv` and `SigmaRe.csv`.

Many other arguments can be passed, for example: 

    $ deep_continutation.py --no_cuda --layers 128 256 256 512 -lr 0.001 --no_schedule

To see all possibilities:

    $ deep_continutation.py --help

or see the `default_parameters` dictionary at the beginning of the script. This dictionary serves as a template for the parser. When possible, the latter will:

1. replace the default value with the one found in `params.json`, then
2. replace this value with the one specified by command arguments.

Note that for every bool parameters `--flag`, an additional `--no_flag` allows to turn it off.

## OUTPUTS
Each time the training is launched, a very long but very explicit name is given to the current run, for example:

    NAME = mlp128-512-1024-512_bs1500_lr0.01_wd0_drop0_wup_scheduled0.5-8

which can be read as follow: fully connected neural network (MLP, _multi-layer perceptron_) with input size 128, two hidden layers of size 512 and 1024 and default ouput size of 512, (corresponding to option `--layers 128 512 1024 512`), batch size 1500 (`--batch_size 1500`), learning rate 0.01 (`--lr 0.01`), no weight decay (`--weight_decay 0`), no dropout (`--dropout 0`), a learning rate warmup (a linear increase of the learning rate during the first epoch, `--warmup`) and a scheduler (`--schedule`) which multiply the learning rate by 0.5 `--factor 0.5` every time the training loss does not reduces for 8 epoch (`--patience 8`). Note that some parameters are not included in the name, for simplicity, but these are all saved in the file `params_NAME.jason`.

While `deep_continuation.py` runs, it continually updates a save of the best models for three performance measures: 
1. the cost function used during training, for example `L1loss`.
2. the means square error `mse`.
3. the DC error `dc_error`, i.e. the error at frequency=0.

The name of these performance measure and the according value are included in the name of the `.pt` saved.

    BEST_L1Loss0.0642937_epochN_NAME.pt
    BEST_mse0.0812230_epochN_NAME.pt
    BEST_dc_error0.834125_epochN_NAME.pt

The training curves data is saved in 

    training_NAME.csv

And at the very end of the training, the best model parameters and best loss are all appended to the files:

    all_best_L1Loss.csv
    all_best_mse.csv
    all_best_dc_error.csv

The latter provide an easy way to compare all models trained in the folder so far.

## RANDOM SEARCH
The script `random_search.py` will train many neural networks randomly picking parameters. The parameters and ranges for this random pick are defines in the script:

`random_search.py`

    ...
    # the random pick is recursive:
    #   a list of (lists/tuples) will return a list of random picks
    #   a tuple of (lists/tuples) will pick one list/tuple to choose from
    # at the end level
    # a list defines a range
    # a tuple defines a set to random.choice from
    # a standalone value will be returned

    search_space = {
        "layers": [128, [30,200], [40,800], 512],
        "lr": [0.001,0.00001],
        "batch_size": [5,200],
        "factor": [0.05,1], 
        "patience": [4,10],
        "weight_decay": (0, [0.0,0.8]),
        "dropout": (0, [0.0,0.8]),
    }
    ...

## RUNNING ON A CLUSTER
The script `submit.sh` show the correct way of using this code on compute clusters (cedar, graham, mammouth, etc.).

First it specifies the compute node requirements, and where the terminal output will go:

    #!/bin/bash
    #SBATCH --account=def-tremblay
    #SBATCH --time=0-71:59
    #SBATCH --mem-per-cpu=4000M 
    #SBATCH --job-name=deep_continuation
    #SBATCH --output=%x-%j.out  ### %x=job-name, %j=job-ID

then it changes to directory (fastest access to memory, faster than $scratch):

    cd $SLURM_TMPDIR

brings all the relevant code and data:

    mkdir job
    mkdir sdata
    cp ~/codes/deep_continuation/mlp/* job/
    cp ~/scratch/deep_cont/data/Database_Gaussian_beta20/Training/Pi.csv sdata/
    cp ~/scratch/deep_cont/data/Database_Gaussian_beta20/Training/SigmaRe.csv sdata/

setup all local python executable

    # create a local virtual environnement (on the compute node)
    module load python/3.7
    virtualenv --no-download env
    source env/bin/activate
    # install the relevant packages 
    # (--no-index means we use already downloaded packages)
    pip install --no-index --upgrade pip
    pip install --no-index numpy
    pip install --no-index torch

run the script `random_search.py` on the cluster:

    cd job
    mkdir results
    python random_search.py
    cd ..

copy the job folder back to the submit directory once it is over:

    DATE=$(date -u +%Y%m%d)
    cp -r job $SLURM_SUBMIT_DIR/running-id$SLURM_JOB_ID

Note that `random_search.py` also copies the directory periodically, thanks to the lines:

    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        os.system("cp -r ./ $SLURM_SUBMIT_DIR/running-id$SLURM_JOB_ID")

So every time one training is done the folder comes back.

## TODO

-  