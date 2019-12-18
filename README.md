# DEEP CONTINUATION
Simon Verret,
Reza Nourafkan,
& Andre-Marie Tremablay

Research project on analytical continuation of the conductivity with neural networks.

## USAGE
Train the default fully connected neural network (MLP) with:

    cd mlp
    python deep_continuation.py

The default dataset location is a `../sdata/` directory relative to the `mlp` directory. The training and validation data are splitted automatically. The name of datafiles, from the mlp directory, are: `../data/Pi.csv` (input) `../data/SigmaRe.csv` (target output).

You can pass arguments to the python script. For example:

    python deep_continuation.py --path ./Database/

will fetch the dataset from `./Database/` instead of the default `../sdata/`. However, the filenames still have to be `Pi.csv` and `SigmaRe.csv`. Other examples of paramters you can would be:

    python deep_continuation.py --lr 0.03 --seed 120

which will change the learning rate and the seed determining random number. Refer to the code for more details. All such tunable are also be defined in `params.json`.

## OUTPUTS
Each time the training is launched, a (very long but very explicit) name is given to the current run, for example:

    NAME = mlp128-512-1024_bs1500_lr0.01_wd0_drop0_wup_scheduled0.5-8

which can be read as follow: MLP with input size 128, two hidden layers of size 512 `--h1 512` and 512 `--h2 1024`, batch size used was 1500 `--batch_size 1500`, learning rate 0.01 `--lr 0.01`, no weight decay `--weight_decay 0`, no dropout `--dropout 0`, with a learning rate warmup `--warmup` and scheduler that multiply the learning rate by 0.5 `--factor 0.5` every time the training loss does not reduces for 8 epoch (`--patience 8`). Note that other parameters are not included in the name, for simplicity. These parameter are saved in the file

    params_NAME.jason

While `deep_continuation.py` runs, it continually updates a save of the model that performed the best according to three measures, the loss function used for training, for example the L1 loss, specified with `--loss L1Loss`, the means square error for all outputs `mse`, and the DC error `dc_error`, i.e. the error at frequency=0. Including the value of these score and the epoch at which they were obtained in the file name ensure that the code output unique files for each training:

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

    search_ranges = {
        "h1": [2,5],                #x10 implicit
        "h2": [2,5],                #x10 implicit
        "lr": [0.001,0.00001],
        "batch_size": [5,200],      #x10 implicit
        "factor": [0.05,1], 
        "patience": [4,10],
        "weight_decay": [0.0,0.8],
        "dropout": [0.0,0.8],
    }
    ...

the implicit factor of 10 are implemented below in the script. One may want to explicitely modify this script.

Note at the end of each command training, the command

## RUNNING ON CLUSTER
The script `submit.sh` show the correct way of using this code on compute clusters (cedar, graham, mammouth, etc.).

    #!/bin/bash
    #SBATCH --account=def-tremblay
    #SBATCH --time=0-71:59
    #SBATCH --mem-per-cpu=4000M 
    #SBATCH --job-name=deep_continuation
    #SBATCH --output=%x-%j.out  ### %x=job-name, %j=job-ID

specifies the compute node requirements, and where the terminal output will go

    cd $SLURM_TMPDIR

relocate in the local directory (fastest access to memory, faster than $scratch)

    mkdir job
    mkdir sdata
    cp ~/codes/deep_continuation/mlp/* job/
    cp ~/scratch/deep_cont/data/Database_Gaussian_beta20/Training/Pi.csv sdata/
    cp ~/scratch/deep_cont/data/Database_Gaussian_beta20/Training/SigmaRe.csv sdata/

brings all the relevant code and data

    # create a local virtual environnement (on the compute node)
    module load python/3.7
    virtualenv --no-download env
    source env/bin/activate
    # install the relevant packages 
    # (--no-index means we use already downloaded packages)
    pip install --no-index --upgrade pip
    pip install --no-index numpy
    pip install --no-index torch

setup all local python executable

    cd job
    mkdir results
    python random_search.py
    cd ..

run the script `random_search.py` on the cluster

    DATE=$(date -u +%Y%m%d)
    cp -r job ~/scratch/deep_cont/deep_continuation_$DATE-id$SLURM_JOB_ID

copy the job folder back to scratch.

Note that the `random_search.py` script also copies the job folder periodically, thanks to the lines:

    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        os.system("cp -r ./ $SLURM_SUBMIT_DIR/running-id$SLURM_JOB_ID")


