#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-71:59
#SBATCH --mem-per-cpu=4000M 
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

cd $SLURM_TMPDIR

# create a local virtual environnement (on the compute node)
module load python/3.7
virtualenv --no-download env
source env/bin/activate
# install the relevant packages 
# (--no-index means we use already downloaded packages)
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index scipy
pip install --no-index matplotlib
pip install --no-index torch

mkdir job
cp -r ~/codes/deep_continuation/mlp/* job/

cd job
python data.py data/G3_train.json --generate 50000
python data.py data/G3_valid.json --generate 10000

cp -r data/G3 ~/scratch/deep_continuation/data/
