#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-11:59
#SBATCH --mem-per-cpu=4000M 
#SBATCH --job-name=FournierB
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
cp -r ~/codes/deep_continuation/* job/
cd job
pip install --no-index -e .
cd deep_continuation
python data_generator.py data/FournierB_valid.json --generate 10000 --beta 2 10 15 20 25 30 35 50 --rescale 4
python data_generator.py data/FournierB_train.json --generate 100000 --beta 2 10 15 20 25 30 35 50 --rescale 4

cp -r data/* ~/scratch/deep_continuation/data/
