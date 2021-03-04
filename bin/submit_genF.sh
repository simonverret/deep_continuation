#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-17:59
#SBATCH --mem-per-cpu=16000M 
#SBATCH --job-name=genF
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

cd $SLURM_TMPDIR
module load httpproxy/1.0

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
python data_generator.py data/Fournier_valid.json --generate 10000 --beta 2 10 15 20 25 30 35 50 --rescale 4
python data_generator.py data/Fournier_train.json --generate 100000 --beta 2 10 15 20 25 30 35 50 --rescale 4

# mkdir -p ~/scratch/deep_continuation/data/
cp -r data/* ~/scratch/deep_continuation/data/
