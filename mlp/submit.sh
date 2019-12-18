#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-71:59
#SBATCH --mem-per-cpu=4000M 
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

cd $SLURM_TMPDIR

mkdir job
mkdir sdata
cp ~/codes/deep_continuation/mlp/* job/
cp ~/scratch/deep_cont/data/Database_Gaussian_beta20/Training/Pi.csv sdata/
cp ~/scratch/deep_cont/data/Database_Gaussian_beta20/Training/SigmaRe.csv sdata/

# create a local virtual environnement (on the compute node)
module load python/3.7
virtualenv --no-download env
source env/bin/activate
# install the relevant packages 
# (--no-index means we use already downloaded packages)
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index torch

cd job
python random_search.py
cd ..

DATE=$(date -u +%Y%m%d)
cp -r job ~/scratch/deep_cont/deep_continuation_$DATE-id$SLURM_JOB_ID
