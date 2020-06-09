#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-71:59
#SBATCH --mem-per-cpu=8000M 
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

cd $SLURM_TMPDIR

mkdir job
cp ~/codes/deep_continuation/mlp/* job/
cp -r ~/scratch/deep_continuation/data job/

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

cd job
mkdir results
python random_search.py
cd ..

mv job/data ./

DATE=$(date -u +%Y%m%d)
cp -r job $SLURM_SUBMIT_DIR/deep_cont_$DATE-id$SLURM_JOB_ID
