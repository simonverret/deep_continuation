#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-71:59
#SBATCH --mem-per-cpu=8000M 
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

# igpu='salloc --time=0-02:00 --gres=gpu:v100l:1 --mem=46G --account=def-bengioy'

# create a local virtual environnement (on the compute node)
cd $SLURM_TMPDIR
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

cp -r ~/codes/deep_continuation $SLURM_TMPDIR/
cp -r ~/scratch/deep_continuation/data $SLURM_TMPDIR/deep_continuation/deep_continuation/
cd $SLURM_TMPDIR/deep_continuation/
pip install --no-index -e .
cd $SLURM_TMPDIR/deep_continuation/deep_continuation/

python random_search.py

mv $SLURM_TMPDIR/deep_continuation/deep_continuation/ ./

DATE=$(date -u +%Y%m%d)
cp -r job $SLURM_SUBMIT_DIR/wandb/
