#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --time=0-71:59
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=8G 
#SBATCH --cpus-per-task=8
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

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
pip install wandb

cp -r ~/codes/deep_continuation $SLURM_TMPDIR/
cp -r ~/scratch/deep_continuation/data $SLURM_TMPDIR/deep_continuation/deep_continuation/
cd $SLURM_TMPDIR/deep_continuation/
pip install --no-index -e .
cd $SLURM_TMPDIR/deep_continuation/deep_continuation/


# TODO: We are currently waiting for a bug fix! 
# srun --exclusive --cpu-bind=cores -c1 --mem=4G python random_search.py $SLURM_TMPDIR &  # First job (don't forget the '&')
# srun --exclusive --cpu-bind=cores -c1 --mem=4G python random_search.py $SLURM_TMPDIR &  # Second job (don't forget the '&')
wait    # Wait for both jobs to finish


DATE=$(date -u +%Y%m%d)
cp -r wandb $SLURM_SUBMIT_DIR/wandb_$DATE-id$SLURM_JOB_ID
