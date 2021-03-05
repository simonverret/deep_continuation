#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=0-71:57
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node=3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=46G
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

# create a local virtual environnement (on the compute node)
cd $SLURM_TMPDIR
# module load httpproxy/1.0
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

# wandb off
srun -l --multi-prog ~/codes/deep_continuation/bin/silly4.conf
# cp -r wandb $SLURM_SUBMIT_DIR/wandb_$SLURM_JOB_ID