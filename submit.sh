#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-18:59
#SBATCH --mem-per-cpu=16000M 
#SBATCH --job-name=deep_continuation
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

cd $SLURM_TMPDIR

mkdir job
cp ~/codes/deep_continuation/* job/
tar -xf ~/scratch/deep_continuation/reza_data.tar

module load python/3.7
virtualenv --no-download env
source env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index torch

cd job
python random_search.py
cd ..

DATE=$(date -u +%Y%m%d)
cp -r job ~/scratch/deep_continuation/deep_continuation_$DATE-id$SLURM_JOB_ID
