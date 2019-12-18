#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --time=0-23:59
#SBATCH --mem-per-cpu=4000M 
#SBATCH --job-name=deep_cont_data_gen
#SBATCH --output=%x-%j.out      ### %x=job-name, %j=job-ID

cd $SLURM_TMPDIR
mkdir job
cp ~/codes/deep_continuation/generateDataConductivity/* job/
cd job
make
./generateDataCond
cd ..

DATE=$(date -u +%Y%m%d)
cp -r job ~/scratch/deep_cont/data_generated_$DATE-id$SLURM_JOB_ID
