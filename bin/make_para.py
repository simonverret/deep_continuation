import os
import json
import glob



scale_dict = {
        'N': False,
        'R': True
    }

beta_dict = {
        'T10': [10.0],
        'T20': [20.0],
        'T30': [30.0],
        'l3T': [15.0, 20.0, 25.0], 
        'l5T': [10.0, 15.0, 20.0, 25.0, 30.0],
    }

path_dict = {
        'F': 'data/Fournier/valid/',
        'G': 'data/G1/valid/',
        'B': 'data/B1/valid/',
    }
import glob

file_list = []
for p, path in path_dict.items():
    for b, beta, in beta_dict.items():
        for s, scale in scale_dict.items():
            paradict = {
                'data':p,
                'rescale':scale,
                'beta':beta,
            }
            filename = f"{p+b+s}.json"
            with open(filename, 'w') as fp:
                json.dump(paradict, fp)
            file_list.append(filename)

nruns = 10
njobs = 3
ncpus = 4
assert(nruns*njobs == 30)

for filename in glob.glob('silly*.conf'):
    os.remove(filename)


for i in range(nruns):
    with open(f"silly{i}.conf", 'w') as f:
        for i, filename in enumerate(file_list[i*njobs:(i+1)*njobs]):
            f.write(f"{i} python random_search.py ../bin/{filename} --num_workers {ncpus}\n")





for i in range(nruns):

    submit_script = f'''#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=0-71:57
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node={njobs}
#SBATCH --ntasks={njobs}
#SBATCH --cpus-per-task={ncpus}
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
srun -l --multi-prog ~/codes/deep_continuation/bin/silly{i}.conf
# cp -r wandb $SLURM_SUBMIT_DIR/wandb_$SLURM_JOB_ID'''

    with open(f"submit{i}.sh", 'w') as f:
        f.write(submit_script)

        