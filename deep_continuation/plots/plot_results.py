#
#   deep_continuation
#
#   © Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

#%%
import sys
import re
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from deep_continuation.simpler_mlp import MLP, default_parameters
from deep_continuation import data
from deep_continuation import utils

# try: filename = sys.argv[1]
# except IndexError: raise IndexError('provide the filename as first argument')

try: run_name = sys.argv[1]
except IndexError: print('requires a wandb run name as first arg, ex: 2k6y30k8')

try: number = int(sys.argv[3])
except: number = 3

api = wandb.Api()
run = api.run(f"deep_continuation/nrm_smpl_mlp/{run_name}")
args = utils.ObjectView(run.config)
weights_file = wandb.restore('best_weights.pt', run_path=f"deep_continuation/nrm_smpl_mlp/{run_name}")  # B1

## IMPORT THE MODEL
mlp = MLP(args)
mlp.load_state_dict(torch.load(weights_file.name))
mlp.eval()

## RELOAD THE DATA
dataset = data.ContinuationData(f'data/{args.data}/train/', noise=0.0)

## PLOT RANDOM DATA
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[4,8])
ax4.get_shared_x_axes().join(ax2, ax3, ax4)
ax2.set_xticklabels([])
ax3.set_xticklabels([])

start=np.random.randint(0,100)

criterion = torch.nn.MSELoss()
criterion = torch.nn.MSELoss()


end=start+number
for ii in range(start,end):
    x = torch.tensor(dataset[ii][0]).float()
    ax1.plot(x.detach().numpy())
    ax1.set_title('input (matsubara freqs)',loc='right', pad=-12)
    ax1.set_xlabel('iwn')

    t = torch.tensor(dataset[ii][1]).float()
    ax2.plot(t.detach().numpy())
    ax2.set_title('target (real freqs)',loc='right', pad=-12)

    y = mlp(x.unsqueeze(0)).squeeze()
    ax3.plot(y.detach().numpy())
    ax3.set_title('NN output (real freqs)',loc='right', pad=-12)

    e = y-t
    mse = criterion(y,t)
    ax4.plot(e.detach().numpy())
    ax4.set_title(f'difference (MSE={mse})',loc='right', pad=(-12),)
    ax4.set_xlabel('w')

plt.show()
# plt.savefig('last_plot.pdf')
