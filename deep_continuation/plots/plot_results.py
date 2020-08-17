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

args = utils.ObjectView(default_parameters)

# weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/yvmgqz69")  # G1
weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/wx4re7kd")  # G1
args.data = "G1"
# weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/2k6y30k8")  # P1
# weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/sigcg270")  # P1
# args.data = "P1"
# weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/3c68phvh")  # P2
# weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/2xvfxwlw")  # P2
# args.data = "P2"
# weights_file = wandb.restore('best_weights.pt', run_path="deep_continuation/nrm_smpl_mlp/9omya0ju")  # B1
# args.data = "B1"


try: datafile = sys.argv[2]
except IndexError: datafile = args.data

try: number = int(sys.argv[3])
except: number = 3


## IMPORT THE MODEL
mlp = MLP(args)
mlp.load_state_dict(torch.load(weights_file.name))
mlp.eval()

## RELOAD THE DATA
dataset = data.ContinuationData(f'data/{datafile}/valid/', noise=0.0)

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
