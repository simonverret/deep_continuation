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
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from deep_continuation.train import MLP
import data
import utils



try: filename = sys.argv[1]
except IndexError: raise IndexError('provide the filename as first argument')





#%%

filename = 'results_beluga/deep_cont_20200227-id5933377/results/BEST_G1bse/mse0.001726982_epoch999_mlp128-588-128-418-446-223-512_MSELoss_G1n0.0_bs479_lr2e-05_wd0_0_None_wup_sch0.083-9.pt'
location = filename.split("BEST_", 1)[0]
name = filename.split("mlp", 1)[1].strip('.pt')
params_file = f'{location}params_mlp{name}.json'

with open(params_file) as f:
    params = json.load(f)
args = utils.ObjectView(params)

try: datafile = sys.argv[2]
except IndexError: dataset = args.data

try: number = int(sys.argv[3])
except: number = 1


## IMPORT THE MODEL
mlp = MLP(args)
mlp.load_state_dict(torch.load(filename))
mlp.eval()

## RELOAD THE DATA
dataset = data.ContinuationData(f'data/{datafile}/valid/', noise=0.0)

## PLOT RANDOM DATA
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[4,8])
ax4.get_shared_x_axes().join(ax2, ax3, ax4)
ax2.set_xticklabels([])
ax3.set_xticklabels([])

start=np.random.randint(0,100)

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
    ax4.plot(e.detach().numpy())
    ax4.set_title('difference',loc='right', pad=(-12),)
    ax4.set_xlabel('w')

plt.show()
# plt.savefig('last_plot.pdf')
