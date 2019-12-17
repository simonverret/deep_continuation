#
#   deep_continuation
#
#   © Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from deep_continuation import MLP
from data_reader import RezaDataset

## READ THE PARAMS USED
class ObjectView():
    def __init__(self,dict):
        self.__dict__.update(dict)
with open('results/params_mlp128-512-512_bs1500_lr0.01_wd0_drop0_wup_scheduled0.5-8.json') as f:
    params = json.load(f)
args = ObjectView(params)

## IMPORT THE MODEL
mlp = MLP(args)
mlp.load_state_dict(torch.load('results/BEST_loss0.063989654_epoch774_mlp128-512-512_bs1500_lr0.01_wd0_drop0_wup_scheduled0.5-8.pt'))

## RELOAD THE DATA
dataset = RezaDataset(args.path)

## PLOT RANDOM DATA
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[4,8])
ax4.get_shared_x_axes().join(ax2, ax3, ax4)
ax2.set_xticklabels([])
ax3.set_xticklabels([])

start=np.random.randint(0,500)
end=start+5
for ii in range(start,end):
    x = torch.tensor(dataset[ii][0]).float()
    ax1.plot(x.detach().numpy())
    ax1.set_title('input (matsubara freqs)',loc='right', pad=-15)
    ax1.set_xlabel('iwn')

    t = torch.tensor(dataset[ii][1]).float()
    ax2.plot(t.detach().numpy())
    ax2.set_title('target (real freqs)',loc='right', pad=-15)

    y = mlp(x)
    ax3.plot(y.detach().numpy())
    ax3.set_title('NN output (real freqs)',loc='right', pad=-15)

    e = y-t
    ax4.plot(e.detach().numpy())
    ax4.set_title('difference',loc='right', pad=(-15),)
    ax1.set_xlabel('w')

plt.savefig('results.pdf')
