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
import numpy as np
import torch
import json
from deep_continuation import MLP
import data
import utils
from glob import glob

train_data = 'G1bse'
loss = 'mse'
test_data = 'G1'
test_noise = 0.00
test_criteria = 'mse'

## finding the best model
best_score = 10.0
best_name = None
best_epoch = None
best_model_path = None
best_params_path = None

for job_path in glob('results_beluga/deep_cont_*/results/'):
    for best_path in glob(job_path+'BEST*/'):
        best_dir = best_path.replace(job_path,'')
        data_str = best_path.replace(job_path+'BEST_','').strip('/')
        if data_str == train_data:
            for model_path in glob(best_path+loss+'*'):
                model_file = model_path.replace(best_path,'')
                ## manage the underscor of dc_error
                if loss == 'dc_error':
                    score_epoch_name = model_file.split('_',3)
                    score_epoch_name.remove('dc')
                    score_epoch_name[0] = 'dc_'+score_epoch_name[0]
                else:
                    score_epoch_name = model_file.split('_',2)
                score = float(score_epoch_name[0].replace(loss,''))
                epoch = int(score_epoch_name[1].replace('epoch',''))
                name = score_epoch_name[2].strip('.pt')               
                if score < best_score:
                    best_score = score
                    best_name = name
                    best_epoch = epoch
                    best_model_path = model_path
                    best_params_path = f'{job_path}params_{name}.json'

with open(best_params_path) as f:
    params = json.load(f)
args = utils.ObjectView(params)

## IMPORT THE MODEL
mlp = MLP(args)
mlp.load_state_dict(torch.load(best_model_path))
mlp.eval()

## RELOAD THE DATA
dataset = data.ContinuationData(f'data/{test_data}/valid/', noise=test_noise)
avg_score = best_score


best10 = [10 for _ in range(10)]
best10_idx = [0 for _ in range(10)]
worst10 = [1e-12 for _ in range(10)]
worst10_idx = [0 for _ in range(10)]
avg10 = [10 for _ in range(10)]
avg10_idx = [0 for _ in range(10)]

for ii in range(len(dataset)):
    x = torch.tensor(dataset[ii][0]).float()
    t = torch.tensor(dataset[ii][1]).float()
    y = mlp(x.unsqueeze(0)).squeeze()
    loss = torch.sum((y-t)**2)/len(y)
    loss = loss.item()

    tobeat = max(best10)
    idx = best10.index(tobeat)
    if loss < tobeat:
        best10[idx] = loss
        best10_idx[idx] = ii
    
    tobeat = min(worst10)
    idx = worst10.index(tobeat)
    if loss > tobeat:
        worst10[idx] = loss
        worst10_idx[idx] = ii

    tobeat = max(avg10)
    idx = avg10.index(tobeat)
    if (loss - avg_score) < tobeat:
        avg10[idx] = loss
        avg10_idx[idx] = ii

print(best10)
print(best10_idx)
print(worst10)
print(worst10_idx)
print(avg10)
print(avg10_idx)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

# rc('text', usetex=True)
rc('axes', linewidth=0.5)
rc('xtick.major', width=0.5)
rc('ytick.major', width=0.5)
# plt.rc('font', family='Helvetica')

# for ii in avg10_idx:
## PLOT RANDOM DATA
start=0
end=start+30
for ii in range(start,end):
    
    x = torch.tensor(dataset[ii][0]).float()
    t = torch.tensor(dataset[ii][1]).float()
    y = mlp(x.unsqueeze(0)).squeeze()
    e = y-t
    wn = np.arange(128)*2*np.pi/10.0
    w = np.linspace(0,20,512)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[4,8])
    fig = plt.figure(figsize=(4, 5)) 
    gs = gridspec.GridSpec(40, 1) 
    plt.gcf().subplots_adjust(bottom=0.1)
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(right=0.98)
    plt.gcf().subplots_adjust(top=0.98)

    ax1 = plt.subplot(gs[ 0:10 ,0])
    ax3 = plt.subplot(gs[32:  ,0])
    ax2 = plt.subplot(gs[15:31,0], sharex=ax3)
    
    ax1.plot(wn, x.detach().numpy(), color='#555555')
    ax1.text(0.95, 0.9, '$\Pi$ input', ha='right', va='top', transform=ax1.transAxes)
    ax1.set_xlabel('$\omega_n$')
    ax1.set_xlim(0,128*2*np.pi/10.0)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('$\Pi(i\omega_n)$')

    ax2.plot(w, t.detach().numpy(), color='#000000', label='target')
    ax2.plot(w, y.detach().numpy(), color='#ff0000', label='neural net')
    ax2.set_ylabel('$\sigma(\omega)$')
    ax2.set_xlim(0,20)
    ax2.set_ylim(0,1.)
    ax2.legend(frameon=False, handlelength=0.8)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(w, e.detach().numpy(), color='#ff0000')
    ax3.plot(np.zeros(512) , color='#000000')
    ax3.text(0.95, 0.15, 'difference', ha='right', va='bottom', transform=ax3.transAxes)
    ax3.set_ylim(-0.1,0.1)
    ax3.set_ylabel('error')
    ax3.set_xlabel('$\omega$')
    
    print(ii)
    plt.savefig(f'fit{ii}_plot.pdf', bbox_inches='tight')
    # plt.show()
