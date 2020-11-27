#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

from deep_continuation import utils
from deep_continuation import train
from deep_continuation import data
from deep_continuation import wandb_utils as wdbu

#%%
if torch.cuda.is_available():
    # torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
    print('using GPU')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('no GPU available')

def beta_to_tag(beta):
    if beta == [10]: return 'T10'
    elif beta == [20]: return 'T20'
    elif beta == [30]: return 'T30'
    elif beta == [15,20,25]: return 'l3T'
    elif beta == [10,15,20,25,30]: return 'l5T'

all_df = wdbu.download_wandb_table("deep_continuation/beta_and_scale")
all_df["beta_list"] = all_df["beta"]
all_df["beta"] = all_df["beta"].apply(beta_to_tag)
all_df = all_df.set_index('wandb_id', drop=False)

print("available columns")
for col in list(all_df.columns):
    print(col)


#%% BARPLOT
metrics = { 
    f"{L}_{D}{S}": [
        f"{L}_{D+n+b+S}"
        for n in ['3']#noise_dict.keys()
        for b in ['T10', 'T20', 'T30', 'l3T', 'l5T']
    ] 
    for D in ['G','B','F']
    for S in ['N','R']
    for L in ['mse', 'mae']
}

R_df = all_df[
    # (all_df['loss']=='mse') &\
    # (all_df['out_unit']=="None") &\
    # (all_df['standardize']==False)&\
    (all_df['rescale']==True)
]
N_df = all_df[
    # (all_df['loss']=='mse') &\
    # (all_df['out_unit']=="None") &\
    # (all_df['standardize']==False)&\
    (all_df['rescale']==False)
]

fig = plt.figure(figsize=[7.5,2.5], dpi=80)
ax1 = plt.subplot(321)
ax2 = plt.subplot(323, sharey=ax1)
ax3 = plt.subplot(325, sharey=ax1)
ax4 = plt.subplot(322, sharey=ax1)
ax5 = plt.subplot(324, sharey=ax1)
ax6 = plt.subplot(326, sharey=ax1)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)

# ax1.set_yscale('log')
ax1.set_ylabel('loss') 
ax1.set_ylim(0.000,0.01)

N_df.groupby(["beta"]).agg({m:'min' for m in metrics['mse_FN']}).plot(kind='bar', ax=ax1, width=0.6, title=f"without rescaling", legend=False)
N_df.groupby(["beta"]).agg({m:'min' for m in metrics['mse_GN']}).plot(kind='bar', ax=ax2, width=0.6, legend=False)
N_df.groupby(["beta"]).agg({m:'min' for m in metrics['mse_BN']}).plot(kind='bar', ax=ax3, width=0.6, legend=False)
R_df.groupby(["beta"]).agg({m:'min' for m in metrics['mse_FR']}).plot(kind='bar', ax=ax4, width=0.6, title=f"with rescaling", legend=False)
R_df.groupby(["beta"]).agg({m:'min' for m in metrics['mse_GR']}).plot(kind='bar', ax=ax5, width=0.6, legend=False)
R_df.groupby(["beta"]).agg({m:'min' for m in metrics['mse_BR']}).plot(kind='bar', ax=ax6, width=0.6, legend=False)

plt.tight_layout()
# plt.savefig("compare_rescaling.pdf")
plt.show()


#%% TABLE WITH 
tag = 'mse_BR'
index = 'wandb_id'
sub_df = R_df.set_index(index, drop=False)
sub_df.groupby(["beta"]).agg({m:'min' for m in metrics[tag]}).style.background_gradient(axis=None, low=0, high=-0.008)
sub_df.groupby(["beta"]).agg({m:'idxmin' for m in metrics[tag]})


#%%  GET MODEL
metric_name = 'mse_B0T30R'
sub_df = R_df
model_id = sub_df[all_df[metric_name] == sub_df[metric_name].min()]['wandb_id'].iloc[0]
# model_id = 'z3m7fq6n' # '33du76v1' ## (clean)

print(f"best on {metric_name}: {model_id} - {all_df.data.loc[model_id]}, {all_df.noise.loc[model_id]}, {all_df.beta.loc[model_id]}, {all_df.out_unit.loc[model_id]}, {all_df.epoch.loc[model_id]}, {all_df['epoch_'+metric_name].loc[model_id]}") 
print(f"  old: {all_df[metric_name].loc[model_id]}")

mlp, args = wdbu.get_wandb_model(model_id, device)
dataset = data.ContinuationData(
    standardize=args.standardize,
    **(wdbu.data_args_from_name(metric_name))
)

metric = data.EvaluationMetric(
    dataset=dataset,
    std=args.standardize,
    bs=args.metric_batch_size,
    num_workers=args.num_workers,
    **(wdbu.metric_args_from_name(metric_name))
)

loss = metric_name.split('_')[0]
metric.evaluate(mlp, device, fraction=1.0)
print(f"  new: {metric.loss_values[loss]}")


#%% PLOT TEST SPECTRA
start=np.random.randint(0,100)

mse_crit = torch.nn.MSELoss()
mae_crit = torch.nn.L1Loss()

bs = args.batch_size
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

mlp.eval()
for (inputs, targets) in loader:
    inputs = inputs.to(device).float()
    targets = targets.to(device).float()
    outputs = mlp(inputs)
    break

x = inputs.cpu()[0]
t = targets.cpu()[0]
y = outputs.cpu()[0]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[4,8])
ax4.get_shared_x_axes().join(ax2, ax3, ax4)
ax2.set_xticklabels([])
ax3.set_xticklabels([])

ax1.plot(x.detach().numpy())
ax1.set_title('input (matsubara freqs)',loc='right', pad=-12)
ax1.set_xlabel('iwn')

ax2.plot(t.detach().numpy())
ax2.set_title('target (real freqs)',loc='right', pad=-12)

ax3.plot(y.detach().numpy())
ax3.set_title('NN output (real freqs)',loc='right', pad=-12)

e = y-t
mse = mse_crit(y,t)
mae = mae_crit(y,t)
ax4.plot(e.detach().numpy())
ax4.set_title(f'(MSE={mse:4f}), (MAE={mae:4f})',loc='right', pad=(-12),)
ax4.set_ylim(-0.02,0.02)
ax4.set_xlabel('w')

plt.show()
# plt.savefig('last_plot.pdf')

