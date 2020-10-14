#%%
import os
import wandb
import matplotlib.pyplot as plt
api = wandb.Api()

runs = api.runs("deep_continuation/beta_and_scale")
summary_list = [] 
config_list = [] 
name_list = [] 
id_list = []
for run in runs: 
    if run.state == "finished":
        summary_list.append(run.summary._json_dict) 
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 
        name_list.append(run.name)     
        id_list.append(run.id)

import pandas as pd 
summary_df = pd.DataFrame.from_records(summary_list) 
config_df = pd.DataFrame.from_records(config_list) 
name_df = pd.DataFrame({'wandb_name': name_list}) 
name_df = pd.DataFrame({'wandb_id': id_list}) 
all_df = pd.concat([name_df, config_df, summary_df], axis=1)

path_dict = {
    'F': '../data/Fournier/valid/',
    'G': '../data/G1/valid/',
    'B': '../data/B1/valid/',
}

loss_dict = {
    'mse': 0,
    'dcs': 0,
    'mae': 0,
    'dca': 0,
}
    
scale_dict = {
    'N': False,
    'R': True
}

noise_dict = {
    '0': 0,
    '5': 1e-5,
    '3': 1e-3,
    '2': 1e-2,
}

beta_dict = {
    'T10': [10.0],
    'T20': [20.0],
    'T30': [30.0],
    'l3T': [15.0, 20.0, 25.0], 
    'l5T': [10.0, 15.0, 20.0, 25.0, 30.0],
}

def change_beta_to_tag(beta):
    if beta == [10]:
        return 'T10'
    elif beta == [20]:
        return 'T20'
    elif beta == [30]:
        return 'T30'
    elif beta == [15,20,25]:
        return 'l3T'
    elif beta == [10,15,20,25,30]:
        return 'l5T'

all_df["beta"] = all_df["beta"].apply(change_beta_to_tag)

print("available columns")
for col in list(config_df.columns):
    print(col)


#%% 
metrics = [
    f"{l}_{p+n+b+s}"
    for s in ['N', 'R']#scale_dict.keys()
    for p in ['G']#path_dict.keys()
    for n in ['0']#noise_dict.keys()
    for b in beta_dict.keys()
    for l in ['mae']#loss_dict.keys()
]
sub_df = all_df[
    # (all_df['loss']=='mse') &\
    (all_df['out_unit']=="None") &\
    (all_df['standardize']==False)
]
sub_df.groupby(["rescale","beta"])\
.agg({metric:'min' for metric in metrics})\
.style.background_gradient(axis=None, low=0, high=-0.8)


#%% BARPLOT

metrics = { f"{L}_{D}{S}": [
    f"{L}_{D+n+b+S}"
    for n in ['3']#noise_dict.keys()
    for b in beta_dict.keys()
] for D in ['G','B','F'] for S in ['N','R'] for L in ['mse', 'mae']}

R_df = all_df[
    # (all_df['loss']=='mse') &\
    (all_df['out_unit']=="None") &\
    (all_df['standardize']==False)&\
    (all_df['rescale']==True)
]
N_df = all_df[
    # (all_df['loss']=='mse') &\
    (all_df['out_unit']=="None") &\
    (all_df['standardize']==False)&\
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


N_df.groupby(["beta"]).agg({metric:'min' for metric in metrics['mse_FN']}).plot(kind='bar', ax=ax1, width=0.6, title=f"without rescaling {metrics['mse_FN'][0]}", legend=False)
N_df.groupby(["beta"]).agg({metric:'min' for metric in metrics['mse_GN']}).plot(kind='bar', ax=ax2, width=0.6, legend=False)
N_df.groupby(["beta"]).agg({metric:'min' for metric in metrics['mse_BN']}).plot(kind='bar', ax=ax3, width=0.6, legend=False)
R_df.groupby(["beta"]).agg({metric:'min' for metric in metrics['mse_FR']}).plot(kind='bar', ax=ax4, width=0.6, title=f"with rescaling {metrics['mse_FR'][0]}", legend=False)
R_df.groupby(["beta"]).agg({metric:'min' for metric in metrics['mse_GR']}).plot(kind='bar', ax=ax5, width=0.6, legend=False)
R_df.groupby(["beta"]).agg({metric:'min' for metric in metrics['mse_BR']}).plot(kind='bar', ax=ax6, width=0.6, legend=False)

plt.tight_layout()
plt.savefig("compare_rescaling.pdf")

#%%

for col in all_df.columns:
    if "id" in col: print(col)

#%% DOWNLOAD THE WEIGHTS FILE   
# NOTE: This part requires torch 1.6 because of cedar. Work in the virtual env
from deep_continuation.train import MLP, default_parameters
from deep_continuation import utils
import torch
import yaml
import json

device = torch.device('cpu')


metric = 'mae_B0T30R'
# model_id = sub_df[all_df[metric] == sub_df[metric].min()]['wandb_id'].iloc[0]
model_id = '39zc99qn'
score = all_df[all_df['wandb_id'] == model_id][metric].iloc[0]
print(f"wandb_id: {model_id}")
print(f"{metric} = {score}")

if not os.path.exists(f"best_models/{model_id}.pt"):
    print("downloading...")
    run = api.run(f"deep_continuation/beta_and_scale/{model_id}")
    run.file("best_valid_loss_model.pt").download(replace=True)
    os.rename("best_valid_loss_model.pt", f"best_models/{model_id}.pt")
    run.file("config.yaml").download(replace=True)
    os.rename("config.yaml", f"best_models/{model_id}.yaml")
    print("done")
else:
    print("already downloaded")


# LOAD THE MODEL
with open(f"best_models/{model_id}.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    config.pop("wandb_version")
    config.pop("_wandb")
    for k, v in config.items():
        config[k] = v['value']

for k, v in config.items():
    print(k, v) 

args = utils.ObjectView(config)
mlp = MLP(args)
mlp.load_state_dict(torch.load(f"best_models/{model_id}.pt", map_location=device))
mlp.eval()
print(mlp)


# LOAD THE DATA
from deep_continuation import data
import matplotlib.pyplot as plt
import numpy as np

path_dict = {
    'F': '/Users/Simon/codes/deep_continuation/deep_continuation/data/Fournier/valid/',
    'G': '/Users/Simon/codes/deep_continuation/deep_continuation/data/G1/valid/',
    'B': '/Users/Simon/codes/deep_continuation/deep_continuation/data/B1/valid/',
}

# TEST ON TRAINING DATA
dataset = data.ContinuationData(
    path_dict[args.data],
    beta=args.beta,
    noise=args.noise,
    rescaled=args.rescale,
    standardize=args.standardize,
    base_scale=15 if args.data=="F" else 20
)


# #%% TEST ON OTHER DATA
# other_data = 'F'
# dataset = data.ContinuationData(
#     path_dict[other_data],
#     beta=[30],
#     noise=1e-5,
#     rescaled=args.rescale,
#     standardize=args.standardize,
#     base_scale=15 if other_data=="F" else 20
# )

# FIGURE

start=np.random.randint(0,100)

mse_crit = torch.nn.MSELoss()
mae_crit = torch.nn.L1Loss()

bs = args.batch_size
loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=bs,
    shuffle=False,
    drop_last=True,
    num_workers=0,
)

mlp.eval()
for (inputs, targets) in loader:
    inputs = inputs.to(device).float()
    targets = targets.to(device).float()
    outputs = mlp(inputs)
    break

x = inputs[0]
t = targets[0]
y = outputs[0]

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
