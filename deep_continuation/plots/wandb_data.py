#%%
import numpy as np
import matplotlib.pyplot as plt
import torch

from deep_continuation import utils
from deep_continuation import train
from deep_continuation import data
from deep_continuation import wandb_utils as wdbu

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
plt.savefig("compare_rescaling.pdf")
# plt.show()


#%% TABLE WITH 
tag = 'mse_BR'
index = 'wandb_id'
# index = 'out_unit'
sub_df = R_df.set_index(index, drop=False)
sub_df.groupby(["beta"]).agg({m:'min' for m in metrics[tag]}).style.background_gradient(axis=None, low=0, high=-0.008)
# sub_df.groupby(["beta"]).agg({m:'idxmin' for m in metrics[tag]})


#%%  GET MODEL
metric_name = 'mse_B3T30R'
sub_df = R_df
# model_id = sub_df[all_df[metric_name] == sub_df[metric_name].min()]['wandb_id'].iloc[0]
# model_id = '21xrjubd' # this one is not horrible on actual data
# model_id = '2i86ocj9' # best on B and G multiple T! pretty good on actual data, trained with DCS loss (WTF?)
# model_id = '30a1ffdn' #
# model_id = '302fdds9' # very noisy
# model_id = 'bjksdy3j' # also noisy
# model_id = '2l26i59j' # good in its training domain, but fails outside
model_id = '30a1ffdn' # The chosen one

print(f"best on {metric_name}: {model_id} - {all_df.data.loc[model_id]}, {all_df.loss.loc[model_id]}, {all_df.noise.loc[model_id]}, {all_df.beta.loc[model_id]}, {all_df.out_unit.loc[model_id]}, {all_df.epoch.loc[model_id]}, {all_df['epoch_'+metric_name].loc[model_id]}") 
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


#%% APPLICATION ON REAL DATA

rx, ry = np.loadtxt('reza_pol.txt').T
# sigma = np.loadtxt('reza_SigmaRe_scaled_4.000000.csv', delimiter=',')
# plt.plot(sigma.T)
# plt.plot(x.detach().numpy())
# plt.plot(ry)
#%%
mex, mey = np.loadtxt('optimal_spectral_function_tem0.03333_alpha1.59e+07.dat').T

#%%

inputs = torch.Tensor(ry).float().unsqueeze(0)
# print(inputs)
outputs = mlp(inputs).squeeze()
plt.plot(np.linspace(0,100, 512) , 512/200*outputs.detach().numpy())
plt.plot(mey)

#%% THESE DATA FROM REZA are unusable because not the right format
x1, y1 = np.loadtxt("OpticalConductivity1.dat").T
x2, y2 = np.loadtxt("OpticalConductivity1.dat").T
x3, y3, _ = np.loadtxt("OpticalPolarization.dat").T
# plt.plot(y1)
# plt.plot(y2)
# plt.plot(y3)

inputs = torch.Tensor(y3[:128]/y3[0]).float()
inputs = (inputs/inputs[0]).unsqueeze(0)
outputs = mlp(inputs).squeeze()
plt.plot(outputs.detach().numpy())


#%% NEW SCALE FIGURE
import deep_continuation.data_generator as gen

N_wn = 20
fac=1.2
wmax = 20
beta1 = 20
beta2 = fac*beta1
beta3 = beta1/fac

## SIGMA FUNCTIONS

generator = gen.BetaMix(
    nmbrs=[[0, 4],[0, 6]],
    cntrs=[[0.00, 0.00], [4.00, 16.0]],
    wdths=[[0.20, 4.00], [0.40, 4.00]],
    wgths=[[0.00, 1.00], [0.00, 1.00]],
    arngs=[[2.00, 10.0], [0.50, 10.0]],
    brths=[[2.00, 10.0], [0.50, 10.0]]
)
sigma_func, _ = generator.generate_functions()

sigma1 = lambda x: sigma_func(x) # gen.sum_on_args(gen.free_beta, x, c, w, h, a, b)
sigma2 = lambda x: fac*sigma1(fac*x)
sigma3 = lambda x: sigma1(x/fac)/fac

W = np.linspace(0,wmax,512)
S1 = sigma1(W)
S2 = sigma2(W)
S3 = sigma3(W)

## PI FUNCTIONS
pi1 = lambda x: gen.pi_integral(x, sigma1, grid_end=wmax)
pi2 = lambda x: gen.pi_integral(x, sigma2, grid_end=wmax)
pi3 = lambda x: gen.pi_integral(x, sigma3, grid_end=wmax)

WN1 = (2*np.pi/beta1) * np.arange(0,N_wn)
WN2 = (2*np.pi/beta2) * np.arange(0,N_wn)
WN3 = (2*np.pi/beta3) * np.arange(0,N_wn)
P22 = pi2(WN2)
P33 = pi3(WN3)

zmax = N_wn*(2*np.pi/beta3)
Z = np.linspace(0,zmax,512)
P1 = pi1(Z)
P2 = pi2(Z)
P3 = pi3(Z)


#%% MODEL PREDICTION
INF = 1e10
s = INF**2*gen.pi_integral(INF, sigma1, grid_end=wmax)
swmax = np.cbrt(s) * 4.0
SW = np.linspace(0, swmax, 512)
targets = sigma1(SW) #* swmax/wmax

INF = 1e10
s2 = INF**2*gen.pi_integral(INF, sigma2, grid_end=wmax)
swmax2 = np.cbrt(s2) * 4.0
SW2 = np.linspace(0, swmax2, 512)
targets2 = sigma2(SW2) #* swmax2/wmax

plt.plot(targets)
plt.plot(targets2)

#%%
wn_grid = (2*np.pi/beta1) * np.arange(0,128)
inputs = pi1(wn_grid)
pi0 = inputs[0]
inputs = torch.Tensor(inputs).float()
inputs = (inputs/pi0).unsqueeze(0)
outputs = mlp(inputs).squeeze().detach().numpy() * pi0

#%% FIGURE
import matplotlib.colors as mcolors

plt.rcParams.update({
    'text.usetex': False,
    'text.latex.preamble': [r'\usepackage{amsmath}',
]})

C = list(mcolors.TABLEAU_COLORS)
fig = plt.figure(figsize=[7.,3.], dpi=80)
ax1 = plt.subplot(141)
ax2 = plt.subplot(142)
ax3 = plt.subplot(143)
ax4 = plt.subplot(144)

ax1.plot(Z, P2, c=C[3], lw=1)
ax1.plot(Z, P3, c=C[0], lw=1)
ax1.plot(WN2, P22, '.', c=C[3], ms=6 , label=r'$\Pi[\sigma(\omega)](i s\omega_n)$')
ax1.plot(WN3, P33, '.', c=C[0], ms=10, label=r'$\Pi[s\sigma(s\omega)](i\omega_n)$')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize='small', handlelength=1, frameon=False)
ax1.set_ylim(0.55,1)
ax1.set_xlim(0,8)
ax1.set_yticks([.6, .7, .8, .9, 1])
ax1.set_yticklabels(['.6', '.7', '.8', '.9', '1'])
ax1.set_xlabel(r"$\omega_n$")
ax1.set_ylabel(r"$\Pi(i\omega_n)$")
ax1.set_xticks([0,2,4,6,8])


ax2.plot(P33/P33[0], '.', c=C[0], ms=10, label=r'$\Pi[\sigma(\omega)](i s\omega_n)$')
ax2.plot(P22/P22[0], '.', c=C[3], ms=6 , label=r'$\Pi[s\sigma(s\omega)](i\omega_n)$')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize='small', handlelength=1, frameon=False)
ax2.set_ylim(0.55,1)
ax2.set_xlim(0,20)
ax2.set_ylabel(r"$[\Pi]_n$")
ax2.set_xlabel(r"$n$")
ax2.set_yticks([.6, .7, .8, .9, 1])
ax2.set_yticklabels(['.6', '.7', '.8', '.9', '1'])
ax2.set_xticks([0,5,10,15,20])


ax3.plot(targets, c='k', label=r"target")
mepred = np.loadtxt('optimal_spectral_function_tem0.05_alpha1.73e+06.dat').T
cap = 100
corr = (swmax/wmax) * (512/swmax)
ax3.plot(mepred[0][:cap]*corr, mepred[1][:cap], c=C[2], label=r"max entropy")
ax3.plot(outputs, '.', ms=4, c=C[1])#, label=r"prediction")
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize='small', handlelength=1, frameon=False)
ax3.set_ylabel(r"$[\sigma]_m$")
ax3.set_xlabel(r"$m$")
ax3.set_ylim(0,.45)
ax3.set_yticks([0, .1, .2, .3, .4])
ax3.set_yticklabels(['0', '.1', '.2', '.3', '.4'])
ax3.set_xticks([0,256,512])

ax4.plot(W, S2, c=C[3], label=r"$\sigma(\omega)$")
ax4.plot(W, S3, c=C[0], label=r"$s\sigma(s\omega)$")
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize='small', handlelength=1, frameon=False)
ax4.set_ylabel(r"$\sigma(\omega)$")
ax4.set_xlabel(r"$\omega$")
ax4.set_ylim(0,.6)
ax4.set_yticks([0, .2, .4, .6])
ax4.set_yticklabels(['0', '.2', '.4', '.6'])
ax4.set_xticks([0, 5, 10, 15, 20])

plt.tight_layout()
plt.savefig('rescale_NN.pdf')

#%%

np.savetxt('Pi_figure.csv', inputs, delimiter=',')
np.savetxt('sigma_figure.csv', targets, delimiter=',')


#%% CONSISTENCY CHECK
from scipy import integrate
print(integrate.simps(mepred[1],mepred[0] *swmax/wmax))
print(integrate.simps(targets, SW))
print(integrate.simps(outputs, SW))
