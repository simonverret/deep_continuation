# %%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from deep_continuation.training import MODELS_PATH, MLP
from deep_continuation.dataset import DATAPATH

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load validation set and standardize with training set stats
# # fixed T
# sigma    = np.load(os.path.join(DATAPATH, "default", "sigma_10000x1x1_seed555_Nw512_wmax20.npy"))
# pi       = np.load(os.path.join(DATAPATH, "default", "Pi_10000x1x1_seed555_Nwn128_beta30.npy"))
# train_pi = np.load(os.path.join(DATAPATH, "default", "Pi_100000x1x1_seed55555_Nwn128_beta30.npy"))
# model_path = os.path.join(MODELS_PATH, "best_model_n0.0_10000x1x1_seed555_beta30_512x128_wmax20.pt")

# fixed s
sigma      = np.load(os.path.join(DATAPATH, "default", "sigma_10000x1x1_seed555_Nw512_wmax20_std8.86.npy"))
pi         = np.load(os.path.join(DATAPATH, "default", "Pi_10000x1x1_seed555_Nwn128_beta0to60_std8.86.npy"))
train_pi   = np.load(os.path.join(DATAPATH, "default", "Pi_100000x1x1_seed55555_Nwn128_beta0to60_std8.86.npy"))
model_path = os.path.join(MODELS_PATH, "best_model_n0.0_10000x1x1_seed555_beta0to60_std8.86_512x128_wmax20.pt")

avg = train_pi.mean(axis=-2)
std = train_pi.std(axis=-2)
pi_stdized = (pi - avg)/std


#%% Apply model
model = MLP()
model.load_state_dict(torch.load(model_path))

input = torch.tensor(pi_stdized).to(device).float()
output = model(input).detach().numpy()


#%% Plot target 
i = np.random.randint(0,10000)
fig, ax = plt.subplots(1,3, figsize=(12,4))
plt.suptitle("i="+str(i))
x = np.arange(128)
ax[0].plot(x, avg, label='avg')
ax[0].fill_between(x, avg-std, avg+std, alpha=0.3, label='std')
ax[0].plot(x, pi[i], label='pi')
ax[0].legend()
ax[1].set_title("input")
ax[1].plot(pi_stdized[i], label='(pi-avg)/std')
ax[1].legend()
ax[2].set_title("results")
ax[2].plot(sigma[i], label="target")
ax[2].plot(output[i], label="output")
ax[2].legend()
plt.show()

