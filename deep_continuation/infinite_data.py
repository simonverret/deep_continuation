#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_continuation.function_generator import (
    default_parameters,
    pi_integral,
    simple_plot,
    SigmaPiGenerator,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% data
default_parameters.update({'wmax':1.0})
generator = SigmaPiGenerator.factory(**default_parameters)
Ns = 1000
Nw = 1000
wmax = default_parameters['wmax']
Nwn = 128

data_list = []
for i in range(Ns):
    sigma_func, pi_func = generator.generate()
    beta = 400
    wn = np.arange(0, Nwn)*2*np.pi/beta
    w = np.random.uniform(0, wmax, Nw)
    pi = pi_func(wn)
    sigma = sigma_func(w)
    pi_repeated = np.broadcast_to(pi, [Nw, Nwn])
    w_column = w[:,np.newaxis]
    sigma_column = sigma[:,np.newaxis]
    data = np.concatenate([pi_repeated, w_column, sigma_column], axis=1)
    data_list.append(data)
    if i%(Ns//10) == 0: print(i)
dataset = np.concatenate(data_list, axis=0)

#%% model
class DeepContinuor(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim):
        super().__init__()
        self.layer1 = nn.Linear(x_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, h_dim)
        self.layer4 = nn.Linear(h_dim, y_dim)


    def forward(self, x):
        x = F.relu(self.layer1(x) + x)
        x = F.relu(self.layer2(x) + x)
        x = F.relu(self.layer3(x) + x)
        x = self.layer4(x)
        return x


def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.normal_(module.weight,0,0.1)
        torch.nn.init.zeros_(module.bias)

model = DeepContinuor(Nwn+1, Nwn+1, 1)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.MSELoss()

# training 1 epoch
np.random.shuffle(dataset)

batch_size = 100
num_batch = (Ns*Nw)//batch_size
for i in range(num_batch):
    batch = dataset[batch_size*i:batch_size*(i+1)]

    model.train()
    inputs = (torch.Tensor(batch[:,:-1])).to(device).float()
    targets = (torch.Tensor(batch[:,-1])).to(device).float()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(targets, outputs)
    loss.backward()
    optimizer.step()

    if i%100 == 0: 
        model.eval()
        sigma_func, pi_func = generator.generate()
        beta = 400
        wn = np.arange(0, Nwn)*2*np.pi/beta
        w = np.sort(np.random.uniform(0, wmax, Nw))
        pi = pi_func(wn)
        sigma = sigma_func(w)

        pi_repeated = np.broadcast_to(pi, [Nw, Nwn])
        w_column = w[:,np.newaxis]
        sigma_column = sigma[:,np.newaxis]
        data = np.concatenate([pi_repeated, w_column], axis=1)

        inputs = (torch.Tensor(data)).to(device).float()
        targets = (torch.Tensor(sigma_column)).to(device).float()
        outputs = model(inputs)
        loss_exemple = criterion(targets, outputs)

        plt.clf()
        plt.suptitle(f"this loss :{loss_exemple}, batch loss:{loss}")
        plt.subplot(121)
        # plt.xlim(0,30)
        plt.plot(wn, pi, '.')
        plt.subplot(122)
        # plt.xlim(0,wmax)
        # plt.ylim(0,1)
        plt.plot(w, targets.detach().numpy(), '.')
        plt.plot(w, outputs.detach().numpy(), '.')
        plt.pause(0.1)
        model.train()

