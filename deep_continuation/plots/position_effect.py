#%%
import numpy as np
from scipy import integrate
from scipy.special import binom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
COLORS = list(mcolors.TABLEAU_COLORS)

from deep_continuation.data_generator import *




N_wn = 1000
beta = 500
X = np.linspace(-10,10,1000)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[15,5], dpi=80)
ax1.set_xlabel(r"$\omega$")
ax2.set_xlabel(r"$\omega_n$")
ax3.set_xlabel(r"$n$")
for pos in [0,1,2,3,4,5]:
    def sigma(x):
        return np.pi*gaussian(x, pos, 1, 1)+np.pi*gaussian(x, -pos, 1, 1)
    S = sigma(X)
    W = (2*np.pi/beta) * np.arange(0,N_wn)
    P = pi_integral(W, sigma)
    ax1.plot(X, S)
    ax2.plot(W, P)
    ax3.plot(P, '.')

plt.show()
# plt.savefig("scale.pdf")

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[15,5], dpi=80)
ax1.set_xlabel(r"$\omega$")
ax2.set_xlabel(r"$\omega_n$")
ax3.set_xlabel(r"$n$")
for width in [0.1,0.2,0.4,0.8,1.6,3.1]:
    def sigma(x):
        return np.pi*gaussian(x, 5, width, 1)+np.pi*gaussian(x, -5, width, 1)
    S = sigma(X)
    W = (2*np.pi/beta) * np.arange(0,N_wn)
    P = pi_integral(W, sigma)
    ax1.plot(X, S)
    ax2.plot(W, P)
    ax3.plot(P, '.')

plt.show()

