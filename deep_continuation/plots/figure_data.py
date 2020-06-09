#%%
#
#   deep_continuation
#
#   © Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

import numpy as np
import matplotlib.pyplot as plt
import data

noise = 0.0
pdf_name = 'dataset_noise00.pdf'

dataset1 = data.ContinuationData(f'data/G1/valid/', noise=noise)
dataset2 = data.ContinuationData(f'data/G2/valid/', noise=noise)
dataset3 = data.ContinuationData(f'data/G3/valid/', noise=noise)
dataset4 = data.ContinuationData(f'data/G4/valid/', noise=noise)

fig, ax = plt.subplots(4, 2, figsize=[6,5])

ax[0,0].get_shared_x_axes().join(ax[1,0], ax[2,0], ax[3,0])
ax[0,0].get_shared_y_axes().join(ax[1,0], ax[2,0], ax[3,0])
ax[0,0].set_xticklabels([])
ax[1,0].set_xticklabels([])
ax[2,0].set_xticklabels([])
ax[3,0].set_xticks([0,32,64,96,128])
ax[3,0].set_yticks([0,.5,1])
ax[3,0].set_xlabel('$i\omega_n$')
[ax[i,0].set_ylabel('$\Pi$') for i in range(4)]
[ax[i,0].set_xlim(0,64) for i in range(4)]
# [ax[i,0].set_ylim(-0,1) for i in range(4)]

ax[0,1].get_shared_x_axes().join(ax[1,1], ax[2,1], ax[3,1])
ax[0,1].set_xticklabels([])
ax[1,1].set_xticklabels([])
ax[2,1].set_xticklabels([])
ax[3,1].set_xticks([0,2,4,6,8,10])
ax[3,1].set_xlabel('$\omega$')
[ax[i,1].set_ylabel('$\sigma$') for i in range(4)]


start=90
end=start+5
for ii in range(start,end):
    y1 = dataset1[ii][0]
    y2 = dataset2[ii][0]
    y3 = dataset3[ii][0]
    y4 = dataset4[ii][0]

    ax[0,0].plot(y1)
    ax[1,0].plot(y2)
    ax[2,0].plot(y3)
    ax[3,0].plot(y4)

    x1 = dataset1[ii][1]
    x2 = dataset2[ii][1]
    x3 = dataset3[ii][1]
    x4 = dataset4[ii][1]

    freq = np.linspace(0,10,512)
    ax[0,1].plot(freq,x1)
    ax[1,1].plot(freq,x2)
    ax[2,1].plot(freq,x3)
    ax[3,1].plot(freq,x4)

fig.tight_layout()
plt.savefig(pdf_name)
