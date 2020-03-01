#!/usr/bin/env python
#
#   deep_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

#%%
import sys
import csv
import matplotlib.pyplot as plt
from plot_hyper import treat_field

filename = 'all_best.csv'
y_name = 'G4n10_mse'
x_name = 'noise'
flag = ''
pdfname = 'figure_G4mse_vs_noise10.pdf'

x_list = [[],[],[],[]]
y_list = [[],[],[],[]]

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    header = True
    for row in csv_reader:
        if header:
            x_idx = row.index(x_name)
            y_idx = row.index(y_name)
            color_idx = row.index('data')
            header = False
        else:
            c = ['G1','G2','G3','G4'].index(row[color_idx])
            x_list[c].append(treat_field(row[x_idx], x_name, flag))
            y_list[c].append(treat_field(row[y_idx], y_name))

plt.figure(figsize=(4, 4))

plt.plot(x_list[0], y_list[0], 'o', markersize=4, label='trained on data 1')
plt.plot(x_list[1], y_list[1], '^', markersize=5, label='trained on data 2')
plt.plot(x_list[2], y_list[2], 'v', markersize=5, label='trained on data 3')
plt.plot(x_list[3], y_list[3], 'D', markersize=4, label='trained on data 4')

plt.xlabel('Noise $\sigma$ during training')
plt.ylabel('MSE on data 4 with $\sigma=0.1$ noise')

plt.axvline(x=0.10, color='black')
plt.xlim(-0.001,0.2)
plt.ylim(0,0.02)

# plt.xticks([0,.01,.02,.03])
# plt.yticks([0,.005,.01])
# plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(pdfname)
plt.show()