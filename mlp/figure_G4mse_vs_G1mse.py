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
from matplotlib import rc

from plot_hyper import treat_field

# rc('text', usetex=True)
rc('axes', linewidth=0.5)
rc('xtick.major', width=0.5)
rc('ytick.major', width=0.5)
rc('lines', markeredgewidth=0)
# plt.rc('font', family='Helvetica')

filename = 'all_best.csv'
y_name = 'G4bse_mse'
x_name = 'G1bse_mse'
flag = ''
pdfname = 'figure_G4mse_vs_G1mse_04.pdf'

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

plt.figure(figsize=(6, 4))

plt.plot(x_list[2], y_list[2], 'v', markersize=5, color='C2', label='trained on $\omega=0$ peak only')
plt.plot(x_list[1], y_list[1], '^', markersize=5, color='C1', label='trained without $\omega=0$ peak')
plt.plot(x_list[3], y_list[3], 'D', markersize=4, color='C3', label='trained on simple data')
plt.plot(x_list[0], y_list[0], 'o', markersize=4, color='C0', label='trained on complex data')

plt.xlabel('Error on complex data')
plt.ylabel('Error on simple data')
plt.xlim(0,0.03)
plt.ylim(0,0.02)
plt.xticks([0,.01,.02,.03])
plt.yticks([0,.01,.02])
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(pdfname)
# plt.show()