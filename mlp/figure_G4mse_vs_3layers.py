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
y_name = 'G1bse_mse'
x_name = 'layers'
flag = '3'
pdfname = 'figure_G4mse_vs_3layers_3.pdf'

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
        elif treat_field(row[x_idx], x_name, 'depth')==3:
            c = ['G1','G2','G3','G4'].index(row[color_idx])
            x_list[c].append(treat_field(row[x_idx], x_name, flag))
            y_list[c].append(treat_field(row[y_idx], y_name))


plt.figure(figsize=(3, 4))

plt.plot(x_list[0], y_list[0], 'o', color='blue', markersize=3, label='trained on data 1')
plt.plot(x_list[1], y_list[1], 'o', color='blue', markersize=3, label='trained on data 1')
plt.plot(x_list[2], y_list[2], 'o', color='blue', markersize=3, label='trained on data 1')
plt.plot(x_list[3], y_list[3], 'o', color='blue', markersize=3, label='trained on data 1')

# plt.plot(x_list[1], y_list[1], '^', markersize=5, label='trained on data 2')
# plt.plot(x_list[2], y_list[2], 'v', markersize=5, label='trained on data 3')
# plt.plot(x_list[3], y_list[3], 'D', markersize=4, label='trained on data 4')

plt.xlabel('hidden size')
plt.ylabel('MSE')
# plt.xlim(0,0.04)
plt.ylim(0,0.02)
# plt.xticks([0,.01,.02,.03,.04])
# plt.yticks([0,.03,.06,.09,.12])
# plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(pdfname)
# plt.show()