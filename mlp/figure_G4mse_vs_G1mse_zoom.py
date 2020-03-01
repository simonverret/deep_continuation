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
y_name = 'G4bse_mse'
x_name = 'G1bse_mse'
flag = ''
pdfname = 'figure_G4mse_vs_G1mse_zoom.pdf'

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


plt.figure(figsize=(3, 3))

plt.plot(x_list[0], y_list[0], 'o', markersize=4, label='trained on data 1')
plt.plot(x_list[1], y_list[1], '^', markersize=5, label='trained on data 2')
plt.plot(x_list[2], y_list[2], 'v', markersize=5, label='trained on data 3')
plt.plot(x_list[3], y_list[3], 'D', markersize=4, label='trained on data 4')


plt.xlabel('MSE on data 1')
plt.ylabel('MSE on data 4')
plt.xlim(0,0.008)
plt.ylim(0,0.0004)
plt.xticks([0,.004,.008])
plt.yticks([0,.0001,.0002,.0003,.0004])
plt.tight_layout()

plt.savefig(pdfname)
# plt.show()