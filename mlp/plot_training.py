import matplotlib.pyplot as plt

import sys
import csv
import matplotlib.pyplot as plt

try: filename = sys.argv[1]
except: raise ValueError('provide the filename as first argument')
try: y_name = ['train_loss']+sys.argv[2:]
except: y_name = ['train_loss']

x_name = 'epoch'
x_list = []
y_list_of_list = []
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    header = True
    for row in csv_reader:
        if header:
            print([name for name in row])
            x_idx = row.index(x_name)
            y_idx = []
            for i, name in enumerate(y_name):
                y_idx.append(row.index(name))
                y_list_of_list.append([])
            header = False
        else:
            x_list.append(float(row[x_idx]))
            for i, idx in enumerate(y_idx):
                y_list_of_list[i].append(float(row[idx]))

for i in range(len(y_idx)):
    plt.plot(x_list, y_list_of_list[i], '-', label=y_name[i])
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.legend()
plt.show()
