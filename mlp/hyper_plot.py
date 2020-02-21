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

'''
note: to cat all 
'''

filename = sys.argv[1]
y_name = sys.argv[2]
x_name = sys.argv[3]
try: flag = sys.argv[4]
except IndexError: flag = None

def treat_field(raw, name, flag=None):
    ''' Use `name` to decide how to use the `raw` data strig. 
    ex: `name='layers'` will enable to use the string at a list and `flag` will
    be the index in the list or `sum` to sum all elements, or `depth` to yield
    the lenght of the list.
    '''
    if name in ['batchnorm','out_unit','loss','warmup', 'schedule']: 
        data = raw
    elif name == 'layers':
        layers = raw.strip('[]').split(',')
        n = len(layers)-2
        if flag == 'depth':
            data = n
        elif flag == 'sum':
            data = sum([int(x) for x in layers[1:-1] ])
        elif abs(int(flag)) <= n: 
            if int(flag)<0: i = int(flag)-1
            else: i = int(flag)
            data = int(layers[i])
        else: 
            data = None
    else:
        data = float(raw)

    return data

x_list = []
y_list = []

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    header = True
    for row in csv_reader:
        if header:
            print([name for name in row])
            x_idx = row.index(x_name)
            y_idx = row.index(y_name)
            header = False
        else:
            x_list.append(treat_field(row[x_idx], x_name, flag))
            y_list.append(treat_field(row[y_idx], y_name))

plt.plot(x_list, y_list, '.')
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.show()