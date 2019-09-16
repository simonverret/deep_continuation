from  torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Database():
    def __init__(self, csvInput, csvTarget, transform= None, numberData= 25000):
        """
        Build the data set structure
        Args:
            csv_target (string): path to the target data (A(omega))
            csv_input (string) : path to the input data (G(tau))
            transform (callable, optional): Optional transform to be applied
                on a sample (eg: add noise).
        """
        self.inputData = pd.read_csv(csvInput, header= None, nrows= numberData)
        self.targetData = pd.read_csv(csvTarget, header= None, nrows= numberData)
        self.transform = transform


    def getLoader(self):
        # transform to pytorch tensor
        data= torch.tensor(self.inputData.values).double()
        normalizationFactor= 1.0
        target= torch.tensor(self.targetData.values).double()/normalizationFactor
        # Dataset wrapping tensor
        return TensorDataset(data.to(device), target.to(device))

    def plotData(self, index):
        # accessing index th row of the input data
        nsize= self.inputData.values[index-1:index, :].size
        x= np.arange(nsize)
        y= np.zeros(nsize, dtype= np.float)
        for n in range(nsize):
            y[n]= self.inputData.values[index-1:index, n]

        plt.plot(x, y, label= 'Loaded from file!', linestyle= '--', marker= 'o', color= 'green')
        plt.xlabel('Bosonic Matsubara Freq. index (n)')
        plt.ylabel(r'$\Pi (\nu_n)$')
        plt.legend()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
