import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ANN(nn.Module):
    # class constructor
    def __init__(self, inputSize, outputSize, hiddenLayers, dropoutProbability= 0.05):
        """  Builds ANN network with arbitrary number of hidden layers.

        Arguments
        ----------
        inputSize : integer, size of the input
        outputSize : integer, size of the output layer
        hiddenLayers: list of integers, the sizes of the hidden layers
        dropProbability: float in (0,1) , value of the dropout probability 
        """
        super(ANN, self).__init__()
        # Add the first layer : inputSize into the first hidden layer
        self.layers= nn.ModuleList([nn.Linear(inputSize, hiddenLayers[0]).to(device)])
        self.normalizations = nn.ModuleList([nn.BatchNorm1d(inputSize).to(device)])

        # Add the other layers
        layersSizes= zip(hiddenLayers[:-1], hiddenLayers[1:])
        self.layers.extend([nn.Linear(h1,h2).to(device) for h1, h2 in layersSizes])
        self.normalizations.extend([nn.BatchNorm1d(size).to(device) for size in hiddenLayers])

        self.output= nn.Linear(hiddenLayers[-1], outputSize).to(device)
        self.dropout= nn.Dropout(dropoutProbability).to(device)
    
    # class memeber function
    def forward(self, x):
        # pass through each layers
        for layer, normalization in zip(self.layers, self.normalizations):
            x= normalization(x)
            x= F.relu(layer(x))
            x= self.dropout(x)
        x= self.output(x)
        return x
#F.softmax(x, dim= 1)
