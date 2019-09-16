import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ANN0 import ANN0
from ANN import ANN
from Database import Database
from torch.utils.data import DataLoader
import torch
import json

def readJson(fileName):
    """  Read json file."""
    jsonFile= open(fileName, 'r')
    values= json.load(jsonFile)
    jsonFile.close()
    return values

def readDataFile():
    """ Reading data file """
    fileHandle = open("../Database_Gaussian_beta20/Validation/SigmaRe_sample.csv", "r")     # open File
    #   Blank lines and lines started with a number sign (#) in the data file will be ignored
    dataIn = np.loadtxt(fileHandle, unpack=True, usecols=[0], delimiter= ",")
    fileHandle.close()
    return np.transpose(dataIn)
    
def predict(model, testData):
    """ """
    model.eval()
    with torch.no_grad():
        prediction= model.forward(testData)
    return prediction.detach().cpu().numpy()

if __name__ == "__main__":
    """ Loading an artificial neural network and use it for a prediction"""
    # reading the network architecture
    fileName= 'Input.json'
    values= readJson(fileName)
    #
    inputSize= values['neuralNetwork']['architecture'][0]['inputSize']
    outputSize= values['neuralNetwork']['architecture'][0]['outputSize']
    hiddenLayers= values['neuralNetwork']['architecture'][0]['hiddenLayers']
    #
    normalization= values['neuralNetwork']['architecture'][1]['normalization']
    dropout= values['neuralNetwork']['architecture'][1]['dropout']
    #
    numberDataValidation= values['neuralNetwork']['architecture'][3]['numberDataValidation']
    #
    dropoutProbability= values['neuralNetwork']['hyperParameter']['dropoutProbability']
    if (normalization and not dropout):
        dropout= True
        dropoutProbability= 0.
    #
    if (normalization and dropout):
        model= ANN(inputSize, outputSize, hiddenLayers, dropoutProbability).double()
    else:
        model= ANN0(inputSize, outputSize, hiddenLayers).double()
    #
    PATH= 'ANNACCond_beta20.pth'
    model.load_state_dict(torch.load(PATH))
    # load the data
    x= readDataFile()
    inputData = pd.read_csv("../Database_Gaussian_beta20/Validation/Pi.csv", header= None, nrows= numberDataValidation)
    targetData = pd.read_csv("../Database_Gaussian_beta20/Validation/SigmaRe.csv", header= None, nrows= numberDataValidation)
    #
    data= torch.tensor(inputData.values).double()
    target= torch.tensor(targetData.values).double()
    #
    y= predict(model, data)
    #
    ## PLOT RANDOM DATA
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[5,8])
    ax3.get_shared_x_axes().join(ax2, ax1)
    ax2.set_xticklabels([])
    ax1.set_xticklabels([])
    ax1.set_xlim([0, 16])
    ax2.set_xlim([0, 16])
    ax3.set_xlim([0, 16])

    start= np.random.randint(0,numberDataValidation-10)
    end= start+5
    for ii in range(start,end):
        t = target.detach().cpu().numpy()
        ax1.plot(x, t[ii][:])
        #ax1.set_title('target',loc='right', pad=-15)
        ax1.legend(['target'])
        ax1.set_ylabel(r'$\sigma(\omega)$')

        ax2.plot(x, y[ii][:])
        #ax2.set_title('NN prediction',loc='right', pad=-15)
        ax2.legend(['nn prediction'])
        ax2.set_ylabel(r'$\sigma(\omega)$')
        
        e = t-y
        ax3.plot(x, e[ii][:])
        #ax3.set_title('error',loc='right', pad=-15)
        ax3.legend(['error'])
        ax3.set_xlabel(r'$\omega$')

    plt.savefig('results.pdf')

