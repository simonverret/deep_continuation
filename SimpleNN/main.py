#!/usr/bin/env python
from ANN0 import ANN0
from ANN import ANN
from train import train
from Database import Database
from torch.nn.modules.loss import KLDivLoss, L1Loss, SmoothL1Loss, MSELoss
from torch.optim import Adam, Adadelta, Rprop, Adamax, RMSprop, SGD, LBFGS
from torch.utils.data import DataLoader
import torch
import json

def readJson(fileName):
    """  Read json file."""
    jsonFile= open(fileName, 'r')
    values= json.load(jsonFile)
    jsonFile.close()
    return values
#


if __name__ == "__main__":
    """ Constructing an artificial neural network for analytic continuation of conductivity"""
    # reading input
    fileName= 'Input.json'
    values= readJson(fileName)
    #
    inputSize= values['neuralNetwork']['architecture'][0]['inputSize']
    outputSize= values['neuralNetwork']['architecture'][0]['outputSize']
    hiddenLayers= values['neuralNetwork']['architecture'][0]['hiddenLayers']
    loadNN= values['neuralNetwork']['architecture'][0]['loadNN']
    #
    activationFunction= values['neuralNetwork']['architecture'][1]['activationFunction']
    normalization= values['neuralNetwork']['architecture'][1]['normalization']
    dropout= values['neuralNetwork']['architecture'][1]['dropout']
    #
    lossFunction= values['neuralNetwork']['architecture'][2]['lossFunction']
    optimizerAlgorithm= values['neuralNetwork']['architecture'][2]['optimizerAlgorithm']
    #
    numberDataTraining= values['neuralNetwork']['architecture'][3]['numberDataTraining']
    numberDataValidation= values['neuralNetwork']['architecture'][3]['numberDataValidation']
    #
    batchSizeTraining= values['neuralNetwork']['hyperParameter']['batchSizeTraining']
    batchSizeValidation= values['neuralNetwork']['hyperParameter']['batchSizeValidation']
    numberEpochs= values['neuralNetwork']['hyperParameter']['numberEpochs']
    learningRate= values['neuralNetwork']['hyperParameter']['learningRate']
    optimizerMomentum= values['neuralNetwork']['hyperParameter']['optimizerMomentum']
    dropoutProbability= values['neuralNetwork']['hyperParameter']['dropoutProbability']
    if (normalization and not dropout):
        dropout= True
        dropoutProbability= 0.
    #
    logInterval= values['neuralNetwork']['printing']['logInterval']
    if (logInterval > int(numberDataTraining/batchSizeTraining)):
        logInterval= int(numberDataTraining/batchSizeTraining/2)
        print("\n log interval is changed to: ", logInterval)
    #
    print ("\n Input File Name: %s "%"Input.json")
    #
    # Import the data
    print(" Loading data")
    trainData= Database(csvInput= "../Database_Gaussian_beta20/Training/Pi.csv", csvTarget= "../Database_Gaussian_beta20/Training/SigmaRe.csv", numberData= numberDataTraining).getLoader()
#    validationData= Database(csvInput= "../Database_Gaussian_beta10/Validation/Pi.csv", csvTarget= "../Database_beta10/Validation/SigmaRe.csv", numberData= numberDataValidation).getLoader()
    VData= Database(csvInput= "../Database_Gaussian_beta20/Validation/Pi.csv", csvTarget= "../Database_Gaussian_beta20/Validation/SigmaRe.csv", numberData= numberDataValidation)
    validationData= VData.getLoader()
    # plot a sample data from validation
#index= 1
#VData.plotData( index)
    # Create custom random sampler class to iter over dataloader
    trainLoader= DataLoader(trainData, batch_size= batchSizeTraining, shuffle= True)
    validationLoader= DataLoader(validationData, batch_size= batchSizeValidation)
    #
    # Initialize model
    print(" Initialize model")
    if (normalization and dropout):
        model= ANN(inputSize, outputSize, hiddenLayers, dropoutProbability).double()
    else:
        model= ANN0(inputSize, outputSize, hiddenLayers).double()
    #
    # load neural network
    if loadNN:
        PATH= 'ANNACCond_beta20.pth'
        model.load_state_dict(torch.load(PATH))
    # Define the loss
    if lossFunction== "L1Loss":
        error= L1Loss()
    elif lossFunction== "KLDivLoss":
        error= KLDivLoss()
    elif lossFunction== "SmoothL1Loss":
        error= SmoothL1Loss()
    elif lossFunction== "MSELoss":
        error= MSELoss()
    else:
        print ('Undefined loss function!')
        sys.exit()
    #
    # Initialize optimizer
    # it is said that SGD+Nesterov can be as good as Adam’s technique
    if optimizerAlgorithm== "Adam":
        optimizer= Adam(model.parameters(), lr= learningRate)
    elif optimizerAlgorithm== "Adadelta":
        optimizer= Adadelta(model.parameters(), lr= learningRate)
    elif optimizerAlgorithm== "SGD":
        optimizer= SGD(model.parameters(), lr= learningRate, momentum= optimizerMomentum, nesterov= True)
    else:
        print ('Undefined optimizer algorithm!')
        sys.exit()
    #
    # Training
    for epoch in range(numberEpochs):
        train(model, trainLoader, validationLoader, error, optimizer, epoch, logInterval)
    #
    # Print model's state_dict
    torch.save(model.state_dict(),'ANNACCond_beta20.pth')


