from ANN0 import ANN0
from ANN import ANN
from torch.nn.modules.loss import KLDivLoss, L1Loss, SmoothL1Loss
from torch.optim import Adam, Adadelta, Rprop, Adamax, RMSprop, SGD, LBFGS
import torch

def train(model, trainLoader, validationLoader, error, optimizer, epoch, logInterval):
    """ """
    model.train()
    #  Load a minibatch
    for batchIndex, (data, target) in enumerate(trainLoader):
        # restart the optimizer
        optimizer.zero_grad()
        # predict
        prediction = model.forward(data)
        # compute the loss
        loss = error(prediction, target)
        # Compute the gradient and optimize
        loss.backward()
        optimizer.step()
        #
        if batchIndex % logInterval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}'.format(epoch, batchIndex * len(data), len(trainLoader.dataset),
                100. * batchIndex / len(trainLoader), loss.item()),
                  ', Validation Loss: {:.6f}'.format(validationScore(model, validationLoader, error)))


def validationScore(model, validationLoader, error):
    """ """
    model.eval()
    with torch.no_grad():
        data, target= next(iter(validationLoader))
        prediction= model.forward(data)
        score= error(prediction, target)
    model.train()
    return score.item()

