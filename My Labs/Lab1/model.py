# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:37:18 2018

@author: Oppai
"""

from torch import nn as nn

class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
           # print('----------------')
            #print(x.size())
           # print('----------------')
            return x.view(x.size(0), -1)

def createModel(arguments):
    

    print('-----MNIST MODEL DEFINITION-----')

    # 10-class prolblem
    nOutputs = 10

    # Input Dimensions !!!with the pytorch dataset the dimensions are 28x28

    # No of channels = 1 because of MNIST
    nfeats = 1
    width = 28
    height = 28
    nInputs = nfeats * width * height

    # No of hidden inputs (For MLP only)
    nHiddens = int(nInputs / 2)

    modelType = arguments.neural_network.upper()

    if modelType == 'LIN':

        model = nn.Sequential().cuda()
        model.add_module('Reshape', Reshape(nInputs))
        model.add_module('Linear', nn.Linear(nInputs, nOutputs))
        model = model.cuda()
        print(model)

    elif modelType == 'MLP':

        model = nn.Sequential()
        model.add_module('Reshape', Reshape(nInputs))
        model.add_module('Linear', nn.Linear(nInputs, nHiddens))
        model.add_module('RELU', nn.ReLU())
        model.add_module('NLinear',nn.Linear(nHiddens, nOutputs))  # Solved problem, had 2 linear modules with the same name LMAO
        model = model.cuda()
        print(model)
        print('************************************\n MODEL IS CUDA:' + str(next(model.parameters()).is_cuda))

    else:
        raise ValueError(str(modelType) + ' Model does not exist')

    return model
