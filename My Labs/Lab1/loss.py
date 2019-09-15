# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:37:34 2018

@author: Oppai
"""

import torch.nn as nn


def createLoss(arguments):
    model = nn.Sequential()
    # No of outputs/classes
    nOutputs = 10

    typeofLoss = arguments.loss.upper()

    if typeofLoss == 'MARGIN':
        #   -- This loss takes a vector of classes, and the index of
        #   -- the grountruth class as arguments. It is an SVM-like loss
        #   -- with a default margin of 1.
        criterion = nn.MultiMarginLoss()
        print(criterion)
    elif typeofLoss == 'NLL':
        #   -- This loss requires the outputs of the trainable model to
        #   -- be properly normalized log-probabilities, which can be
        #   -- achieved using a softmax function
        #    model.add_module('LogSoftmax', nn.LogSoftmax())

        #   -- The loss works like the MultiMarginCriterion: it takes
        #   -- a vector of classes, and the index of the grountruth class
        #   -- as arguments.
        #    criterion = nn.NLLLoss() #May be wrong module
        criterion = nn.CrossEntropyLoss()
        print(criterion)

    elif typeofLoss == 'MSE':  # !!!NOT RECOMMENDED
        #    -- for MSE, we add a tanh, to restrict the model's output
        model.add_module('Tanh', nn.Tanh())

        #   -- The mean-square error is not recommended for classification
        #   -- tasks, as it typically tries to do too much, by exactly modeling
        #   -- the 1-of-N distribution. For the sake of showing more examples,
        #   -- we still provide it here:
        criterion = nn.MSELoss()
        criterion.sizeAverage = False
        print(criterion)

    #   -- Compared to the other losses, the MSE criterion needs a distribution
    #   -- as a target, instead of an index. Indeed, it is a regression loss!
    #   -- So we need to transform the entire label vectors:
    #    if trainData:
    #        #Convert training labels:
    #        trSize = trainData.train_labels
    else:
        raise ValueError(typeofLoss + ' Uknown Loss type')

    return criterion
