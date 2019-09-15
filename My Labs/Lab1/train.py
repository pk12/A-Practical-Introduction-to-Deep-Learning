# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:02:32 2018

@author: Oppai
"""

import logging
from datetime import datetime

import torch
import torch.optim as optim


def train(arguments,trainData,device,criterion,model):
    # Set default logging format
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    def setup_logger(name, log_file, level=logging.INFO):
        #    """Function setup as many loggers as you want"""

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    print('Defining some tools')

    print('************************************\n MODEL IS CUDA:' + str(next(model.parameters()).is_cuda))

    # -- Retrieve parameters and gradients:
    # -- this extracts and flattens all the trainable parameters of the mode
    # -- into a 1-dim vector
    # Not needed, added to the optimizer at once
    # if model is not None:
    #    oldparameters,oldgradParameters = model.parameters()

    optimizer = arguments.optim_Method.upper()

    print('Configuring Optimizer')

    if optimizer == 'CG':  # No CG model in torch
        #    Insert Values
        maxIter = arguments.max_iter

        optimMethod = optim.Optimizer(model.parameters())

    elif optimizer == 'LBFGS':  # !!!NEEDS CLOSURE FUNCTION
        #    Insert Values

        maxIter = arguments.max_iter
        learningRate = arguments.lr

        optimMethod = optim.LBFGS(model.parameters(), lr=learningRate, max_iter=maxIter)

    elif optimizer == 'SGD':
        #    Insert Values
        weightDecay = arguments.weight_decay
        learningRate = arguments.lr
        momentum = arguments.momentum

        optimMethod = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum, weight_decay=weightDecay)

    elif optimizer == 'ASGD':
        learningRate = arguments.lr
        eta0 = arguments.t0

        optimMethod = optim.ASGD(model.parameters(), lr=learningRate, t0=eta0 * trainData.dataset.train_data.size()[0])

    else:
        raise ValueError('Uknown optimization method')

    print(model.parameters())

    # !!!!!START TRAINING!!!!
    #   Since train is called multiple times it is checked if it is loaded in the memory first
    #    !!!!WORKS LIKE THIS IN LUA, IN PYTHON WE WILL NEED ANOTHER WAY

    #    Set model to training mode
    model = model.train()
    print(model)

    #   do one epoch
    print('--->Doing epoch on training data')
    print("--->Online epoch # " + str(arguments.epochs) + "[batchSize = " + str(arguments.batch_Size) + "]")

    #   Begin Fetching batches from Dataloader
    #   Got this part from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    model.cuda().to(device)
    criterion.cuda().to(device)
    time = datetime.now()

    for i in range(arguments.epochs):
        #       Training
        for index, (data, target) in enumerate(trainData):

            #           Transfer to GPU
            data, target = data.cuda().to(device), target.cuda().to(device)

            #          Forward pass to the NN
            if optimizer == 'LBFGS':
                #               If optimizer needs eval function
                #               Το τρεχω σωστα??
                def closure():
                    optimMethod.zero_grad()
                    outputs = model.forward(data)
                    print(outputs.size())
                    loss = criterion(outputs, target)
                    loss.backward()
                    return loss

                loss = optimMethod.step(closure)
                print('Loss for batch ' + str(index) + ': ' + str(loss[0]))



            else:
                #               if optimizer does not need eval function
                outputs = model.forward(data)
                loss = criterion(outputs, target)

                #               BackProp and optimize
                optimMethod.zero_grad()
                loss.backward()
                optimMethod.step()  # Feval)
                print('Loss for batch ' + str(index) + ': ' + str(loss.data))
        
        #Time for each epoch        
        print(datetime.now() - time)

        #       Save current trained net
        torch.save(model.state_dict,
                   'model_' + arguments.neural_network + '_' + arguments.loss + '_' + optimizer + '.pt')  # load with model.load_state_dict and model.eval() to get the correct results
