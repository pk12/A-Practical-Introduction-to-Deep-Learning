# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:37:13 2018

@author: Oppai
"""




from datetime import datetime
from torch import nn as nn
import torch as torch
import torchvision
import matplotlib.pyplot as plt

def test(model, criterion, testData):
    start = datetime.now()
    torch.cuda.empty_cache()
#    datasetPath = r"C:\Users\Oppai\Downloads\Πτυχιακη\Lab2A\Datasets"
#    
#    params = {'shuffle': True,
#              'pin_memory': True,
#              'num_workers': 0}  
#
#    Ttransform = torchvision.transforms.Compose(
#        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307),(0.3081, 0.3081, 0.3081))])  # Needs the 0.5, not 0,5 to make the object iterable !! Try with 0.5, Normalization to see the results
#    
#    testData = torch.utils.data.DataLoader(
#        torchvision.datasets.STL10(datasetPath, download=True, transform=Ttransform, split='test'), **params,
#        batch_size=64) 
    # load trained model
    model.load_state_dict(torch.load('model_MLP_NLL_ADAM.pt'))
    criterion.load_state_dict(torch.load('optimizer_MLP_NLL_ADAM.pt'))
    
    #    Set the model to evaluation mode
    model.eval()
    model.cuda()
    
    criterion.eval()
    criterion.cuda()

    correct = 0
    total = 0
    print(model)
    print(criterion)
    for index, (data, target) in enumerate(testData):
        #        Transfer to gpu
        data, target = data.cuda(), target.cuda()

        out, tsneData = model(data)
        loss = criterion(out, target)
        predLabel = out.data.max(1, keepdim=True)[1]

        total += data.size()[0]
        correct += predLabel.eq(target.data.view_as(predLabel)).sum()
        plt.scatter(tsneData[:,0] , tsneData[:,1])
        
#        print(str(predLabel) + '---> ' + str(target.data.view_as(predLabel)) +
#              '(' + str(loss.data) + ')\n')

    print(str(correct) + '\n' + '---------------')
    print("Features for")
    plt.show()
    

