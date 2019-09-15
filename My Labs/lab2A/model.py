# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:37:18 2018

@author: Oppai
"""

from torch import nn as nn
import torch as torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class Reshape(nn.Module):
        def __init__(self, *args):
            super(Reshape, self).__init__()
            self.shape = args

        def forward(self, x):
           # print('----------------')
            #print(x.size())
           # print('----------------')
            return x.view(x.size(0), -1)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Stage 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1).cuda()
        self.SpatialBatchNormalization1 = nn.BatchNorm2d(num_features=64 ,eps=1e-3).cuda()
        self.ReLU = nn.ReLU().cuda()
        self.maxPool1 = nn.MaxPool2d(kernel_size=2,ceil_mode=True).cuda()
        
        #Stage 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1).cuda()
        self.SpatialBatchNormalization2 = nn.BatchNorm2d(num_features=256, eps=1e-3).cuda()
        self.ReLU2 = nn.ReLU().cuda()
        self.maxPool2 = nn.MaxPool2d(kernel_size=2,ceil_mode=True).cuda()
        
        #Classification layers
        self.Reshape1 = Reshape().cuda()
        self.Dropout1 = nn.Dropout(p=0.5).cuda()
        self.Linear1 = nn.Linear(147456, 128).cuda()
        self.BatchNorm1 = nn.BatchNorm1d(128).cuda()
        self.Relu1 = nn.ReLU().cuda()
        self.Dropout2 = nn.Dropout(p=0.5).cuda()
        self.Linear2 = nn.Linear(128, 10).cuda()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.SpatialBatchNormalization1(x)
        x = self.ReLU(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.SpatialBatchNormalization2(x)
        x = self.ReLU2(x)
        
        x = self.maxPool2(x)
        x = self.Reshape1(x)
        x = self.Dropout1(x)
        x = self.Linear1(x)
        x = self.BatchNorm1(x)
        x = self.Relu1(x)
        x = self.Dropout2(x)
        x = self.Linear2(x)
        
        t_sne_model = TSNE(learning_rate=500)
        data = x.cpu().clone().detach();
        shape = data.shape
        tsne_results = t_sne_model.fit_transform(data)
        kmeans_data = KMeans(n_clusters=10).fit(tsne_results)
        return x, tsne_results, kmeans_data
        
def ConvBNReLU(model, nInputs, nOutputs, timeCalled):
    model.add_module('Conv2D' + str(timeCalled), nn.Conv2d(in_channels=nInputs, out_channels=nOutputs, kernel_size=3, stride=1, padding=1).cuda()) #--> nn SpatialConvolution
    model.add_module('SpatialBatchNormalization_' + str(timeCalled),nn.BatchNorm2d(num_features=nOutputs, eps=1e-3)) #--> nn SpatialBatchNormalization
    model.add_module('ReLU' + str(timeCalled), nn.ReLU())
    timeCalled = timeCalled + 1

def createModel(arguments):
    torch.cuda.empty_cache()

    print('-----STL10 MODEL DEFINITION-----')
    HEIGHT = 96
    WIDTH = 96
    DEPTH = 3
    
    Size = HEIGHT * WIDTH * DEPTH

    nHiddens = int(Size / 2)
    
    timeCalled = 1;

    model = Net();
    
#    ConvBNReLU(model, 3, 64,timeCalled)
#    model.add_module('MaxPool2d_1', nn.MaxPool2d(kernel_size=2,ceil_mode=True).cuda())
    
#    ConvBNReLU(model,64,128,timeCalled)
#    model.add_module('MaxPool2d_2', nn.MaxPool2d(kernel_size=3,ceil_mode=True).cuda())
#    
#    ConvBNReLU(model,128,256,timeCalled)
#    ConvBNReLU(model,256,256,timeCalled)
#    model.add_module('MaxPool2d_3', nn.MaxPool2d(kernel_size=3,ceil_mode=True))
#    
#    ConvBNReLU(model,256,256,timeCalled)
#    ConvBNReLU(model,256,256,timeCalled)
#    ConvBNReLU(model,256,256,timeCalled)
#    model.add_module('MaxPool2d_3', nn.MaxPool2d(kernel_size=3,ceil_mode=True))
    
#    model.add_module('Reshape', Reshape().cuda())
#    
#    classifier = nn.Sequential()
#    
#    classifier.add_module('Dropout', nn.Dropout(p=0.5).cuda())
#    classifier.add_module('Linear', nn.Linear(36864, 64).cuda())
#    classifier.add_module('BatchNorm', nn.BatchNorm1d(64).cuda())
#    classifier.add_module('Relu', nn.ReLU().cuda())
#    classifier.add_module('Dropout1', nn.Dropout(p=0.5).cuda())
#    classifier.add_module('Linear1', nn.Linear(64, 10).cuda())
#    
#    model.add_module('Classifier', classifier.cuda())
    model = model.cuda()
    
    
    print('************************************\n MODEL IS CUDA:' + str(next(model.parameters()).is_cuda))

    return model
