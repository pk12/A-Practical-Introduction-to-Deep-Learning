# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:11:57 2018

@author: Oppai
"""

import torch
from torch.utils import data
from torch.backends import cudnn
import torchvision
import argparse


def main():
    # Create Parser
    parser = argparse.ArgumentParser(description='Pytorch MNIST LAB1')
    parser.add_argument("--batch-Size", type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--neural-network', type=str, default='MLP', metavar='M',
                        help='NN type (MLP | Linear)')
    parser.add_argument('--loss', type=str, default='NLL', metavar='M',
                        help='NN type (Margin | NLL)')
    parser.add_argument('--optim-Method', type=str, default='ASGD', metavar='M',
                        help='NN type (SGD | ASGD | LBFGS |)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='LR',
                        help='weight decay (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--max-iter', type=float, default=2, metavar='M',
                        help='CG and LBFGS max iterations (default: 2)')
    parser.add_argument('--t0', type=float, default=1, metavar='LR',
                        help='t0 for ASGD (default: 1)')

    arguments = parser.parse_args()
    print(arguments)
    #   Get Batch Size

    # USE CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0")
    cudnn.benchmark = True
    # device = "cpu"

    # Parameters
    params = {'shuffle': True,
              'pin_memory': True,
              'num_workers': 4}  # Exception because of bad multiworker handling !! FIXED IN NEW PYTORCH VERSION

    # Download dataset
    # torchvision.transforms.Normalize(0.5,0.5)
    datasetPath = r"..\Datasets"
    Ttransform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])  # Needs the 0.5, not 0,5 to make the object iterable !! Try with 0.5, Normalization to see the results
    testData = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(datasetPath, download=True, transform=Ttransform, train=False), **params,
        batch_size=arguments.test_batch_size)
    trainData = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(datasetPath, download=True, transform=Ttransform, train=True), **params,
        batch_size=arguments.batch_Size)

#     Start creating the model
    from loss import createLoss
    from model import createModel
    from train import train
    from test import test
    
    model = createModel(arguments)
    criterion = createLoss(arguments)
    train(arguments, trainData, device, criterion,model)
    test(model, criterion, testData)



if __name__ == '__main__':
    main()
