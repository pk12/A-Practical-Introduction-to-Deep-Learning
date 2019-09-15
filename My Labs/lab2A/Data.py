import torch
from torch.utils import data
from torch.backends import cudnn
import torchvision
import argparse

#     Start creating the model
from loss import createLoss
from model import createModel, Net
from train import train
from test import test

def main():
    # Create Parser
    parser = argparse.ArgumentParser(description='Pytorch MNIST LAB1')
    parser.add_argument("--batch-Size", type=int, default=160, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--neural-network', type=str, default='MLP', metavar='M',
                        help='NN type (MLP | Linear)')
    parser.add_argument('--loss', type=str, default='NLL', metavar='M',
                        help='NN type (Margin | NLL)')
    parser.add_argument('--optim-Method', type=str, default='ADAM', metavar='M',
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
    parser.add_argument('--mode', type=str, default='Test', metavar='LR',
                        help='t0 for ASGD (default: Train)')

    arguments = parser.parse_args()
    #   Get Batch Size

    # USE CUDA
    use_cuda = torch.cuda.is_available()
    torch.cuda.empty_cache();
    torch.cuda.init();
    device = torch.device('cuda')
    cudnn.benchmark = True
    # device = "cpu"

    # Parameters
    params = {'shuffle': True,
              'pin_memory': True,
              'num_workers': 0}  # Exception because of bad multiworker handling !! FIXED IN NEW PYTORCH VERSION

    # Download dataset
    datasetPath = r"..\Datasets"
    
    Ttransform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307, 0.1307, 0.1307),(0.3081, 0.3081, 0.3081))])  # Needs the 0.5, not 0,5 to make the object iterable !! Try with 0.5, Normalization to see the results
    
    testData = torch.utils.data.DataLoader(
        torchvision.datasets.STL10(datasetPath, download=True, transform=Ttransform, split='test'), **params,
        batch_size=arguments.test_batch_size) #Had problem with the Transformation, I was using Target_Transform instead of simple transform AND ToTensor() should be called before normalize
    
    trainData = torch.utils.data.DataLoader(
        torchvision.datasets.STL10(datasetPath, download=True, transform=Ttransform, split='train'), **params,
        batch_size=arguments.batch_Size)
    
    
    if (arguments.mode == 'Teest'):
        criterion = createLoss(arguments)
        modelClass = Net()
        test(modelClass ,criterion, testData)
        
    else:
        model = createModel(arguments)
        criterion = createLoss(arguments)
        train(arguments, trainData, device, criterion,model)


if __name__ == '__main__':
    main()
