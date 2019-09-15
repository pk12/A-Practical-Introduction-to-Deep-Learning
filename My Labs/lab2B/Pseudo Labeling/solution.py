
# coding: utf-8

# In[1]:


from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys

np.set_printoptions(threshold=sys.maxsize)


# In[2]:


from sub import subMNIST       # testing the subclass of MNIST dataset


# # Split Data

# In[4]:

#Load Numpy Arrays
#train_data_sub = torch.from_numpy(np.load("trainset_np_labeledData.npy"))
#train_labels_sub = torch.from_numpy(np.load("trainset_label_labeledLabels.npy"))
#valid_data_sub = torch.from_numpy(np.load("validset_np.npy"))
#valid_labels_sub = torch.from_numpy(np.load("validset_label_np.npy"))
#train_data_sub_unl = torch.from_numpy(np.load("trainset_np_unlData.npy"))
#train_labels_sub_unlArray = np.load("trainset_np_unlLabels.npy")
##
train_data_sub = None
train_labels_sub = None
valid_data_sub = None
valid_labels_sub = None
train_data_sub_unl = None
train_labels_sub_unlArray = None
train_labels_sub_unl = None

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                             ])


# In[4]:


trainset_original = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)


# In[5]:


train_label_index = []
valid_label_index = []

if (train_data_sub is None and train_labels_sub is None):
    

    # for each Number split the dataset into labels and valid train labels
    # use the label array to easily find the indexes
    
    for i in range(10):
        train_label_list = trainset_original.train_labels.numpy()
        label_index = np.where(train_label_list == i)[0]
        label_subindex = list(label_index[:300])
        valid_subindex = list(label_index[300: 1000 + 300])
        train_label_index += label_subindex
        valid_label_index += valid_subindex
    
    
    # ## Train Set
    
    # In[6]:
    
    
    trainset_np = trainset_original.train_data.numpy()
    trainset_label_np = trainset_original.train_labels.numpy()
    train_data_sub = torch.from_numpy(trainset_np[train_label_index])
    train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])
    
    #Save  Labeled
    np.save("trainset_np_labeledData.npy", trainset_np[train_label_index])
    np.save("trainset_label_labeledLabels.npy",trainset_label_np[train_label_index])
    # In[7]:

# Create a new DataLoader and clone the traindata objects into the new DataLoader (k=size=3000)
trainset_new = subMNIST(root='./data', train=True, download=True, transform=transform, k=3000)
trainset_new.train_data = train_data_sub.clone()
trainset_new.train_labels = train_labels_sub.clone()


# ## Validation Set


if (valid_data_sub is None and valid_labels_sub is None):
    
    validset_np = trainset_original.train_data.numpy()
    validset_label_np = trainset_original.train_labels.numpy()
    valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
    valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])
    
    np.save("validset_np.npy", validset_np[valid_label_index])
    np.save("validset_label_np.npy", validset_label_np[valid_label_index])



validset = subMNIST(root='./data', train=False, download=True, transform=transform, k=10000)
validset.test_data = valid_data_sub.clone()
validset.test_labels = valid_labels_sub.clone()



# ## Unlabeled Data
if (train_data_sub_unl is None and train_labels_sub_unl is None):

    train_unlabel_index = []
    for i in range(60000):
        if i in train_label_index or i in valid_label_index:
            pass
        else:
            train_unlabel_index.append(i)
    
    
    
    trainset_np = trainset_original.train_data.numpy()
    trainset_label_np = trainset_original.train_labels.numpy()
    train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
    
    #Fill labels with fake ones ---> -1
    nparr = np.asarray([-1] * trainset_np[train_unlabel_index].shape[0])
    
    np.save("trainset_np_unlData.npy", trainset_np[train_unlabel_index])
    np.save("trainset_np_unlLabels.npy", nparr)
    train_labels_sub_unlArray = nparr
    train_labels_sub_unl = torch.from_numpy(nparr)


train_labels_sub_unl = train_labels_sub_unl.type(torch.long)


trainset_new_unl = subMNIST(root='./data', train=True, download=True, transform=transform, k=47000)
trainset_new_unl.train_data = train_data_sub_unl.clone()
trainset_new_unl.train_labels = train_labels_sub_unl.clone()      # Unlabeled!!


print("Unlabeled Data " + str(trainset_new_unl.train_data.size()))


pickle.dump(trainset_new_unl, open("train_unlabeled.p", "wb" ))


# # Train Model


#train_loader = torch.utils.data.DataLoader(trainset_new, batch_size=64, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)


#Unify both data and label sets

#Create Full trainset
#Unite Labeled and Unlabeled Data Tensors
#trainset_full  = subMNIST(root='./data', train=True, transform=transform, k=50000)
#trainset_full.train_data = torch.cat((trainset_new_unl.train_data, trainset_new.train_data),0)
#trainset_full.train_labels = torch.cat((trainset_new_unl.train_labels, trainset_new.train_labels),0)
#print('Trainset train_data ' + str(trainset_full.train_data.size()))

train_loader = torch.utils.data.DataLoader(trainset_new, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.batchNorm4 = nn.BatchNorm2d(128)
        self.batchNorm5 = nn.BatchNorm1d(100)
        self.batchNorm6 = nn.BatchNorm1d(10)
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.leaky_relu(self.batchNorm1(F.max_pool2d(self.conv1(x), 2)))
        x = F.leaky_relu(self.batchNorm2(self.conv2(x)))
        x = F.leaky_relu(self.batchNorm3(F.max_pool2d(self.conv3(x), 2)))
        x = F.leaky_relu(self.batchNorm4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.dropout2d(F.relu(self.batchNorm5(self.fc1(x))), p=0.5)
        x = self.batchNorm6(self.fc2(x))
        return F.log_softmax(x)
    
nfeats = 1
width = 28
height = 28
nInputs = nfeats * width * height    
nHiddens = int(nInputs / 2)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(nInputs, nHiddens)
        self.fc2 = nn.Linear(nHiddens, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
model = model.cuda()

#optimizer = optim.Adam(model.parameters(), lr=0.01,betas=(0.9, 0.999))
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.5,weight_decay=0.001)

# CPU only training
def train(epoch, model, train_loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
#    criterion = nn.NLLLoss(ignore_index=-1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
            
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
#        if batch_idx % 10 == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(train_loader.dataset),
#                100. * batch_idx / len(train_loader), loss.data))

def test(epoch, valid_loader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = data.cuda(), target.cuda()
        data.volatile=True
        output = model(data)
        test_loss += criterion(output, target).data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nEpoch: ' + str(epoch) + 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    
def testAndLabel(epoch, trainedModel, train_loader2):
    trainedModel.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader2):
        data, target = data.cuda(), target.cuda()
        data.volatile=True
        output = model(data)
        
        #Get Max Value and Index
        maxVal = torch.exp(output).max().item()
        maxIndex = output.data.max(1)[1].item()
        
        if (maxVal >= 0.9):
            train_labels_sub_unlArray[batch_idx] = maxIndex
#            print("Label {} Updated at Index {} ---> Value {}".format(batch_idx,maxIndex, maxVal))
        if (maxIndex > 9):
            print ('Error at index {}'.format(batch_idx))
        
        
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(train_loader2) # loss function already averages over batch size
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(train_loader2.dataset),
#        100. * correct / len(train_loader2.dataset)))
    #Save the updated Dataset
    np.save("trainset_np_unlLabels.npy", train_labels_sub_unlArray)

print('Beginning training loops')
for epoch in range(1, 11):
    train(epoch, model, train_loader)
    test(epoch, valid_loader, model)
    
torch.save(model.state_dict(), 'model_' + '3000Trained' + '.pt')
trainedModel = SimpleNet()
trainedModel.load_state_dict(torch.load('model_' + '3000Trained' + '.pt'))
trainedModel = trainedModel.cuda()
    
def loadDataForLabeling():    
    train_loader2 = torch.utils.data.DataLoader(trainset_new_unl, batch_size=1, shuffle=False) #Careful must not be shuffled !!!WORKS LOSS IS REDUCED
    
    #Add unlabeled test set
    testAndLabel(1, trainedModel, train_loader2)
    
    #Train on full newly labeled set
    print('Beginning training loops phase 2')
    
    #Load new Labels
    train_labels_sub_unl = torch.from_numpy(np.load("trainset_np_unlLabels.npy"))
    train_labels_sub_unl = train_labels_sub_unl.type(torch.long)
    trainset_new_unl.train_labels = train_labels_sub_unl
    
    trainset_full  = subMNIST(root='./data', train=True, transform=transform, k=50000)
    trainset_full.train_data = torch.cat((trainset_new_unl.train_data, trainset_new.train_data),0)
    trainset_full.train_labels = torch.cat((trainset_new_unl.train_labels, trainset_new.train_labels),0)
    
    train_loader3 = torch.utils.data.DataLoader(trainset_full,batch_size=64, shuffle=False)
    print('Trainset train_data ' + str(trainset_full.train_data.size()))
    return train_loader3

for cycle in range(1,6):
    train_loader3 = loadDataForLabeling()
    
    for epoch in range(1, 6):
        train(epoch, trainedModel, train_loader3)
        test(epoch, valid_loader, trainedModel)



