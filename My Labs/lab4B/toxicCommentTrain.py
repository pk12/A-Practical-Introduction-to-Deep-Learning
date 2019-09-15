# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:42:59 2019

@author: Oppai
"""
from torchtext import data
import torch
from torch import optim
from torch import nn
from BatchWrapper import BatchWrapper
from Model import SimpleLSTM
import tqdm

#Tokenizer function (No need for a complex one) 
tokenize = lambda x: x.split()

#Create the Fields
textField = data.Field(sequential=True, tokenize=tokenize, lower=True)
labelField = data.LabelField(sequential=False, use_vocab=False)

#Create dataFields and datasets only for the training and validation sets
#As the columns/dataFields differ from the test set
dataFields = [("id", None), ("comment_text", textField), ("toxic", labelField), ("severe_toxic", labelField), 
              ("obscene", labelField), ("threat", labelField), ("insult", labelField),("identity_hate", labelField)]

trainSet = data.TabularDataset(path="data/train.csv"
                                              , format="csv", skip_header=True, fields=dataFields)

testSet = data.TabularDataset(path="data/test.csv", format='csv', 
                              skip_header=True, fields=dataFields)

#Build the Vocabulary
textField.build_vocab(trainSet, testSet)

#Create Iterators
#Bucket iterator creates Similar length batches by adding padding
trainIter = data.BucketIterator((trainSet), batch_size=64, 
                                                device=torch.device("cuda:0"), sort_key= lambda x: len(x.comment_text)
                                                ,sort_within_batch=False, repeat=False)

testIter = data.BucketIterator(testSet, batch_size=64, device=torch.device("cuda:0"),
                               sort_within_batch=False, repeat=False)

train_dl = BatchWrapper(trainIter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

test_dl = BatchWrapper(testIter, "comment_text", ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

#Initialize model
embeddedSize = 100
nhidden = 500
numLinear = 3

model = SimpleLSTM(nhidden, emb_dim=embeddedSize, textField=textField, num_linear=numLinear)
model = model.cuda();

#Initialize loss and optimizer functions
opt = optim.Adam(model.parameters(), lr=1e-8)
loss_func = nn.BCEWithLogitsLoss()
epochs = 10

#Train and save
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train() 

    for x, y in tqdm.tqdm(train_dl): 
        opt.zero_grad()
        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()
        running_loss += loss.data.item() * x.size(0)
#        print(model.embedding.weight)

    
    epoch_loss = running_loss / len(trainSet)
    val_loss = 0.0 


    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    
torch.save(model.state_dict(), 'ToxicCommentModeDeepl.pt')
print(model)

