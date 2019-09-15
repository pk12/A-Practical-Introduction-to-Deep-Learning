# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:45:12 2019

@author: Oppai
"""
from torchtext import data
import torch
from torch import optim
from torch import nn
from BatchWrapper import BatchWrapper
from Model import SimpleLSTM
import tqdm
import numpy as np
import pandas as pd

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

#Create dataFields and testSet for the Test Data
testDataFields = [("id", None), ("comment_text", textField)]

testSet = data.TabularDataset(path="data/test.csv", format='csv', 
                              skip_header=True, fields=testDataFields)

#Build the Vocabulary
textField.build_vocab(trainSet, testSet)


#Create Iterators
#Bucket iterator creates Similar length batches by adding padding
testIter = data.Iterator(testSet, batch_size=128, device=torch.device("cuda:0"), sort=False, sort_within_batch=False, repeat=False)


test_dl = BatchWrapper(testIter, "comment_text", None)

#Initialize model
embeddedSize = 100
nhidden = 500
numLinear = 3

model = SimpleLSTM(nhidden, emb_dim=embeddedSize, textField=textField)
model.load_state_dict(torch.load("ToxicCommentModel.pt"))
model = model.cuda();
model = model.eval();

#Test and write the predictions in the matrxi
test_preds = []
i = 0

for x, y in tqdm.tqdm(test_dl):
    
    preds = model(x)
    preds = preds.data.cpu().numpy()
    preds = 1 / (1 + np.exp(-preds))
    if i == 0:
        test_preds= np.array(preds)
        i = i + 1
    else:
        test_preds = np.append(test_preds,preds, axis=0)    
    
    
#test_preds = np.hstack(np.array(test_preds)[0])

#Write the predictions in a CSV file
df = pd.read_csv("data/test.csv")

for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    df[col] = test_preds[:, i]

df.drop("comment_text", axis=1).to_csv("submission_Deep.csv", index=False)