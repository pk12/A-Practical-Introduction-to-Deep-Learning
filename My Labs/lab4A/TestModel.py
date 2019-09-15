# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:42:41 2019

@author: Oppai
"""
###FOR THIS EXAMPLE TO RUN YOU NEED TO SET THE CODE IN iterator.py

import torch as torch
import torch.optim as optim
from torchtext.data import dataset
from torchtext import data
from torchtext import datasets
import spacy
from spacy.symbols import ORTH
from tqdm import tqdm
from torch import nn as nn
from LSTMModel import RNN_GRUModel
import numpy as np

my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
my_tok.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]



#Function to convert text id to text
def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)

batchSize = 1

    
# Other Stuff    
TEST_TEXT = data.Field(lower=True, tokenize=spacy_tok)
trainSet, valid, test = datasets.WikiText2.splits(TEST_TEXT)
myTestSet = datasets.LanguageModelingDataset(path="F:\Πτυχιακη\Lab4A\.data\wikitext-2\wikitext-2\TestMine.tokens", text_field=TEST_TEXT)


#The Vocabulary is constructed from the dataset so we need to load the large one for more Variety
TEST_TEXT.build_vocab(trainSet, vectors="glove.6B.200d")
myWeight_matrix = TEST_TEXT.vocab.vectors

#Create  Iterator over my input
myTestIter = data.BPTTIterator(dataset=myTestSet, batch_size=batchSize,
    bptt_len=1,
    device=torch.device("cuda:0"),
    repeat=False)

    
#Load model
#model = RNNModel(myWeight_matrix.size(0), myWeight_matrix.size()[1], 200, 1, 1)

#Model for special case in tokenizer
model = RNN_GRUModel(28869, myWeight_matrix.size()[1], 201, 1, batchSize)

##Model for No special case in tokenizer
#model = RNNModel(28870, myWeight_matrix.size()[1], 200, 1, bsz=1)
#
model.encoder.weight.data.copy_(myWeight_matrix)


# Comment if no model exists
#Model for custom tokens
#model.load_state_dict(torch.load('F:\Πτυχιακη\Lab4A\model_LSTM_trained.pt'))

#Model without custom tokens
model.load_state_dict(torch.load('F:\Πτυχιακη\Lab4A\model_LSTM_trainedWithSpacy.pt'))

model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))
n_tokens = myWeight_matrix.size(0)

def test():
    model.eval()
#    val_loss = 0
    for batch in myTestIter:
        model.reset_history()
        text, targets = batch.text, batch.target
        arrs = model(text).cpu().data.numpy()
#        loss = criterion(prediction.view(-1, 57738), targets.view(-1, targets.size(0)))
#        val_loss += loss.data.item() * text.size()[0]
#    val_loss /= len(myTestSet.examples[0].text)
        result = word_ids_to_sentence(np.argmax(arrs, axis=2), TEST_TEXT.vocab, join=' ')[:210]
        print(result)
       
    
test()   
    
    
    
    
    
    