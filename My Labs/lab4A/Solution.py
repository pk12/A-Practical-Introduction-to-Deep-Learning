from LSTMModel import RNN_GRUModel
from torch import nn as nn
import torch as torch
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import spacy
from spacy.symbols import ORTH
from tqdm import tqdm

my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
my_tok.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]

TEXT = data.Field(lower=True, tokenize=spacy_tok)

# Need SPLITS to get all 3 datasets
train, valid, test = datasets.WikiText2.splits(TEXT) # loading custom datasets requires passing in the field, but nothing else.

#Build Vocabulary
TEXT.build_vocab(train, vectors="glove.6B.200d")

#Build all Iterators  based on the datasets
train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
    (train, valid, test),
    batch_size=128,
    bptt_len=30, # this is where we specify the sequence length
    device=torch.device("cuda:0"),
    repeat=False)

weight_matrix = TEXT.vocab.vectors
model = RNN_GRUModel(weight_matrix.size(0), weight_matrix.size()[1], 201, 1, 128)
model.encoder.weight.data.copy_(weight_matrix)


# Comment if no model exists
#model.load_state_dict(torch.load('model_LSTM_trainedWithSpacy.pt'))

model = model.to(torch.device("cuda:0"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.99))

n_tokens = weight_matrix.size(0)

def train_epoch(epoch):
    """One epoch of a training loop"""
    epoch_loss = 0
    model.train()
    for batch in tqdm(train_iter):
        # reset the hidden state or else the model will try to backpropagate to the
        # beginning of the dataset, requiring lots of time and a lot of memory
        model.reset_history()
        
        optimizer.zero_grad()
        
        text, targets = batch.text, batch.target
        prediction = model(text)
        # pytorch currently only supports cross entropy loss for inputs of 2 or 4 dimensions.
        # we therefore flatten the predictions out across the batch axis so that it becomes
        # shape (batch_size * sequence_length, n_tokens)
        # in accordance to this, we reshape the targets to be
        # shape (batch_size * sequence_length)
        loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.data.item() * prediction.size()[0] * prediction.size()[1]

    epoch_loss /= len(train.examples[0].text)

    # monitor the loss
    val_loss = 0
    model.eval()
    for batch in valid_iter:
        model.reset_history()
        text, targets = batch.text, batch.target
        prediction = model(text)
        loss = criterion(prediction.view(-1, n_tokens), targets.view(-1))
        val_loss += loss.data.item() * text.size()[0]
    val_loss /= len(valid.examples[0].text)
    
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    
n_epochs = 30    
for epoch in range(1, n_epochs + 1):
    train_epoch(epoch)
    

torch.save(model.state_dict(), 'model_LSTM_trainedWithSpacy' + '.pt')