from IPython import get_ipython

import pyprind

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


import pandas as pd
import numpy as np

from dataloader import *
from model import *
from train import *


PATH = get_ipython().run_line_magic('pwd', '')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
main_df = main_df.sample(n=main_df.shape[0])
main_df = main_df[["question_text", "target"]]

o_class = main_df.loc[main_df.target == 0, :]
l_class = main_df.loc[main_df.target == 1, :]

test_o = o_class.iloc[:10000, :]
test_l = l_class.iloc[:10000, :]

valid_o = o_class.iloc[10000:20000, :]
valid_l = l_class.iloc[10000:20000, :]

train_o = o_class.iloc[20000:, :]
train_l = l_class.iloc[20000:, :]

train = pd.concat([train_o, train_l], axis=0)
valid = pd.concat([valid_o, valid_l], axis=0)
test = pd.concat([test_o, test_l], axis=0)

get_ipython().system('mkdir inputs')

train.to_csv(os.path.join(PATH, "inputs/train.csv"), index=False)
test.to_csv(os.path.join(PATH, "inputs/test.csv"), index=False)
valid.to_csv(os.path.join(PATH, "inputs/valid.csv"), index=False)

del main_df, train, test, valid, train_l, train_o, test_l, test_o, valid_l,valid_o, o_class, l_class

dataset = CreateDataset(PATH)

train_iterator, valid_iterator, test_iterator = dataset.getData()

pretrained_embeddings = dataset.getEmbeddings()

input_dim = dataset.lengthVocab()[0]
embedding_dim = 300
hidden_dim = 374
output_dim = 2
num_layers = 2
batch_size = 32

model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)

model.embedding.weight.data = pretrained_embeddings.to(device)
class_weights = torch.tensor([1.0, 15.0]).to(device)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model = model.to(device)
criterion = criterion.to(device)




def binary_accuracy(preds, y):

    preds, ind= torch.max(F.softmax(preds, dim=-1), 1)
    correct = (ind == y).float()
    acc = correct.sum()/float(len(correct))
    return acc



def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    bar = pyprind.ProgBar(len(iterator), bar_char='█')
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.Text).squeeze(0)

        loss = criterion(predictions, batch.Label)

        acc = binary_accuracy(predictions, batch.Label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        bar.update()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        bar = pyprind.ProgBar(len(iterator), bar_char='█')
        for batch in iterator:

            predictions = model(batch.Text).squeeze(0)
            
            loss = criterion(predictions, batch.Label)
            
            acc = binary_accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            bar.update()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

epochs = 2

for epoch in range(epochs):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')



