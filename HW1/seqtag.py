#!/usr/bin/env python
# coding: utf-8

# In[21]:


import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset2 import Seq2SeqDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Tuple, Dict
import torch.nn.functional as F
import random
from argparse import ArgumentParser

class Model(nn.Module):
    def __init__(self,
                 embedding_path,
                 embed_size,
                 rnn_hidden_size,
                 layer_num) -> None:
        super(Model, self).__init__()
        with open(embedding_path, 'rb') as f:
          	embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.LSTM(embed_size, rnn_hidden_size, layer_num, batch_first=True, bidirectional=True)
        self.linear=nn.Linear(rnn_hidden_size * 2, 1)
        # init a LSTM/RNN

    def forward(self, idxs) -> Tuple[torch.tensor, torch.tensor]:
        embed = self.embedding(idxs)
        output, state = self.rnn(embed)
        output = self.linear(output)
        return output, state


# In[14]:

parser = ArgumentParser()
parser.add_argument('--batch_size')
parser.add_argument('--learn_rate')

args = parser.parse_args()

BATCH_SIZE = int(args.batch_size) ## 16
LEARN_RATE = float(args.learn_rate) ## 0.001

with open('train.pkl', 'rb') as f:
    trainDS = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(trainDS, BATCH_SIZE, shuffle=False, collate_fn=trainDS.collate_fn)

# In[22]:


model = Model('embedding.pkl', 300, 128, 2).to(device)
loss_fnc = nn.BCEWithLogitsLoss(
    reduction='none', 
    pos_weight=torch.tensor(7))
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


# In[52]:


for epoch in range(5):
    lossSum = 0
    for step, data in enumerate(tqdm(loader)):
        batchInput = data['text'].to(device)
        
        optimizer.zero_grad()
        
        output, state = model(batchInput)
        output = torch.squeeze(output)

        target = data['label']
               
        output = output.view(-1, 1).float().to(device)
        target = target.view(-1, 1).float().to(device)
        
        mask = torch.where(target > -100, torch.full_like(target, 1), torch.full_like(target, 0))
        mask = mask.type(torch.BoolTensor).to(device)
        output = torch.masked_select(output, mask)
        target = torch.masked_select(target, mask)
                
        loss = loss_fnc(output, target)

        loss = torch.mean(loss)
        loss.backward()                
                
        optimizer.step()
        lossSum += loss
        
    checkpoint_path = f'ckpt1.{epoch}.pt'
    torch.save(
        {
            'state_dict': model.state_dict(),
            'epoch': epoch,
        },
        checkpoint_path
    )
    print("epoch: %d, loss: %f" %(epoch, lossSum / len(loader)))
          