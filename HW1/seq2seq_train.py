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

class Encoder(nn.Module):
    def __init__(self,
                 embedding_path,
                 embed_size,
                 rnn_hidden_size,
                 layer_num) -> None:
        super(Encoder, self).__init__()
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.LSTM(embed_size, rnn_hidden_size, layer_num, batch_first=True, bidirectional=True)
        self.linear=nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)

    def forward(self, idxs) -> Tuple[torch.tensor, torch.tensor]:
        embed = self.embedding(idxs)
        output, (hidden, cell) = self.rnn(embed)
        catHid = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        linHid = self.linear(catHid)
        outHid = torch.tanh(linHid).unsqueeze(0)
        catCell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)
        linCell = self.linear(catCell)
        outCell = torch.tanh(linCell).unsqueeze(0)
        return outHid, outCell

class Decoder(nn.Module):
    def __init__(self,
                 embedding_path,
                 embed_size,
                 rnn_hidden_size,
                 layer_num, 
                 output_dim) -> None:
        super(Decoder, self).__init__()
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.LSTM(embed_size, rnn_hidden_size, layer_num, batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, output_dim) #####
    def forward(self, idxs, hidden, cell) -> Tuple[torch.tensor, torch.tensor]:
        idxs = idxs.unsqueeze(1)
        embed = self.embedding(idxs)
        output, (hidden, cell) = self.rnn(embed, (hidden, cell))
        output = self.linear(output)
        output = output.squeeze(1)
        return output, hidden, cell
    
    
parser = ArgumentParser()
parser.add_argument('--batch_size')
parser.add_argument('--learn_rate')

args = parser.parse_args()

BATCH_SIZE = int(args.batch_size) ## 10
LEARN_RATE = float(args.learn_rate) ## 0.001

with open('embedding2.pkl', 'rb') as f1:
	embedDs = pickle.load(f1)

with open('train.pkl', 'rb') as f:
	trainDS = pickle.load(f)

with open('valid.pkl', 'rb') as f:
    validDS = pickle.load(f)
validloader = DataLoader(validDS, 1, shuffle=False, collate_fn=validDS.collate_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(trainDS, BATCH_SIZE, shuffle=True, collate_fn=trainDS.collate_fn)
encoder = Encoder('embedding2.pkl', 300, 128, 1).to(device)
decoder = Decoder('embedding2.pkl', 300, 128, 1, len(embedDs.vocab)).to(device)
teacher_forcing_ratio = 1
loss_func = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': LEARN_RATE}, 
    {'params': decoder.parameters(), 'lr': LEARN_RATE}
    ])
for epoch in range(3):
    lossSum = 0
    for step, data in enumerate(tqdm(loader)):
        batchInput = data['text'].to(device)
        batchSize = batchInput.shape[0]
        optimizer.zero_grad()

        hid, cell = encoder(batchInput)

        target = data['summary'].to(device)
        targetLen = data['len_summary']
        targetMaxLen = target.shape[1]
        decInput = target[:, 0].to(device)
        outputTs = torch.zeros(batchSize, targetMaxLen, len(embedDs.vocab)).to(device)
        for t in range(1, targetMaxLen):
            output, hid, cell = decoder(decInput, hid, cell)
            softmax = F.softmax(output, dim=1)
            maxVal, maxIdx = torch.max(softmax, 1)

            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                decInput = target[:, t].to(device)
            else:
                decInput = maxIdx.to(device)
            outputTs[:, t, :] = output

        new_target = target
        new_outputTs = outputTs

        new_target = new_target.reshape(-1, 1).squeeze(1)
        new_outputTs = new_outputTs.reshape(-1, new_outputTs.shape[2])

        loss = loss_func(new_outputTs, new_target)
        loss.backward()
        lossSum += loss

        optimizer.step()
   

    checkpoint_path = f'ckpt_a.{epoch}.pt'
    torch.save(
        {
            'enc_state_dict': encoder.state_dict(),
            'dec_state_dict': decoder.state_dict(),
            'epoch': epoch,
        },
        checkpoint_path
    )

    print("epoch: %d, loss: %f" %(epoch, lossSum / len(loader)))


