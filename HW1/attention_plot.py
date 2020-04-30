import argparse
import logging
import os
import json
import pickle
from pathlib import Path
from utils import Tokenizer, Embedding
from dataset import Seq2SeqDataset
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
        output = self.linear(output)
        catHid = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        linHid = self.linear(catHid)
        outHid = torch.tanh(linHid).unsqueeze(0)
        catCell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)
        linCell = self.linear(catCell)
        outCell = torch.tanh(linCell).unsqueeze(0)
        return output, outHid, outCell

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
        self.linear = nn.Linear(rnn_hidden_size * 2, output_dim)
    def forward(self, enc_out, idxs, hidden, cell) -> Tuple[torch.tensor, torch.tensor]:
        idxs = idxs.unsqueeze(1)
        embed = self.embedding(idxs)
        rnn_out, (hidden, cell) = self.rnn(embed, (hidden, cell))
        query = rnn_out.permute(0, 2, 1)
        attn_weight = torch.bmm(enc_out, query)
        attn_weight = F.softmax(attn_weight, dim=1)
        attn_weight2 = attn_weight.permute(0, 2, 1)
        context = torch.bmm(attn_weight2, enc_out)
        cat = torch.cat((context, rnn_out), dim=2)

        dec_out = self.linear(cat)
        dec_out = dec_out.squeeze(1)

        return dec_out, hidden, cell, attn_weight

with open('embedding2.pkl', 'rb') as f1:
    embedding = pickle.load(f1)
    embedding_weight = embedding.vectors

with open('valid.pkl', 'rb') as f:
    validDS = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(validDS, 1, shuffle=False, collate_fn=validDS.collate_fn)
encoder = Encoder('embedding2.pkl', 300, 128, 1).to(device)
decoder = Decoder('embedding2.pkl', 300, 128, 1, len(embedding.vocab)).to(device)

ckpt = torch.load('ckpt_a.5.pt')
encoder.load_state_dict(ckpt['enc_state_dict'])
encoder.eval()
decoder.load_state_dict(ckpt['dec_state_dict'])
decoder.eval()

loss_func = nn.CrossEntropyLoss(ignore_index=0)

lossSum = 0
targetMaxLen = 80
input_sentence = ""
output_words = ""
attentions = 0
with open('output4.jsonl', "w") as optFile:
    with torch.no_grad():
        for step, data in enumerate(tqdm(loader)):
            predSen = ""
            batchInput = data['text'].to(device)
            batchSize = batchInput.shape[0]

            enc_out, hid, cell = encoder(batchInput)

            decInput = torch.ones(batchSize)
            decInput = decInput.long().to(device)
            targetLen = data['len_summary']

            outputTs = torch.zeros(batchSize, targetMaxLen, len(embedding.vocab)).to(device)
            predIdx = torch.zeros(batchSize, targetMaxLen)
            attn_mat = torch.zeros(targetLen[0], batchInput.shape[1]).to(device)
            for t in range(1, targetMaxLen):
                dec_out, hid, cell, attn_weight = decoder(enc_out, decInput, hid, cell)
                if t <= targetLen[0]:
                    attn_weight = attn_weight.squeeze(dim=0)
                    attn_weight = attn_weight.squeeze(dim=1)
                    attn_mat[t-1] = attn_weight
                attentions = attn_mat
                softmax = F.softmax(dec_out, dim=1)
                maxVal, maxIdx = torch.max(softmax, 1)

                decInput = maxIdx.to(device)
                for i in range(batchSize):
                    predIdx[i][t] = decInput[i]
                outputTs[:, t, :] = dec_out
            for j in range(batchSize):
                predSen = ""
                for k in range(1, targetMaxLen):
                    wordIdx = int(predIdx[j][k].item())
                    if wordIdx == 2:
                        predSen += embedding.vocab[wordIdx] + " "
                        break
                    else:
                        predSen += embedding.vocab[wordIdx] + " "
                predDict = {}
                predDict["id"] = data['id'][j]
                predDict["predict"] = predSen
                predObj = json.dumps(predDict)
                optFile.write(predObj)
                optFile.write("\n")
            

            if step == 0:
                sentenceIdx = data['text']
                for i in range(sentenceIdx.shape[1]):
                    input_sentence += embedding.vocab[sentenceIdx[0][i]] + " "
                print(input_sentence)
                print(predSen)
                output_words = predSen
                break

        print("loss: %f" %(lossSum / len(loader)))

optFile.close()

attentions = attentions.permute(1, 0)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig = plt.figure(figsize=(30, 297))
ax = fig.add_subplot(111)
cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
fig.colorbar(cax)

ax.set_xticklabels([''] + output_words.split(' ') + ['<EOS>'], rotation=90)
ax.set_yticklabels([''] + input_sentence.split(' ') + ['<EOS>'])

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  
plt.show()


