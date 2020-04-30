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

    def forward(self, idxs) -> Tuple[torch.tensor, torch.tensor]:
        embed = self.embedding(idxs)
        output, state = self.rnn(embed)
        output = self.linear(output)
        return output, state

parser = ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()

with open('valid_seqtag.pkl', 'rb') as f:
    validDS = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = DataLoader(validDS, 16, shuffle=False, collate_fn=validDS.collate_fn)

model = Model('embedding.pkl', 300, 128, 2).to(device)
sigmoid = nn.Sigmoid()
ckpt = torch.load('ckpt2.1.pt')
model.load_state_dict(ckpt['state_dict'])
model.eval()

re_loc = []
with open(args.output_path, "w") as optFile:
    lossSum = 0
    with torch.no_grad():
        for step, data in enumerate(tqdm(loader)):
            batchInput = data['text'].to(device)
            batchSize = batchInput.shape[0]
            if batchInput.shape[1] == 1:
                predDict = {}
                predDict["id"] = data['id'][0]
                predDict["predict_sentence_index"] = [0]
                predObj = json.dumps(predDict)
                optFile.write(predObj)
                optFile.write("\n")
                continue

            output, state = model(batchInput)
            output = torch.squeeze(output)
            
            outSig = sigmoid(output)
          
            for b in range(batchSize):
                predIdx = []
                senSum = []
                for i in range(len(data['sent_range'][b])):
                    start_bound = data['sent_range'][b][i][0]
                    end_bound = data['sent_range'][b][i][1]
                    if end_bound > 300:
                        break
                    tempSum = 0
                    for j in range(start_bound, end_bound):
                        tempSum += outSig[b][j]
                    senSum.append(tempSum / (end_bound - start_bound))
                predIdx.append(senSum.index(max(senSum)))
                predDict = {}
                predDict["id"] = data['id'][b]
                predDict["predict_sentence_index"] = predIdx
                predObj = json.dumps(predDict)
                optFile.write(predObj)
                optFile.write("\n")
                        
optFile.close()

