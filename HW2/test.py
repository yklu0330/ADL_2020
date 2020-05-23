#!/usr/bin/env python
# coding: utf-8

from transformers.modeling_bert import BertPreTrainedModel
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import BertForSequenceClassification
import sys


PRETRAINED_MODEL_NAME = "bert-base-chinese"

tokenizer = BertTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME, do_lower_case=True)


class Paragraph():
    def __init__(self, contextId, ques):
        self.contextId = contextId
        self.ques = ques

    def getVal(self):
        return {
            'contextId': self.contextId,
            'ques': self.ques,
        }


contextDic = {}
pDictList = []
qIdList = []
with open(sys.argv[1], 'r') as testFile:
    dic = json.load(testFile)
    dataNum = len(dic['data'])
    for i in tqdm(range(dataNum)):
        paraNum = len(dic['data'][i]['paragraphs'])
        for j in range(paraNum):
            paragraph = dic['data'][i]['paragraphs'][j]
            id = paragraph['id']
            context = paragraph['context']
            contextDic[id] = context

            for k in range(len(paragraph['qas'])):
                qa = paragraph['qas'][k]
                qIdList.append(qa['id'])
                ques = qa['question']
                pDict = Paragraph(id, ques)
                pDictList.append(pDict)

MAX_QUE_LEN = 60


class paragraphDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        para = self.data[index].getVal()
        context = contextDic[para['contextId']]
        cont_tokens = self.tokenizer.tokenize(context)

        ques = para['ques']
        ques_tokens = self.tokenizer.tokenize(ques)
        if len(ques_tokens) > MAX_QUE_LEN - 1:
            ques_tokens = ques_tokens[:MAX_QUE_LEN - 1]

        if len(cont_tokens) > 512 - 3 - len(ques_tokens):
            end = 509 - len(ques_tokens)
            cont_tokens = cont_tokens[:end]

        input_tokens = ["[CLS]"] + cont_tokens + \
            ["[SEP]"] + ques_tokens + ["[SEP]"]
        tokenIds = self.tokenizer.convert_tokens_to_ids(input_tokens)

        token = torch.tensor(tokenIds)
        tokType = [0] * (len(cont_tokens) + 2) + [1] * (len(ques_tokens) + 1)
        tokType = torch.tensor(tokType)

        contEnd = torch.tensor([len(cont_tokens) + 1])

        return {
            'token': token,
            'tokType': tokType,
            'contEnd': contEnd
        }

    def collate_fn(self, samples):
        token = [s['token'] for s in samples]
        tokType = [s['tokType'] for s in samples]
        contEnd = [s['contEnd'] for s in samples]

        token = pad_sequence(token, batch_first=True)
        tokType = pad_sequence(tokType, batch_first=True)
        mask = torch.zeros(token.shape, dtype=torch.long)
        mask = mask.masked_fill(token != 0, 1)
        contEnd = torch.tensor(contEnd)

        return {
            'token': token,
            'tokType': tokType,
            'mask': mask,
            'contEnd': contEnd
        }


BATCH_SIZE = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testSet = paragraphDataset(pDictList, tokenizer)
testLoader = DataLoader(testSet, BATCH_SIZE, shuffle=False,
                        collate_fn=testSet.collate_fn)


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.answerable = nn.Linear(config.hidden_size, 1)
        self.startClassfier = nn.Linear(config.hidden_size, 1)
        self.endClassifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        answerable = self.answerable(outputs[1])
        start = self.startClassfier(outputs[0]).squeeze(2)
        end = self.endClassifier(outputs[0]).squeeze(2)

        return answerable, start, end


NUM_LABELS = 1
model = BertForQuestionAnswering.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS).to(device)
ckpt = torch.load('ckpt20.0.pt')
model.load_state_dict(ckpt['state_dict'])
model.eval()

sigmoid = nn.Sigmoid()
outputDic = {}
with torch.no_grad():
    lossSum = 0
    for step, data in enumerate(tqdm(testLoader)):
        token = data['token'].to(device)
        tokType = data['tokType'].to(device)
        mask = data['mask'].to(device)
        contEnd = data['contEnd'].to(device)

        outAble, outStart, outEnd = model(
            input_ids=token, token_type_ids=tokType, attention_mask=mask)

        outAble = outAble.view(-1, 1)

        for i in range(outStart.shape[0]):
            outStart[i][0] = -float('inf')
            for j in range(contEnd[i], outStart.shape[1]):
                outStart[i][j] = -float('inf')

        for i in range(outEnd.shape[0]):
            outEnd[i][0] = -float('inf')
            for j in range(contEnd[i], outEnd.shape[1]):
                outEnd[i][j] = -float('inf')

        ableSig = sigmoid(outAble)

        for i in range(ableSig.shape[0]):
            if ableSig[i] > 0.6:
                endIdx = contEnd[i].item()
                startMask = outStart[i][:endIdx]
                endMask = outEnd[i][:endIdx]

                predStart = torch.max(startMask, 0)[1]
                predEnd = torch.max(endMask, 0)[1]
                if predEnd - predStart > 30 or predStart > predEnd:
                    outputDic[qIdList[step * BATCH_SIZE + i]] = ""
                    continue

                predAnsId = token[i][predStart:predEnd]
                predAnsList = tokenizer.convert_ids_to_tokens(
                    predAnsId.tolist())
                if '[UNK]' in predAnsList:
                    predAnsList.remove('[UNK]')
                predAns = "".join(predAnsList)
                predAns = predAns.replace("#", "")
                outputDic[qIdList[step * BATCH_SIZE + i]] = predAns

            else:
                outputDic[qIdList[step * BATCH_SIZE + i]] = ""

    json.dump(outputDic, open(sys.argv[2], "w"))
