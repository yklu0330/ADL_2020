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


PRETRAINED_MODEL_NAME = "bert-base-chinese"

tokenizer = BertTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME, do_lower_case=True)


class Paragraph():
    def __init__(self, contextId, ques, ans_start, ans_end, ans_able):
        self.contextId = contextId
        self.ques = ques
        self.ans_start = ans_start
        self.ans_end = ans_end
        self.ans_able = ans_able

    def getVal(self):
        return {
            'contextId': self.contextId,
            'ques': self.ques,
            'ans_start': self.ans_start,
            'ans_end': self.ans_end,
            'ans_able': self.ans_able
        }


contextDic = {}
pDictList = []

with open('train.json', 'r') as trainFile:
    dic = json.load(trainFile)
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
                ans = qa['answers'][0]
                ques = qa['question']
                if qa['answerable'] == True:
                    start = ans['answer_start']
                    ansStart = len(tokenizer.tokenize(context[:start]))
                    ansTok = tokenizer.tokenize(ans['text'])
                    ansEnd = ansStart + len(ansTok)
                    pDict = Paragraph(id, ques, ansStart+1,
                                      ansEnd+1, qa['answerable'])
                else:
                    pDict = Paragraph(id, ques, -1, -1, qa['answerable'])
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
        label = torch.tensor([1 if para['ans_able'] else 0])

        start = torch.tensor([para['ans_start']])
        end = torch.tensor([para['ans_end']])

        contEnd = torch.tensor([len(cont_tokens) + 1])

        return {
            'token': token,
            'tokType': tokType,
            'label': label,
            'start': start,
            'end': end,
            'contEnd': contEnd
        }

    def collate_fn(self, samples):
        token = [s['token'] for s in samples]
        tokType = [s['tokType'] for s in samples]
        label = [s['label'] for s in samples]
        start = [s['start'] for s in samples]
        end = [s['end'] for s in samples]
        contEnd = [s['contEnd'] for s in samples]

        token = pad_sequence(token, batch_first=True)
        tokType = pad_sequence(tokType, batch_first=True)
        mask = torch.zeros(token.shape, dtype=torch.long)
        mask = mask.masked_fill(token != 0, 1)
        label = torch.tensor(label)
        start = torch.tensor(start)
        end = torch.tensor(end)
        contEnd = torch.tensor(contEnd)

        return {
            'token': token,
            'tokType': tokType,
            'mask': mask,
            'label': label,
            'start': start,
            'end': end,
            'contEnd': contEnd
        }


BATCH_SIZE = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainSet = paragraphDataset(pDictList, tokenizer)
trainLoader = DataLoader(trainSet, BATCH_SIZE,
                         shuffle=True, collate_fn=trainSet.collate_fn)


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

loss_fnc1 = nn.BCEWithLogitsLoss()
loss_fnc2 = nn.CrossEntropyLoss(ignore_index=-1)
loss_fnc3 = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=9e-6)
sigmoid = nn.Sigmoid()

for epoch in range(1):
    lossSum = 0
    for step, data in enumerate(tqdm(trainLoader)):
        optimizer.zero_grad()

        token = data['token'].to(device)
        tokType = data['tokType'].to(device)
        mask = data['mask'].to(device)
        label = data['label'].float().to(device)
        start = data['start'].to(device)
        end = data['end'].to(device)
        contEnd = data['contEnd'].to(device)

        outAble, outStart, outEnd = model(
            input_ids=token, token_type_ids=tokType, attention_mask=mask)

        outAble = outAble.view(-1, 1)
        tarAble = label.view(-1, 1)
        tarStart = start
        tarEnd = end

        for i in range(outStart.shape[0]):
            outStart[i][0] = -float('inf')
            for j in range(contEnd[i], outStart.shape[1]):
                outStart[i][j] = -float('inf')

        for i in range(outEnd.shape[0]):
            outEnd[i][0] = -float('inf')
            for j in range(contEnd[i], outEnd.shape[1]):
                outEnd[i][j] = -float('inf')

        for i in range(tarStart.shape[0]):
            tarStart[i] = torch.where(
                tarStart[i] >= contEnd[i], torch.full_like(tarStart[i], -1), tarStart[i])
            tarEnd[i] = torch.where(
                tarEnd[i] >= contEnd[i], torch.full_like(tarEnd[i], -1), tarEnd[i])

        loss1 = loss_fnc1(outAble, tarAble)
        loss2 = loss_fnc2(outStart, tarStart)
        loss3 = loss_fnc3(outEnd, tarEnd)
        loss = loss1 + loss2 + loss3
        loss.backward()

        optimizer.step()

        lossSum += loss.item() / 3

    checkpoint_path = f'ckpt20.{epoch}.pt'
    torch.save(
        {
            'state_dict': model.state_dict(),
            'epoch': epoch,
        },
        checkpoint_path
    )
    print("epoch: %d, loss: %f" % (epoch, lossSum / len(trainLoader)))
