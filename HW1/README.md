# ADL_HW1

**The preprocessing code files are provided by the TAs of this course, including preprocess_seq_tag_train.py, preprocess_seq_tag.py, preprocess_seq2seq_train.py, preprocess_seq2seq.py, dataset.py, dataset2.py, utils.py.**

## Install
This project uses [PyTorch](https://pytorch.org/), json, pickle. Go check them out if you don't have them locally installed.

``$ pip install torch torchvision``  
``$ pip install json``  
``$ pip install pickle``  

## How to train
Before training, you should have train.jsonl, valid.jsonl, test.jsonl in the same directory.
### 1. seq_tag
```
$ python3 preprocess_seq_tag_train.py /PATH_TO_CONFIG_FILE
$ python3 seqtag.py --batch_size 10 --learn_rate 0.001
```
### 2. seq2seq
```
$ python3 preprocess_seq2seq_train.py /PATH_TO_CONFIG_FILE
$ python3 seq2seq_train.py --batch_size 10 --learn_rate 0.001
```

### 3. attention
```
$ python3 preprocess_seq2seq_train.py /PATH_TO_CONFIG_FILE
$ python3 attention_train.py --batch_size 10 --learn_rate 0.001
```

## How to predict
### 1. seq_tag
#### Usage
```
usage: preprocess_seq_tag.py --valid_data_path test_data_path

positional arguments:
    test_data_path      Test file
```
```
usage: seqtag_eval.py --output_path output_path

positional arguments:
    output_path      Prediction result save file
```
#### Example
```
python3 preprocess_seq_tag.py --valid_data_path test.jsonl
python3 seqtag_eval.py --output_path ./output.jsonl
```

#### Sample input
- A **jsonl** file  
```
{
    "id": "3000000", 
    "text": "The Snowdrop will have a new look next month designed by Sir Peter Blake, who created The Beatles' Sgt Peppers Lonely 
    Hearts Club Band cover in 1967.\nIt will celebrate World War One ship designers who used the dazzle effect to try to avoid 
    detection by the enemy.\nVisitors boarding the Snowdrop can learn more about the technique.\nIt was commissioned by arts festival 
    Liverpool Biennial and Tate Liverpool.\nThe camouflage works by confusing the eye, making it difficult to estimate a target's 
    range, speed and direction, said a gallery spokesman.\nArtist Norman Wilkinson was credited with inventing the technique with each 
    ship's pattern making it difficult to recognise classes of ships.\nSir Peter, 83, one of the major figures of British pop art, has 
    strong links with Liverpool and first visited the city during his National Service with the RAF.\n", 
    "sent_bounds": [[0, 150], [150, 264], [264, 331], [331, 407], [407, 547], [547, 689], [689, 850]]
}
```

#### Sample output
- A **jsonl** file  
```
{
    "id": "3000000", "predict_sentence_index": [6]
}
```

### 2.seq2seq
#### Usage
```
usage: preprocess_seq2seq.py --valid_data_path test_data_path

positional arguments:
    test_data_path      Test file
```
```
usage: seq2seq_eval.py --output_path output_path

positional arguments:
    output_path      Prediction result save file
```
#### Example
```
python3 preprocess_seq2seq.py --valid_data_path test.jsonl
python3 seq2seq_eval.py --output_path ./output.jsonl
```

#### Sample input
- A **jsonl** file  
```
{
    "id": "3000006", 
    "text": "Malachy Goodman, 57, of Rockmore Road, Belfast, was remanded in custody until 28 November.\nMr Gibson, 28, was shot in his 
    stomach and thigh in an alley near Divis Tower on 24 October. He died in hospital.\nMr Goodman was also charged with  possession of 
    a firearm and ammunition with intent to endanger life, and having cannabis with intent to supply.\nPolice told Belfast Magistrates'
     Court they are still searching for both the gun used in the killing and a second suspect.\nA judge was told police were strongly 
    opposed to Mr Goodman being released on bail. A detective sergeant claimed witnesses in the case could be put at risk.\n\"The 
    suspect knows many of those who have made statements,\" he said.\n\"The firearm used remains outstanding and a second suspect 
    remains at large.\"\nIt was also revealed that the home of another person said to have been involved in the incident has been 
    attacked.\n\"Tensions remain extremely high in the community in relation to his murder,\" the detective added.\nDuring 
    cross-examination by a defence solicitor, he accepted that Mr Goodman was not picked out at an identification process.\nThe 
    solicitor also claimed a description given of the alleged killer failed to match his client and said that three different versions
     of events were provided to police.\n", 
    "sent_bounds": [[0, 91], [91, 205], [205, 351], [351, 474], [474, 631], [631, 700], [700, 778], [778, 893], [893, 991], [991, 1117], [1117, 1287]]
}
```

#### Sample output
- A **jsonl** file  
```
{"id": "3000006", "predict": "a man has been charged with murder after a man was stabbed in a house in west belfast . <unk> </s> "}
```

### 3. attention
#### Usage
```
usage: preprocess_seq2seq.py --valid_data_path test_data_path

positional arguments:
    test_data_path      Test file
```
```
usage: attention_eval.py --output_path output_path

positional arguments:
    output_path      Prediction result save file
```
#### Example
```
python3 preprocess_seq2seq.py --valid_data_path test.jsonl
python3 attention_eval.py --output_path ./output.jsonl
```

#### Sample input
- A **jsonl** file  
```
{
    "id": "3000006", 
    "text": "Malachy Goodman, 57, of Rockmore Road, Belfast, was remanded in custody until 28 November.\nMr Gibson, 28, was shot in his 
    stomach and thigh in an alley near Divis Tower on 24 October. He died in hospital.\nMr Goodman was also charged with  possession of 
    a firearm and ammunition with intent to endanger life, and having cannabis with intent to supply.\nPolice told Belfast Magistrates' 
    Court they are still searching for both the gun used in the killing and a second suspect.\nA judge was told police were strongly 
    opposed to Mr Goodman being released on bail. A detective sergeant claimed witnesses in the case could be put at risk.\n\"The 
    suspect knows many of those who have made statements,\" he said.\n\"The firearm used remains outstanding and a second suspect 
    remains at large.\"\nIt was also revealed that the home of another person said to have been involved in the incident has been 
    attacked.\n\"Tensions remain extremely high in the community in relation to his murder,\" the detective added.\nDuring 
    cross-examination by a defence solicitor, he accepted that Mr Goodman was not picked out at an identification process.\nThe 
    solicitor also claimed a description given of the alleged killer failed to match his client and said that three different versions 
    of events were provided to police.\n", 
    "sent_bounds": [[0, 91], [91, 205], [205, 351], [351, 474], [474, 631], [631, 700], [700, 778], [778, 893], [893, 991], [991, 1117], [1117, 1287]]
}
```

#### Sample output
- A **jsonl** file  
```
{
    "id": "3000006", "predict": "a man has appeared in court charged with the murder of a man who was shot in the head in belfast . <unk> </s> "
}
```

## How to plot the figures in my report
### Distribution of relative locations
``$ python3 seqtag_plot.py``  

### Attention weights
``$ python3 attention_plot.py``  