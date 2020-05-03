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
``$ python3.6 preprocess_seq_tag_train.py /PATH_TO_CONFIG_FILE``  
``$ python3.6 seqtag.py --batch_size 10 --learn_rate 0.001``  
### 2. seq2seq
``$ python3.6 preprocess_seq2seq_train.py /PATH_TO_CONFIG_FILE``  
``$ python3.6 seq2seq_train.py --batch_size 10 --learn_rate 0.001``  

### 3. attention
``$ python3.6 preprocess_seq2seq_train.py /PATH_TO_CONFIG_FILE``  
``$ python3.6 attention_train.py --batch_size 10 --learn_rate 0.001``  

## How to predict
### 1. seq_tag
#### Sample input:
- A **jsonl** file  
``{"id": "3000000", "text": "The Snowdrop will have a new look next month designed by Sir Peter Blake, who created The Beatles' Sgt Peppers Lonely Hearts Club Band cover in 1967.\nIt will celebrate World War One ship designers who used the dazzle effect to try to avoid detection by the enemy.\nVisitors boarding the Snowdrop can learn more about the technique.\nIt was commissioned by arts festival Liverpool Biennial and Tate Liverpool.\nThe camouflage works by confusing the eye, making it difficult to estimate a target's range, speed and direction, said a gallery spokesman.\nArtist Norman Wilkinson was credited with inventing the technique with each ship's pattern making it difficult to recognise classes of ships.\nSir Peter, 83, one of the major figures of British pop art, has strong links with Liverpool and first visited the city during his National Service with the RAF.\n", "sent_bounds": [[0, 150], [150, 264], [264, 331], [331, 407], [407, 547], [547, 689], [689, 850]]}``

#### Sample output:
- A **jsonl** file  
``{"id": "3000000", "predict_sentence_index": [6]}``

### How to plot the figures in my report
#### Distribution of relative locations
``$ python3.6 seqtag_plot.py``  

#### Attention weights
``$ python3.6 attention_plot.py``  