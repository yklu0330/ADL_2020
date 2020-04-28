# ADL_HW1

**The preprocessing code files are provided by the TAs of this course, including preprocess_seq_tag_train.py, preprocess_seq_tag.py, preprocess_seq2seq_train.py, preprocess_seq2seq.py, dataset.py, dataset2.py, utils.py.**

### Install
This project uses [PyTorch](https://pytorch.org/), json, pickle. Go check them out if you don't have them locally installed.

``$ pip install torch torchvision``  
``$ pip install json``  
``$ pip install pickle``  

### How to train
Before training, you should have train.jsonl, valid.jsonl, test.jsonl in the same directory.
#### 1. seqtag
``$ python3.6 preprocess_seq_tag_train.py .``  
``$ python3.6 seqtag.py --batch_size 10 --learn_rate 0.001``  
#### 2. seq2seq
``$ python3.6 preprocess_seq2seq_train.py .``  
``$ python3.6 seq2seq_train.py --batch_size 10 --learn_rate 0.001``  

#### 3. attention
``$ python3.6 preprocess_seq2seq_train.py .``  
``$ python3.6 attention_train.py --batch_size 10 --learn_rate 0.001``  

### How to plot the figures in my report
#### Distribution of relative locations
``$ python3.6 seqtag_plot.py``  

#### Attention weights
``$ python3.6 attention_plot.py``  