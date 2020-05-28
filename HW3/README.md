# ADL_HW3

## Install
This project uses [PyTorch](https://pytorch.org/) for training and testing. Go check them out if you don't have them locally installed.

``$ pip3 install torch torchvision``

For plotting results, it uses **ta-lib**, **matplotlib**.
To install **ta-lib**, please download the .whl file by the [link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) first and type the following command.

``$ pip3 install <whl_file>``

## How to train

### Training policy gradient:  
``$ python3 main.py --train_pg``

Sample output:
```
Epochs: 0/100000 | Avg reward: -373.311894
Epochs: 10/100000 | Avg reward: -250.650406
Epochs: 20/100000 | Avg reward: -196.623736
Epochs: 30/100000 | Avg reward: -182.458882
Epochs: 40/100000 | Avg reward: -136.781666
Epochs: 50/100000 | Avg reward: -150.711554
Epochs: 60/100000 | Avg reward: -132.935763
Epochs: 70/100000 | Avg reward: -147.397742
Epochs: 80/100000 | Avg reward: -128.801300
Epochs: 90/100000 | Avg reward: -164.116125
Epochs: 100/100000 | Avg reward: -217.895684
Epochs: 110/100000 | Avg reward: -244.482326
Epochs: 120/100000 | Avg reward: -212.503929
Epochs: 130/100000 | Avg reward: -178.566620
Epochs: 140/100000 | Avg reward: -199.046087
Epochs: 150/100000 | Avg reward: -178.804998
                    .
                    .
                    .
```

### Training DQN:
``$ python3 main.py --train_dqn``

Sample output:
```
Episode: 0 | Steps: 142/3000000 | Avg reward: 1.700000 | loss: 0.000000
Episode: 10 | Steps: 1606/3000000 | Avg reward: 8.100000 | loss: 0.000000
Episode: 20 | Steps: 2809/3000000 | Avg reward: 6.300000 | loss: 0.000000
Episode: 30 | Steps: 4350/3000000 | Avg reward: 8.400000 | loss: 0.000000
Episode: 40 | Steps: 5554/3000000 | Avg reward: 6.300000 | loss: 0.000000
Episode: 50 | Steps: 6755/3000000 | Avg reward: 6.300000 | loss: 0.000000
Episode: 60 | Steps: 8062/3000000 | Avg reward: 9.300000 | loss: 0.000000
Episode: 70 | Steps: 9296/3000000 | Avg reward: 6.500000 | loss: 0.000000
Episode: 80 | Steps: 10520/3000000 | Avg reward: 7.500000 | loss: 0.055227
Episode: 90 | Steps: 11964/3000000 | Avg reward: 10.000000 | loss: 0.053439
Episode: 100 | Steps: 13772/3000000 | Avg reward: 13.500000 | loss: 0.044024
Episode: 110 | Steps: 15505/3000000 | Avg reward: 14.100000 | loss: 0.018854
Episode: 120 | Steps: 17239/3000000 | Avg reward: 16.900000 | loss: 0.045332
Episode: 130 | Steps: 18951/3000000 | Avg reward: 16.400000 | loss: 0.085640
Episode: 140 | Steps: 20750/3000000 | Avg reward: 18.000000 | loss: 0.156334
                                    .
                                    .
                                    .
```

## How to test

### Testing policy gradient:
``$ python3 test.py --test_pg``

Sample output:
```
load model from pg.cpt
Run 30 episodes
Mean: 39.76524580010926
Median: 35.55891821542154
```

### Testing DQN:
``$ python3 test.py --test_DQN``

Sample output:
```
load model from dqn
Run 50 episodes
Mean: 458.8
Median: 100.0
```

## How to plot the figures in my report

### Learning curve of policy gradient:
Modify main.py from  
```
from agent_dir.agent_pg import AgentPG
```
to
```
from agent_dir.plot_agent_pg import AgentPG
```
and run the following command.  
``$ python3 main.py --train_pg``

### Learning curve of DQN:
Modify main.py from  
```
from agent_dir.agent_dqn import AgentDQN
```
to
```
from agent_dir.plot_agent_dqn import AgentDQN
```
and run the following command.  
``$ python3 main.py --train_dqn``

### Learning curve of DQN & Double DQN & Dueling DQN:
Modify main.py
```
from agent_dir.agent_dqn import AgentDQN
agent = AgentDQN(env, args)
agent.train()
```
to
```
from agent_dir.plot_agent_dqn import AgentDQN
agent = AgentDQN(env, args)
agent.train()

from agent_dir.plot_agent_double_dqn import AgentDQN
agent = AgentDQN(env, args)
agent.train()

from agent_dir.plot_agent_duel_dqn import AgentDQN
agent = AgentDQN(env, args)
agent.train()
```
and run the following command.  
```
$ python3 main.py --train_dqn
$ python3 agent_dir/plot_dqn_improve.py
```