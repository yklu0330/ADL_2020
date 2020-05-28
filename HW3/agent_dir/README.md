# ADL_HW3

### Install
This project uses [PyTorch](https://pytorch.org/) for training and testing. Go check them out if you don't have them locally installed.

``$ pip3 install torch torchvision``

For plotting results, it uses **ta-lib**, **matplotlib**.
To install **ta-lib**, please download the .whl file by the [link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) first and type the following command.

``$ pip3 install <whl_file>``

### How to train

#### Training policy gradient:  
``$ python3 main.py --train_pg``

#### Training DQN:
``$ python3 main.py --train_dqn``

### How to plot the figures in my report

#### Learning curve of policy gradient:
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

#### Learning curve of DQN:
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

#### Learning curve of DQN & Double DQN & Dueling DQN:
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