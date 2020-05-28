import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment

import talib as ta
import pickle

use_cuda = torch.cuda.is_available()


class DQN(nn.Module):
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class Transition():
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = torch.tensor([action], dtype=torch.float)
        self.next_state = next_state
        self.reward = torch.tensor([reward], dtype=torch.float)

        self.state = self.state.cuda() if use_cuda else self.state
        self.action = self.action.cuda() if use_cuda else self.action
        self.next_state = self.next_state.cuda() if use_cuda else self.next_state
        self.reward = self.reward.cuda() if use_cuda else self.reward

    def getItem(self):
        return {
            'state': self.state,
            'action': self.action,
            'next_state': self.next_state,
            'reward': self.reward
        }


class ReplayBuffer():
    def __init__(self, buf_size):
        self.buf_size = buf_size
        self.buffer = []
        self.bufIdx = 0

    def setItem(self, state, action, next_state, reward):
        if len(self.buffer) < self.buf_size:
            self.buffer.append(None)
        self.buffer[self.bufIdx] = Transition(
            state, action, next_state, reward)
        self.bufIdx = (self.bufIdx + 1) % self.buf_size

    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n

        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')

        # discounted reward
        self.GAMMA = 0.99

        # training hyperparameters
        self.train_freq = 4  # frequency to train the online network
        # before we start to update our network, we wait a few steps first to fill the replay.
        self.learning_start = 10000
        self.batch_size = 32
        self.num_timesteps = 3000000  # total training steps
        self.display_freq = 10  # frequency to display training progress
        self.save_freq = 2000  # frequency to save the model
        self.target_update_freq = 1000  # frequency to update target network
        self.buffer_size = 10000  # max size of replay buffer

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0  # num. of passed steps

        self.replayBuf = ReplayBuffer(self.buffer_size)

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(
                torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(
                torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(
                load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(
                load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        pass

    def make_action(self, state, test=False):
        # test
        if test == True:
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                state = state.cuda() if use_cuda else state
                predProb = self.online_net(state)
                action = torch.max(predProb, 1)[1].cpu().item()
                return action

        # train
        randProb = random.random()
        if self.steps > 1000:
            epsilon = 0.01
        elif self.steps > 500:
            epsilon = 0.03
        else:
            epsilon = 0.05
        if randProb > epsilon:
            with torch.no_grad():
                state = state.cuda() if use_cuda else state
                predProb = self.online_net(state)
                action = torch.max(predProb, 1)[1].cpu().item()
        else:
            action = self.env.action_space.sample()

        return action

    def update(self):
        sampleBuf = self.replayBuf.sample(self.batch_size)
        states = torch.tensor([])
        actions = torch.tensor([])
        next_states = torch.tensor([])
        rewards = torch.tensor([])

        states = states.cuda() if use_cuda else states
        actions = actions.cuda() if use_cuda else actions
        next_states = next_states.cuda() if use_cuda else next_states
        rewards = rewards.cuda() if use_cuda else rewards

        for sample in sampleBuf:
            states = torch.cat((states, sample.getItem()['state']), dim=0)
            actions = torch.cat((actions, sample.getItem()['action']), dim=0)
            next_states = torch.cat(
                (next_states, sample.getItem()['next_state']), dim=0)
            rewards = torch.cat((rewards, sample.getItem()['reward']), dim=0)

        Q = self.online_net(states)
        Qa = torch.zeros([Q.shape[0]]).cuda()
        for i in range(Q.shape[0]):
            Qa[i] = Q[i][int(actions[i])]

        # Double DQN
        nextActions = torch.max(self.online_net(next_states), 1)[1]
        nextQ = self.target_net(next_states).detach()
        nextQa = torch.zeros([nextQ.shape[0]]).cuda()
        for i in range(nextQ.shape[0]):
            nextQa[i] = nextQ[i][int(nextActions[i])]
        expectQa = rewards + self.GAMMA * nextQa
        Qa = Qa.cuda() if use_cuda else Qa
        expectQa = expectQa.cuda() if use_cuda else expectQa

        loss_fn = torch.nn.MSELoss().cuda()
        loss = loss_fn(Qa, expectQa)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0  # passed episodes
        total_reward = 0  # compute average reward
        loss = 0

        rewardAll, episodes = [], []
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0)

            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # process new state
                next_state = torch.from_numpy(
                    next_state).permute(2, 0, 1).unsqueeze(0)

                self.replayBuf.setItem(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(
                        self.online_net.state_dict())

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                rewardAll.append(total_reward / self.display_freq)
                episodes.append(episodes_done_num)
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f ' %
                      (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break

        rewardAll = np.array(rewardAll)
        rewardMA = ta.SMA(rewardAll, 5)
        with open('rewards_double_dqn.pkl', 'wb') as f:
            pickle.dump(rewardMA, f)
