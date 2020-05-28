import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment

import talib as ta
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob


class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim=self.env.observation_space.shape[0],
                               action_num=self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')

        # discounted reward
        self.gamma = 0.99

        # training hyperparameters
        # total training episodes (actually too large...)
        self.num_episodes = 100000
        self.display_freq = 10  # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.action_prob = []

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []
        self.action_prob = []

    def make_action(self, state, test=False):
        state = torch.tensor(state).unsqueeze(0)
        actions = self.model(state)
        m = torch.distributions.Categorical(actions)
        action = m.sample()
        self.action_prob.append(m.log_prob(action))
        return action.item()

    def update(self):
        R_i = 0
        reward = []
        for r_i in self.rewards[::-1]:
            R_i = r_i + self.gamma * R_i
            reward.insert(0, R_i)

        loss = 0
        for R_i, action_prob in zip(reward, self.action_prob):
            loss += -R_i * action_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        avg_reward = None
        rewardAll, epochs = [], []
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            if epoch % self.display_freq == 0:
                rewardAll.append(avg_reward)
                epochs.append(epoch)
                print('Epochs: %d/%d | Avg reward: %f ' %
                      (epoch, self.num_episodes, avg_reward))

            if epoch == 5000:
                break

        rewardAll = np.array(rewardAll)
        rewardMA = ta.SMA(rewardAll, 5)

        plt.plot(epochs, rewardMA, '.-', color='g', label="learning curve")
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.savefig('learning_curve_pg.png')
