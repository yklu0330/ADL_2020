import numpy as np
import pickle
import matplotlib.pyplot as plt
import talib as ta

with open('rewards_dqn.pkl', 'rb') as f1:
    reward1 = pickle.load(f1)
    reward1 = ta.SMA(reward1, 100)

with open('rewards_double_dqn.pkl', 'rb') as f2:
    reward2 = pickle.load(f2)
    reward2 = ta.SMA(reward2, 100)

with open('rewards_duel_dqn.pkl', 'rb') as f3:
    reward3 = pickle.load(f3)
    reward3 = ta.SMA(reward3, 100)

episodes1 = [i for i in range(0, len(reward1) * 10, 10)]
episodes2 = [i for i in range(0, len(reward2) * 10, 10)]
episodes3 = [i for i in range(0, len(reward3) * 10, 10)]

plt.plot(episodes1, reward1, '.-', color='yellow', label="DQN")
plt.plot(episodes2, reward2, '.-', color='b', label="Double DQN")
plt.plot(episodes3, reward3, '.-', color='r', label="Dueling DQN")

plt.legend(loc='best')
plt.title("Learning Curve on Different DQN")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.savefig('learning_curve_problem3.png')
