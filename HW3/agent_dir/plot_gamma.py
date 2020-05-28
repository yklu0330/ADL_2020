import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('../rewards_gamma_0.99.pkl', 'rb') as f1:
    reward1 = pickle.load(f1)

with open('../rewards_gamma_0.6.pkl', 'rb') as f2:
    reward2 = pickle.load(f2)

with open('../rewards_gamma_0.3.pkl', 'rb') as f3:
    reward3 = pickle.load(f3)

with open('../rewards_gamma_0.1.pkl', 'rb') as f4:
    reward4 = pickle.load(f4)

episodes = [i for i in range(0, 1510, 10)]


plt.plot(episodes, reward1, '.-', color='g', label="gamma = 0.99")
plt.plot(episodes, reward2, '.-', color='b', label="gamma = 0.6")
plt.plot(episodes, reward3, '.-', color='r', label="gamma = 0.3")
plt.plot(episodes, reward4, '.-', color='yellow', label="gamma = 0.1")
plt.legend(loc='best')
plt.title("Learning Curve on Different Gamma")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.savefig('learning_curve_problem2.png')
