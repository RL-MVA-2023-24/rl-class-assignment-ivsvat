from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
#import copy
import random
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
from joblib import dump, load
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
# class ProjectAgent:
#     def act(self, observation, use_random=False):
#         return 0

#     def save(self, path):
#         pass

#     def load(self):
#         pass

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity # capacity of the buffer
#         self.data = []
#         self.index = 0 # index of the next cell to be filled
#     def append(self, s, a, r, s_, d):
#         if len(self.data) < self.capacity:
#             self.data.append(None)
#         self.data[self.index] = (s, a, r, s_, d)
#         self.index = (self.index + 1) % self.capacity
#     def sample(self, batch_size):
#         batch = random.sample(self.data, batch_size)
#         return list(map(lambda x:torch.Tensor(np.array(x)), list(zip(*batch))))
#     def __len__(self):
#         return len(self.data)
    
# state_dim = env.observation_space.shape[0]
# n_action = env.action_space.n 
# nb_neurons=36
# DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),

#                           nn.ReLU(),
#                           nn.Linear(nb_neurons, nb_neurons),
#                           nn.ReLU(),
#                           nn.Linear(nb_neurons, nb_neurons),
#                           nn.ReLU(),
#                           nn.Linear(nb_neurons, n_action))

# class ProjectAgent:
#     def __init__(self):
#         self.model = DQN

#     def act(self, observation, use_random=False):
#         return torch.argmax(self.model(torch.FloatTensor(observation)).unsqueeze(0)).item()

#     def save(self, path):
#         torch.save(self.model.state_dict(), r"best.pt")
#     def load(self):
#         self.model.load_state_dict(torch.load(r"best.pt"))

class ProjectAgent:
    def __init__(self):
        self.model = None

    def act(self, observation, use_random=False):
        Qsa =[]
        for a in range(env.action_space.n):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.model.predict(sa))
        a = np.argmax(Qsa)
        return a

    def save(self, path):
        pass
        # torch.save(self.model.state_dict(), r"best.pt")
    def load(self):
        print(os.getcwd())
        self.model = load(os.path.join(os.getcwd(), "src/Q_best"))
        print(f"Using {self.model}")