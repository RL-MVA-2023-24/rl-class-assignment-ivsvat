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
from sklearn.ensemble import HistGradientBoostingRegressor

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# ---------------------By Ivan SVATKO 2024---------------------

# All the training was done in the Jupyter notebook available in /src
# The notebook is purely informative and the results should be reproducible with the code below
# For completeness, a copy of the training loop is also given below
# 

class ProjectAgent:
    """
        A default wrapper slightly changed to use a pretrained model.
    """
    def __init__(self):
        self.model = None

    def act(self, observation, use_random=False):
        """"
            Act greedily with respect to outputs of Q function given by self.model
        """
        if use_random:
            return env.action_space.sample()
        Qsa =[]
        for a in range(env.action_space.n):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.model.predict(sa))
        a = np.argmax(Qsa)
        return a

    def save(self, path):
        _s = dump(self.model, os.path.join(os.getcwd(), "src/Q_saved"))


    def load(self):
        print(os.getcwd())
        self.model = load(os.path.join(os.getcwd(), "src/Q_best"))



def collect_samples(env, total_iter, disable_tqdm=False, print_done_states=False, Q_hat=None, eps=0.15):
    """
        samples trajectories from the environment
        if provided, uses an estimated Q function with eps-greedy policy
        otherwise picks a random action

        Adapted from the sampler used in the class
    """
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    current_R = []
    S2 = []
    D = []
    it = 0
    cumulative_rewards = []
    for _ in range(total_iter):
        if Q_hat is None:
            a = env.action_space.sample()
        else:
            u = np.random.rand(1)
            if u>eps:
                Qsa =[]
                for a in range(env.action_space.n):
                    sa = np.append(s,a).reshape(1, -1)
                    Qsa.append(Q_hat.predict(sa))
                a = np.argmax(Qsa)
            else:
                a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)

        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        current_R.append(r)
        D.append(done)
        it += 1
        if done or trunc:
            cumulative_rewards.append(np.sum(current_R))
            s, _ = env.reset()
            current_R = []
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D, np.mean(cumulative_rewards)


def trees_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, Q_start = None):
    """
        Adapted from the FQI loop from the class
    """
    nb_samples = S.shape[0]
    Qfunctions = Q_start
    SA = np.append(S,A,axis=1)
    for it in range(iterations):
        if Qfunctions is None:
             value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Qfunctions.predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = HistGradientBoostingRegressor()
        Q.fit(SA,value)
        Qfunctions = Q
    return Qfunctions


if __name__ == "__main__":
    """
        Training script for Fitted Q iterations with fast ensemble decision trees algorithm
        The idea is based on the suggested paper from Ernst et al. 2006

        The loop learns an approximation of Q with a progressively growing training set

        At each step we add 20*200 (s, a, s', r) samples using the last approximation of Q available

        For selected iterations we randomize the domain to promote generalisation.
        This proved to be an efficient strategy even with way fewer samples.
    """

    # Experiment hyperparameters
    N_stages = 20
    n_patients = 20
    shuffle_stages = [3, 5, 7, 11, 13, 15]
    Q_functions = [None]
    episode_length = 200 # total_samples = N_stages * n_patients * episode_length

    # Number of successive update of the regression for each training set
    fqi_iter = 100

    gamma = 0.98

    env = TimeLimit(
                env=HIVPatient(domain_randomization=False), max_episode_steps=episode_length
                )


    # We either start from scratch or load a dataset an a model
    # stage 0
    print("Collecting the first sample")
    S, A, R, S2, D, cum_rew = collect_samples(env, n_patients*episode_length, Q_hat=Q_functions[-1])
    sample = [S,A,R,S2,D]
    _s = dump(sample, f"samples\sample_{0}")
    print(f"Stage: {0} \t strategy: {Q_functions[-1]}_{0} \t reward: {cum_rew:e}")
    print("Fitting the Q function")

    Q_next = trees_fqi(S, A, R, S2, D, fqi_iter, 4, gamma, Q_start = Q_functions[-1])
    Q_functions.append(Q_next)
    _s = dump(Q_next, f"samples\Q{0}")

    # loading data and models
    # sample_saved = load("samples\Run_20p_200it_100upd\sample_19")
    # S, A, R, S2, D = sample_saved
    # Q_functions.append(load("samples\Run_20p_200it_100upd\Q19"))
    
    # number of comleted steps if using a pretrained model
    lag = 0

    for n in range(1, N_stages):
        if n in shuffle_stages:
            env = TimeLimit(
                env=HIVPatient(domain_randomization=True), max_episode_steps=episode_length
                )
        else:
            env = TimeLimit(
                env=HIVPatient(domain_randomization=False), max_episode_steps=episode_length
                )
        print(f"Sampling with: \t {Q_functions[-1]}")
        S_next,A_next,R_next,S2_next,D_next, cum_rew = collect_samples(env, n_patients*episode_length, Q_hat=Q_functions[-1])
        print(f"Stage: {n+lag} \t strategy: {Q_functions[-1]}_{n+lag} \t reward: {cum_rew:e}")
        S = np.vstack([S, S_next])
        A = np.vstack([A, A_next])
        R = np.hstack([R, R_next])
        S2 = np.vstack([S2, S2_next])
        D = np.hstack([D, D_next])
        sample = [S,A,R,S2,D]
            
        _s = dump(sample, f"samples\sample_{n+lag}")

        print(f"Fitting the Q function, sample size: \t {S.shape[0]}")

        Q_next = trees_fqi(S, A, R, S2, D, fqi_iter, 4, gamma, Q_start = Q_functions[-1])
        _s = dump(Q_next, f"samples\Q{n+lag}")
        Q_functions.append(Q_next)