# 导入必要的库
from collections import deque, namedtuple
import random

import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn.functional import smooth_l1_loss

LEARNING_RATE = 0.05
GAMMA = 0.995
NUM_EPISODES = 50000
RENDER = True

BATCH_SIZE = 64

env = gym.make('MountainCar-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

num_state = env.observation_space.shape[0]
num_action = env.action_space.n

eps = np.finfo(np.float32).eps.item()
plt.ion()

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(capacity=10000)

saveAction = namedtuple('SavedActions', ['probs', 'action_values'])

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)
        self.value_head = nn.Linear(128, 1)
        self.policy_action_value = []
        self.rewards = []
        self.gamma = GAMMA

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.action_head(x), dim=0)

        value = self.value_head(x)
        return probs, value

policy = Module()

optimizer = Adam(policy.parameters(), lr=LEARNING_RATE)

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('ACER Time')
    ax.set_xlabel('time')
    ax.set_ylabel('ep')
    ax.plot(steps)
    RunTime = len(steps)
    path = './ACER_MountainCar-v0/' + 'RunTime' + str(RunTime) + '.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, value = policy(state)
    c = Categorical(probs)
    action = c.sample()
    log_prob = c.log_prob(action)

    policy.policy_action_value.append(saveAction(log_prob, value))
    action = action.item()
    return action

def finish_episode():
    rewards = []
    saveActions = policy.policy_action_value
    policy_loss = []

    R = 0
    for r in policy.rewards[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob, value), r in zip(saveActions, rewards):
        reward = r - value.item()
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.policy_action_value[:]


def collect_data(state, action, reward, next_state):
    memory.push(state, action, reward, next_state)


def main():
    run_steps = []
    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        if RENDER:
            env.render()

        for t in count():
            action = select_action(state)
            next_state, reward, done = env.step(action)
            reward = next_state[0] + reward
            if RENDER:
                env.render()

            collect_data(state, action, reward, next_state)

            state = next_state

            if done:
                run_steps.append(t)
                break

        if len(memory) > BATCH_SIZE:
            train_model()

        finish_episode()
        plot(run_steps)

        if i_episode % 100 == 0 and i_episode != 0:
            modelPath = './ACER_MountainCar-v0_time/ModelTraing' + str(i_episode) + 'Times.pkl'
            torch.save(policy, modelPath)

def train_model():
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(batch.state, dtype=torch.float32)
    action_batch = torch.tensor(batch.action, dtype=torch.int64)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)

    probs_batch, value_batch = policy(state_batch)
    c = Categorical(probs_batch)
    log_prob_batch = c.log_prob(action_batch)

    value_loss = smooth_l1_loss(value_batch.squeeze(), reward_batch)

    advantage = reward_batch - value_batch.squeeze()
    policy_loss = -log_prob_batch * advantage.detach()

    loss = policy_loss.mean() + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    main()
