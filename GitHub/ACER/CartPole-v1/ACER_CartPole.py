import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import deque

env = gym.make('CartPole-v1')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

batch_size = 128
MEMORY_CAPACITY = 200000
learning_rate = 0.01
gamma = 0.99
episodes = 2000
render = False
eps = np.finfo(np.float32).eps.item()
NUM_STATES = env.observation_space.shape[0]
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 32)

        self.action_head = nn.Linear(32, action_space)
        self.value_head = nn.Linear(32, 1)

        self.save_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)

        return F.softmax(action_score, dim=-1), state_value


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class TargetNetwork:
    def __init__(self, target_net, source_net, tau=0.001):
        self.target_net = target_net
        self.source_net = source_net
        self.tau = tau

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


actor_target = Policy()
critic_target = Policy()

tau = 0.001

actor_target_net = TargetNetwork(actor_target, model, tau)
critic_target_net = TargetNetwork(critic_target, model, tau)

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # 存储优先级

    def add(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        priorities = np.array(self.priorities)
        prob = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=MEMORY_CAPACITY)


def plot_durations(episode_durations, average_length,average_reward):
    plt.ion()
    plt.figure(2)
    plt.clf()
    plt.ylim(0, 500)
    duration_t = torch.FloatTensor(episode_durations)
    duration_age = torch.FloatTensor(average_length)
    duration_average = torch.FloatTensor(average_reward)

    plt.title('ACER CartPole-v1 AVE')
    plt.xlabel('Episodes')
    plt.ylabel('Time Step')
    # plt.plot(duration_t.numpy())
    # plt.plot(duration_average.numpy())
    plt.plot(duration_age.numpy(),color='r')

    plt.pause(0.00001)
    RunTime = len(episode_durations)
    path = './ACER_CartPole-v1-text_reward/' + 'ACER_RunTime' + str(RunTime) + '.jpg'
    if len(episode_durations) % 200 == 0:
        plt.savefig(path)
    plt.pause(0.0000001)


def finish_episode(states, actions, rewards, next_states, dones):
    R = 0
    saved_actions = model.save_actions
    policy_loss = []
    value_loss = []

    returns = []
    for (log_prob, value), reward in zip(saved_actions, reversed(model.rewards)):
        R = reward + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)

    returns = (returns - returns.mean()) / (returns.std() + eps)

    for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        replay_buffer.add((state, action, reward, next_state, done), priority=abs(R))

    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    batch = np.array(batch)
    batch_state = torch.FloatTensor(batch[:, 0].tolist())
    batch_action = torch.LongTensor(batch[:, 1].tolist())
    batch_reward = torch.FloatTensor(batch[:, 2].tolist())

    probs, state_value = model(batch_state)
    m = Categorical(probs)
    action = batch_action
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))
    model.rewards.append(batch_reward)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    max_grad_norm = 0.7
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    actor_target_net.soft_update()
    critic_target_net.soft_update()

    del model.rewards[:]
    del model.save_actions[:]


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item()

def main():
    running_reward = 10
    live_time = []
    ac_age = []
    episode_rewards = []

    for i_episode in count(episodes):
        state = env.reset()
        episode_reward = 0
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for t in count():
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            episode_reward += reward
            model.rewards.append(reward)

            if done or t >= 500:
                break

        finish_episode(states, actions, rewards, next_states, dones)

        running_reward = running_reward * 0.99 + t * 0.01
        live_time.append(t)
        average_reward = np.mean(live_time)
        episode_rewards.append(average_reward)
        ac_age.append(running_reward)
        plot_durations(live_time, ac_age, episode_rewards)

if __name__ == '__main__':
    main()
