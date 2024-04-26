import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.distributions import Categorical
from collections import namedtuple

env = gym.make('MountainCar-v0')
env = env.unwrapped

env.seed(1)
torch.manual_seed(1)

num_state = env.observation_space.shape[0]
num_action = env.action_space.n

LEARNING_RATE = 0.05
GAMMA = 0.995
NUM_EPISODES = 50000
RENDER = True
BUFFER_CAPACITY = 200000
TAU = 0.001
BATCH_SIZE = 200000

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size = x.size(0)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float))
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.bmm(attention_weights, V)
        output = output.squeeze(1)

        return output

class ActorCriticModule(nn.Module):
    def __init__(self):
        super(ActorCriticModule, self).__init__()
        self.fc1 = nn.Linear(num_state, 32)
        self.actor_head = nn.Sequential(
            SelfAttention(32),
            nn.Linear(32, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, num_action)
        )
        self.critic_head = nn.Sequential(
            SelfAttention(32),
            nn.Linear(32, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.actor_head(x)
        state_value = self.critic_head(x)
        return F.softmax(action_scores, dim=-1), state_value


policy_net = ActorCriticModule()
target_net = ActorCriticModule()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience, priority):
        self.buffer.append((experience, priority))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        priorities = np.array([exp[1] for exp in self.buffer])
        prob = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=prob)
        batch = [self.buffer[idx][0] for idx in indices]
        return batch

    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

def select_action(state):
    state = torch.from_numpy(state).float()
    state = state.unsqueeze(0)
    probs, state_value = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    policy_net.policy_action_value.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy_net.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
    for (log_prob, value), R in zip(policy_net.policy_action_value, returns):
        advantage = R - value.item()
        policy_loss.append(-log_prob * advantage)
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    optimizer.step()

    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

    del policy_net.rewards[:]
    del policy_net.policy_action_value[:]

def plot(run_steps, episode):
    plt.figure(figsize=(6, 4))
    plt.plot(range(episode), run_steps)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    if (episode + 1) % 100 == 0:
        plt.savefig(f'./A2CPER_MountainCar-v0-text_reward/Duration_Episode_{episode}.jpg')

def main():
    run_steps = []
    max_position = -2
    for i_episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        start_time = time.time()
        for t in count():
            action = select_action(state)
            next_state, done, _ = env.step(action)
            if RENDER:
                env.render()

            position = next_state[0]
            max_position = max(max_position, position)
            if position > max_position:
                reward = 1
                replay_buffer.add((state, action, reward, next_state, done), reward)
                max_position = position
            else:
                reward = 0

            if done or position >= 0.5:
                run_steps.append(t)
                print("Episode {}, Duration = {}".format(i_episode, t))
                plot(run_steps, i_episode)
                break

            state = next_state

        batch = replay_buffer.sample(batch_size=BATCH_SIZE)

def update_policy_net(batch):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)

    with torch.no_grad():
        next_state_values = target_net(next_states)[1].squeeze().detach()
        expected_state_values = rewards + (1 - dones) * GAMMA * next_state_values

    action_probs, state_values = policy_net(states)
    state_values = state_values.squeeze()
    action_log_probs = torch.log(action_probs.gather(1, actions.view(-1, 1)).squeeze())

    advantages = expected_state_values - state_values
    policy_loss = -(action_log_probs * advantages.detach()).mean()

    value_loss = F.smooth_l1_loss(state_values, expected_state_values.detach())

    optimizer.zero_grad()
    optimizer.step()

    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)


if __name__ == '__main__':
    main()
